import AVFoundation
import CryptoKit
import Foundation

/// Production embedding generation for text, images, audio, and video with the pinned
/// jina-embeddings-v5-omni-small contract.
///
/// Unlike the generic low-level embedders, this façade validates the complete bundle before use,
/// owns inference behind an actor, applies retrieval roles internally, and rejects inputs that the
/// model would otherwise truncate.
public actor OmniSmall {
    public enum Dimensions: Int, Codable, CaseIterable, Sendable {
        case d32 = 32
        case d64 = 64
        case d128 = 128
        case d256 = 256
        case d512 = 512
        case d1024 = 1024
    }

    public enum Input: Sendable {
        case text(String)
        case image(URL)
        case imageData(Data)
        case audio(URL)
        case audioData(Data)
        case video(URL)
        case videoData(Data)
    }

    /// Server video input profile: file decoding, frame sampling, and resizing. Video vectors
    /// from different recipes are not comparable even within one space.
    public static let videoRecipe =
        "assetreader-decode-order-ycbcr-tagged-or-bt601-2fps-min4-max32-hf-smart-resize-aa-bicubic-patch2048-v2"

    public nonisolated let dimensions: Dimensions
    public nonisolated let space: String

    private let artifactFingerprint: String
    private let backend: any OmniSmallBackend
    private nonisolated let spaces: [Dimensions: String]

    /// Load and validate a native-capacity Omni Small bundle. Validation covers the pinned model
    /// contract, actual compiled function shapes, required assets, and every declared checksum.
    public static func load(
        from bundleURL: URL,
        dimensions: Dimensions = .d1024
    ) async throws -> OmniSmall {
        try Task.checkCancellation()
        let loaded = try await OmniSmallBundleValidator.load(from: bundleURL, dimensions: dimensions)
        return OmniSmall(
            backend: OmniSmallProductionBackend(bundle: loaded.bundle),
            dimensions: dimensions,
            spaces: loaded.spaces,
            artifactFingerprint: loaded.artifactFingerprint)
    }

    init(
        backend: any OmniSmallBackend,
        dimensions: Dimensions,
        spaces: [Dimensions: String],
        artifactFingerprint: String
    ) {
        self.backend = backend
        self.dimensions = dimensions
        self.spaces = spaces
        self.space = spaces[dimensions] ?? ""
        self.artifactFingerprint = artifactFingerprint
    }

    /// Test/support initializer retained for fixed-dimension mock backends.
    init(
        backend: any OmniSmallBackend,
        dimensions: Dimensions,
        space: String,
        artifactFingerprint: String
    ) {
        self.init(
            backend: backend,
            dimensions: dimensions,
            spaces: [dimensions: space],
            artifactFingerprint: artifactFingerprint)
    }

    /// Semantic-space identity for any supported Matryoshka projection from this validated bundle.
    /// The Core ML graph emits one native representation; output dimensions are post-inference.
    public nonisolated func space(for dimensions: Dimensions) -> String {
        spaces[dimensions] ?? space
    }

    public func embedQuery(_ input: Input) async throws -> QueryEmbedding {
        try await embedQuery(input, dimensions: dimensions)
    }

    public func embedQuery(
        _ input: Input,
        dimensions targetDimensions: Dimensions
    ) async throws -> QueryEmbedding {
        let targetSpace = space(for: targetDimensions)
        let values = try await embedSingle(
            input, role: .query, dimensions: targetDimensions, space: targetSpace)
        return try QueryEmbedding(
            values: values,
            dimensions: targetDimensions,
            space: targetSpace,
            artifactFingerprint: artifactFingerprint)
    }

    public func embedQueries(_ inputs: [Input]) async throws -> [QueryEmbedding] {
        try await embedQueries(inputs, dimensions: dimensions)
    }

    public func embedQueries(
        _ inputs: [Input],
        dimensions targetDimensions: Dimensions
    ) async throws -> [QueryEmbedding] {
        let targetSpace = space(for: targetDimensions)
        return try await embedBatch(
            inputs, role: .query, dimensions: targetDimensions, space: targetSpace).map {
            try QueryEmbedding(
                values: $0,
                dimensions: targetDimensions,
                space: targetSpace,
                artifactFingerprint: artifactFingerprint)
        }
    }

    public func embedDocument(_ input: Input) async throws -> DocumentEmbedding {
        try await embedDocument(input, dimensions: dimensions)
    }

    public func embedDocument(
        _ input: Input,
        dimensions targetDimensions: Dimensions
    ) async throws -> DocumentEmbedding {
        let targetSpace = space(for: targetDimensions)
        let values = try await embedSingle(
            input, role: .document, dimensions: targetDimensions, space: targetSpace)
        return try DocumentEmbedding(
            values: values,
            dimensions: targetDimensions,
            space: targetSpace,
            artifactFingerprint: artifactFingerprint)
    }

    public func embedDocuments(_ inputs: [Input]) async throws -> [DocumentEmbedding] {
        try await embedDocuments(inputs, dimensions: dimensions)
    }

    public func embedDocuments(
        _ inputs: [Input],
        dimensions targetDimensions: Dimensions
    ) async throws -> [DocumentEmbedding] {
        let targetSpace = space(for: targetDimensions)
        return try await embedBatch(
            inputs, role: .document, dimensions: targetDimensions, space: targetSpace).map {
            try DocumentEmbedding(
                values: $0,
                dimensions: targetDimensions,
                space: targetSpace,
                artifactFingerprint: artifactFingerprint)
        }
    }

    /// Server SPI for rows already tokenized with their retrieval conditioning. This removes the
    /// daemon's otherwise-duplicate tokenizer pass and allows query/document rows to share a native
    /// text function once their semantic conditioning is already encoded into token IDs.
    @_spi(Server)
    public func embedConditionedTokenRows(
        _ tokenIDRows: [[Int32]],
        dimensions targetDimensions: Dimensions
    ) async throws -> [[Float]] {
        let targetSpace = space(for: targetDimensions)
        return try await executeConditionedTokenRows(
            tokenIDRows, dimensions: targetDimensions, space: targetSpace)
    }

    private func embedSingle(
        _ input: Input,
        role: OmniSmallRole,
        dimensions targetDimensions: Dimensions,
        space targetSpace: String
    ) async throws -> [Float] {
        do {
            try Task.checkCancellation()
            try validate(input)
            let values: [Float]
            switch input {
            case let .text(text):
                let prepared = try await backend.prepareTexts([text], role: role)
                guard let row = prepared.first else {
                    throw OmniSmallBackendError.failure(
                        "text backend prepared no rows for one input")
                }
                let rows = try await backend.embedTexts([row], dimensions: targetDimensions)
                guard rows.count == 1 else {
                    throw OmniSmallBackendError.failure(
                        "text backend returned \(rows.count) rows for one input")
                }
                values = rows[0]
            case .image, .imageData, .audio, .audioData, .video, .videoData:
                values = try await backend.embedMedia(
                    input, role: role, dimensions: targetDimensions)
            }
            try Task.checkCancellation()
            return try OmniSmallVectorValidator.validated(
                values,
                dimensions: targetDimensions,
                space: targetSpace,
                artifactFingerprint: artifactFingerprint)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OmniSmallError {
            throw error
        } catch let error as OmniSmallBackendError {
            switch error {
            case let .item(_, reason), let .invalidInput(reason):
                throw OmniSmallError.invalidInput(reason)
            case let .failure(reason):
                throw OmniSmallError.inferenceFailed(reason)
            }
        } catch {
            throw OmniSmallError.inferenceFailed(String(describing: error))
        }
    }

    /// All-or-nothing batch embedding in two phases. Phase 1 validates every input and tokenizes
    /// every text before any inference runs: a failure anywhere throws with the failing input's
    /// original index and no backend encode call was made. Phase 2 executes the prepared batch —
    /// text in chunks of at most 64 rows per encode, media serially — preserving input order
    /// exactly. Media inputs remain individual so the result order and the failing input's
    /// original index are always explicit.
    private func embedBatch(
        _ inputs: [Input],
        role: OmniSmallRole,
        dimensions targetDimensions: Dimensions,
        space targetSpace: String
    ) async throws -> [[Float]] {
        try Task.checkCancellation()
        guard !inputs.isEmpty else { return [] }

        // Phase 1 — prepare. Files are checked; texts are tokenized and limit-checked as a whole.
        var slots = [PreparedInput?](repeating: nil, count: inputs.count)
        var textSlots = [Int]()
        var texts = [String]()
        for (index, input) in inputs.enumerated() {
            try Task.checkCancellation()
            do {
                try validate(input)
                switch input {
                case let .text(text):
                    textSlots.append(index)
                    texts.append(text)
                case let .image(url):
                    slots[index] = .image(url)
                case let .imageData(data):
                    slots[index] = .imageData(data)
                case let .audio(url):
                    slots[index] = .audio(url)
                case let .audioData(data):
                    slots[index] = .audioData(data)
                case let .video(url):
                    slots[index] = .video(url)
                case let .videoData(data):
                    slots[index] = .videoData(data)
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as OmniSmallError {
                throw OmniSmallError.invalidBatchInput(index: index, reason: error.description)
            } catch let error as OmniSmallBackendError {
                switch error {
                case let .item(_, reason), let .invalidInput(reason):
                    throw OmniSmallError.invalidBatchInput(index: index, reason: reason)
                case let .failure(reason):
                    throw OmniSmallError.inferenceFailed(reason)
                }
            } catch {
                throw OmniSmallError.invalidBatchInput(
                    index: index, reason: String(describing: error))
            }
        }

        let rows: [ValidatedText]
        do {
            rows = try await backend.prepareTexts(texts, role: role)
            try Task.checkCancellation()
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OmniSmallBackendError {
            switch error {
            case let .item(localIndex, reason):
                // Backend errors index the text slice; translate back to the original position.
                let global = localIndex < textSlots.count ? textSlots[localIndex] : localIndex
                throw OmniSmallError.invalidBatchInput(index: global, reason: reason)
            case let .invalidInput(reason):
                throw OmniSmallError.invalidBatchInput(
                    index: textSlots.first ?? 0, reason: reason)
            case let .failure(reason):
                throw OmniSmallError.inferenceFailed(reason)
            }
        } catch {
            throw OmniSmallError.inferenceFailed(String(describing: error))
        }
        guard rows.count == texts.count else {
            throw OmniSmallError.inferenceFailed(
                "text backend prepared \(rows.count) rows for \(texts.count) inputs")
        }
        for (offset, row) in rows.enumerated() {
            slots[textSlots[offset]] = .text(row)
        }

        var prepared = [PreparedInput]()
        prepared.reserveCapacity(inputs.count)
        for slot in slots {
            guard let value = slot else {
                throw OmniSmallError.inferenceFailed(
                    "batch preparation did not cover every input")
            }
            prepared.append(value)
        }

        // Phase 2 — execute. Only validated rows reach the backend here.
        var output = [[Float]]()
        output.reserveCapacity(prepared.count)
        var index = 0
        while index < prepared.count {
            try Task.checkCancellation()
            switch prepared[index] {
            case .text:
                let start = index
                var chunk = [ValidatedText]()
                chunk.reserveCapacity(min(64, prepared.count - start))
                while index < prepared.count, chunk.count < 64,
                      case let .text(next) = prepared[index] {
                    chunk.append(next)
                    index += 1
                }
                let encoded: [[Float]]
                do {
                    encoded = try await backend.embedTexts(
                        chunk, dimensions: targetDimensions)
                    try Task.checkCancellation()
                } catch is CancellationError {
                    throw CancellationError()
                } catch let error as OmniSmallBackendError {
                    switch error {
                    case let .item(localIndex, reason):
                        throw OmniSmallError.invalidBatchInput(
                            index: start + localIndex, reason: reason)
                    case let .invalidInput(reason):
                        throw OmniSmallError.invalidBatchInput(index: start, reason: reason)
                    case let .failure(reason):
                        throw OmniSmallError.inferenceFailed(
                            "text chunk starting at \(start): \(reason)")
                    }
                } catch {
                    throw OmniSmallError.inferenceFailed(
                        "text chunk starting at \(start): \(String(describing: error))")
                }
                guard encoded.count == chunk.count else {
                    throw OmniSmallError.inferenceFailed(
                        "text backend returned \(encoded.count) results for \(chunk.count) inputs")
                }
                for (offset, row) in encoded.enumerated() {
                    do {
                        output.append(try OmniSmallVectorValidator.validated(
                            row,
                            dimensions: targetDimensions,
                            space: targetSpace,
                            artifactFingerprint: artifactFingerprint))
                    } catch let error as OmniSmallError {
                        throw OmniSmallError.inferenceFailed(
                            "text output at index \(start + offset): \(error.description)")
                    } catch {
                        throw OmniSmallError.inferenceFailed(
                            "text output at index \(start + offset): \(String(describing: error))")
                    }
                }
            case let .image(url):
                output.append(try await embedMediaValidated(
                    .image(url), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            case let .imageData(data):
                output.append(try await embedMediaValidated(
                    .imageData(data), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            case let .audio(url):
                output.append(try await embedMediaValidated(
                    .audio(url), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            case let .audioData(data):
                output.append(try await embedMediaValidated(
                    .audioData(data), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            case let .video(url):
                output.append(try await embedMediaValidated(
                    .video(url), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            case let .videoData(data):
                output.append(try await embedMediaValidated(
                    .videoData(data), index: index, role: role,
                    dimensions: targetDimensions, space: targetSpace))
                index += 1
            }
        }

        guard output.count == inputs.count else {
            throw OmniSmallError.inferenceFailed(
                "backend returned \(output.count) results for \(inputs.count) inputs")
        }
        try Task.checkCancellation()
        return output
    }

    /// Media encode for one prepared input, with indexed error mapping and output validation.
    private func embedMediaValidated(
        _ input: OmniSmall.Input,
        index: Int,
        role: OmniSmallRole,
        dimensions targetDimensions: Dimensions,
        space targetSpace: String
    ) async throws -> [Float] {
        do {
            let row = try await backend.embedMedia(
                input, role: role, dimensions: targetDimensions)
            try Task.checkCancellation()
            return try OmniSmallVectorValidator.validated(
                row,
                dimensions: targetDimensions,
                space: targetSpace,
                artifactFingerprint: artifactFingerprint)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OmniSmallError {
            if case let .invalidInput(reason) = error {
                throw OmniSmallError.invalidBatchInput(index: index, reason: reason)
            }
            throw OmniSmallError.inferenceFailed(
                "media output at index \(index): \(error.description)")
        } catch let error as OmniSmallBackendError {
            switch error {
            case let .item(_, reason), let .invalidInput(reason):
                throw OmniSmallError.invalidBatchInput(index: index, reason: reason)
            case let .failure(reason):
                throw OmniSmallError.inferenceFailed(reason)
            }
        } catch {
            throw OmniSmallError.inferenceFailed(
                "media output at index \(index): \(String(describing: error))")
        }
    }

    /// Execute already-conditioned token rows without a second tokenizer pass. Rows are still
    /// length-validated and every output goes through the same shape/norm/space validator as the
    /// regular facade. Chunking preserves the production 64-row execution boundary.
    private func executeConditionedTokenRows(
        _ tokenIDRows: [[Int32]],
        dimensions targetDimensions: Dimensions,
        space targetSpace: String
    ) async throws -> [[Float]] {
        try Task.checkCancellation()
        guard !tokenIDRows.isEmpty else { return [] }

        var prepared = [ValidatedText]()
        prepared.reserveCapacity(tokenIDRows.count)
        for (index, ids) in tokenIDRows.enumerated() {
            do {
                prepared.append(try ValidatedText(tokenIDs: ids))
            } catch let error as OmniSmallBackendError {
                throw OmniSmallError.invalidBatchInput(index: index, reason: error.description)
            } catch {
                throw OmniSmallError.invalidBatchInput(
                    index: index, reason: String(describing: error))
            }
        }

        var output = [[Float]]()
        output.reserveCapacity(prepared.count)
        var start = 0
        while start < prepared.count {
            try Task.checkCancellation()
            let end = min(start + 64, prepared.count)
            let chunk = Array(prepared[start..<end])
            let encoded: [[Float]]
            do {
                encoded = try await backend.embedTexts(
                    chunk, dimensions: targetDimensions)
                try Task.checkCancellation()
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as OmniSmallBackendError {
                switch error {
                case let .item(localIndex, reason):
                    throw OmniSmallError.invalidBatchInput(
                        index: start + localIndex, reason: reason)
                case let .invalidInput(reason):
                    throw OmniSmallError.invalidBatchInput(index: start, reason: reason)
                case let .failure(reason):
                    throw OmniSmallError.inferenceFailed(
                        "prepared text chunk starting at \(start): \(reason)")
                }
            } catch {
                throw OmniSmallError.inferenceFailed(
                    "prepared text chunk starting at \(start): \(String(describing: error))")
            }
            guard encoded.count == chunk.count else {
                throw OmniSmallError.inferenceFailed(
                    "text backend returned \(encoded.count) results for \(chunk.count) prepared inputs")
            }
            for (offset, row) in encoded.enumerated() {
                do {
                    output.append(try OmniSmallVectorValidator.validated(
                        row, dimensions: targetDimensions, space: targetSpace,
                        artifactFingerprint: artifactFingerprint))
                } catch let error as OmniSmallError {
                    throw OmniSmallError.inferenceFailed(
                        "text output at index \(start + offset): \(error.description)")
                } catch {
                    throw OmniSmallError.inferenceFailed(
                        "text output at index \(start + offset): \(String(describing: error))")
                }
            }
            start = end
        }
        return output
    }

    private func validate(_ input: Input) throws {
        switch input {
        case let .text(text):
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw OmniSmallError.invalidInput("text must not be empty")
            }
        case let .image(url):
            try validateFile(url, kind: "image")
        case let .imageData(data):
            guard !data.isEmpty else {
                throw OmniSmallError.invalidInput("image data must not be empty")
            }
            guard data.count <= OmniSmallInputLimits.maximumImageBytes else {
                throw OmniSmallError.invalidInput(
                    "image data exceeds \(OmniSmallInputLimits.maximumImageBytes / 1_048_576) MiB")
            }
        case let .audio(url):
            try validateFile(url, kind: "audio")
        case let .audioData(data):
            try validateMediaData(data, kind: "audio", maximumBytes: 20 * 1_048_576)
        case let .video(url):
            try validateFile(url, kind: "video")
        case let .videoData(data):
            try validateMediaData(data, kind: "video", maximumBytes: 32 * 1_048_576)
        }
    }

    private func validateMediaData(_ data: Data, kind: String, maximumBytes: Int) throws {
        guard !data.isEmpty, data.count <= maximumBytes else {
            throw OmniSmallError.invalidInput(
                "\(kind) data must be nonempty and no larger than \(maximumBytes / 1_048_576) MiB")
        }
    }

    private func validateFile(_ url: URL, kind: String) throws {
        guard url.isFileURL else {
            throw OmniSmallError.invalidInput("\(kind) URL must be a local file URL")
        }
        guard let values = try? url.resourceValues(
            forKeys: [.isRegularFileKey, .isReadableKey]),
              values.isRegularFile == true,
              values.isReadable == true else {
            throw OmniSmallError.invalidInput(
                "\(kind) file is missing, unreadable, or not a regular file: \(url.path)")
        }
    }
}

/// A retrieval query vector. It can score only a ``DocumentEmbedding``.
public struct QueryEmbedding: Sendable {
    public let values: [Float]
    public let dimensions: OmniSmall.Dimensions
    public let space: String
    private let artifactFingerprint: String

    init(
        values: [Float],
        dimensions: OmniSmall.Dimensions,
        space: String,
        artifactFingerprint: String
    ) throws {
        self.values = try OmniSmallVectorValidator.validated(
            values,
            dimensions: dimensions,
            space: space,
            artifactFingerprint: artifactFingerprint)
        self.dimensions = dimensions
        self.space = space
        self.artifactFingerprint = artifactFingerprint
    }

    public func similarity(to document: DocumentEmbedding) throws -> Double {
        guard dimensions == document.dimensions, space == document.space else {
            throw OmniSmallError.incompatibleSpace(
                query: space,
                document: document.space)
        }
        var dot = 0.0
        var queryNorm = 0.0
        var documentNorm = 0.0
        for index in values.indices {
            let queryValue = Double(values[index])
            let documentValue = Double(document.values[index])
            dot += queryValue * documentValue
            queryNorm += queryValue * queryValue
            documentNorm += documentValue * documentValue
        }
        let denominator = queryNorm.squareRoot() * documentNorm.squareRoot()
        guard denominator.isFinite, denominator > 0 else {
            throw OmniSmallError.invalidEmbedding("similarity received a zero or non-finite vector")
        }
        let result = dot / denominator
        guard result.isFinite else {
            throw OmniSmallError.invalidEmbedding("similarity produced a non-finite result")
        }
        return result
    }
}

/// A retrieval document vector. Codable storage includes semantic space identity and exact artifact
/// provenance; compatibility uses semantic space and dimension, not compiler-specific bytes.
public struct DocumentEmbedding: Codable, Sendable {
    public let values: [Float]
    public let dimensions: OmniSmall.Dimensions
    public let space: String
    private let artifactFingerprint: String

    enum CodingKeys: String, CodingKey {
        case values
        case dimensions
        case space
        case artifactFingerprint
    }

    init(
        values: [Float],
        dimensions: OmniSmall.Dimensions,
        space: String,
        artifactFingerprint: String
    ) throws {
        self.values = try OmniSmallVectorValidator.validated(
            values,
            dimensions: dimensions,
            space: space,
            artifactFingerprint: artifactFingerprint)
        self.dimensions = dimensions
        self.space = space
        self.artifactFingerprint = artifactFingerprint
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let values = try container.decode([Float].self, forKey: .values)
        let dimensions = try container.decode(OmniSmall.Dimensions.self, forKey: .dimensions)
        let space = try container.decode(String.self, forKey: .space)
        let artifactFingerprint = try container.decode(
            String.self, forKey: .artifactFingerprint)
        do {
            try self.init(
                values: values,
                dimensions: dimensions,
                space: space,
                artifactFingerprint: artifactFingerprint)
        } catch {
            throw DecodingError.dataCorruptedError(
                forKey: .values,
                in: container,
                debugDescription: String(describing: error))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(values, forKey: .values)
        try container.encode(dimensions, forKey: .dimensions)
        try container.encode(space, forKey: .space)
        try container.encode(artifactFingerprint, forKey: .artifactFingerprint)
    }
}

public enum OmniSmallError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidBundle(String)
    case unsupportedCapability(String)
    case artifactIntegrity(String)
    case invalidInput(String)
    case invalidBatchInput(index: Int, reason: String)
    case invalidEmbedding(String)
    case incompatibleSpace(query: String, document: String)
    case inferenceFailed(String)

    public var description: String {
        switch self {
        case let .invalidBundle(reason):
            return "OmniSmall bundle is invalid: \(reason)"
        case let .unsupportedCapability(reason):
            return "OmniSmall bundle lacks a required native capability: \(reason)"
        case let .artifactIntegrity(reason):
            return "OmniSmall artifact integrity check failed: \(reason)"
        case let .invalidInput(reason):
            return "OmniSmall input is invalid: \(reason)"
        case let .invalidBatchInput(index, reason):
            return "OmniSmall batch input \(index) is invalid: \(reason)"
        case let .invalidEmbedding(reason):
            return "OmniSmall embedding is invalid: \(reason)"
        case let .incompatibleSpace(query, document):
            return "OmniSmall embeddings are incompatible: query space \(query), document space \(document)"
        case let .inferenceFailed(reason):
            return "OmniSmall inference failed: \(reason)"
        }
    }
}

enum OmniSmallRole: Sendable {
    case query
    case document

    var prompt: GlossTextEmbedder.Prompt {
        switch self {
        case .query: .query
        case .document: .document
        }
    }
}

protocol OmniSmallBackend: Sendable {
    /// Tokenize every text with its retrieval conditioning and enforce the production token limit.
    /// Preparation is a separate phase so a batch can fail on any input before inference begins:
    /// a thrown error here means no encode call was ever made.
    func prepareTexts(
        _ texts: [String],
        role: OmniSmallRole
    ) async throws -> [ValidatedText]

    func embedTexts(
        _ rows: [ValidatedText],
        dimensions: OmniSmall.Dimensions
    ) async throws -> [[Float]]

    func embedMedia(
        _ input: OmniSmall.Input,
        role: OmniSmallRole,
        dimensions: OmniSmall.Dimensions
    ) async throws -> [Float]
}

/// A text row that passed production preparation: tokenized with its retrieval conditioning and
/// within the total-token limit. The checked constructor is the only way to make one, so batch
/// execution cannot receive a row that was never validated — over-limit input is unrepresentable
/// past the prepare phase.
struct ValidatedText: Sendable {
    let tokenIDs: [Int32]

    /// Checked construction. Throws when tokenization produced no row or the row exceeds the
    /// production token limit (conditioning included).
    init(tokenIDs: [Int32]) throws {
        guard !tokenIDs.isEmpty else {
            throw OmniSmallBackendError.invalidInput("text produced no tokens")
        }
        guard tokenIDs.count <= OmniSmallInputLimits.maximumTextTokens else {
            throw OmniSmallBackendError.invalidInput(
                "text has \(tokenIDs.count) total tokens including conditioning; "
                    + "maximum is \(OmniSmallInputLimits.maximumTextTokens)")
        }
        self.tokenIDs = tokenIDs
    }
}

/// An input that passed phase-one validation and is ready for inference. Text reached this state
    /// only through `ValidatedText`; media through the facade's input checks. Execution reads this type
/// exclusively, so a late failure can only be an inference error, never a rejected input.
enum PreparedInput: Sendable {
    case text(ValidatedText)
    case image(URL)
    case imageData(Data)
    case audio(URL)
    case audioData(Data)
    case video(URL)
    case videoData(Data)
}

enum OmniSmallBackendError: Error, Sendable, CustomStringConvertible {
    case item(index: Int, reason: String)
    case invalidInput(String)
    case failure(String)

    var description: String {
        switch self {
        case let .item(index, reason):
            "item \(index): \(reason)"
        case let .invalidInput(reason):
            reason
        case let .failure(reason):
            reason
        }
    }
}

enum OmniSmallInputLimits {
    static let maximumTextTokens = 32_768
    static let maximumImageBytes = 20 * 1_048_576
    static let maximumAudioSamples = 480_000
    /// AVAudioFile expands the whole source to Float32 before resampling. A short, highly
    /// multichannel or high-rate compressed file can otherwise allocate gigabytes first.
    static let maximumDecodedSourceAudioBytes = 128 * 1_048_576

    static func validateEstimatedAudio(
        frameCount: Int64,
        sampleRate: Double,
        channelCount: AVAudioChannelCount
    ) throws {
        guard frameCount > 0,
              UInt64(frameCount) <= UInt64(AVAudioFrameCount.max) else {
            throw OmniSmallBackendError.invalidInput(
                "audio file has an invalid or unsupported frame count")
        }
        guard sampleRate.isFinite, sampleRate > 0 else {
            throw OmniSmallBackendError.invalidInput("audio file has an invalid sample rate")
        }
        let samples = UInt64(frameCount).multipliedReportingOverflow(by: UInt64(channelCount))
        let bytes = samples.partialValue.multipliedReportingOverflow(by: UInt64(MemoryLayout<Float>.size))
        guard channelCount > 0,
              !samples.overflow, !bytes.overflow,
              bytes.partialValue <= UInt64(maximumDecodedSourceAudioBytes) else {
            throw OmniSmallBackendError.invalidInput(
                "decoded source audio would exceed \(maximumDecodedSourceAudioBytes / 1_048_576) MiB")
        }
        let estimatedSamples = (Double(frameCount) * 16_000 / sampleRate).rounded(.up)
        guard estimatedSamples.isFinite,
              estimatedSamples <= Double(maximumAudioSamples) else {
            throw OmniSmallBackendError.invalidInput(
                "audio duration exceeds 30 seconds before decoding")
        }
    }

    static func validateDecodedAudio(sampleCount: Int) throws {
        guard sampleCount >= 160 else {
            throw OmniSmallBackendError.invalidInput(
                "audio must contain at least one 10 ms mel frame")
        }
        guard sampleCount <= maximumAudioSamples else {
            throw OmniSmallBackendError.invalidInput(
                "audio has \(sampleCount) samples after 16 kHz decoding; maximum is \(maximumAudioSamples) (30 seconds)")
        }
    }
}

private actor OmniSmallProductionBackend: OmniSmallBackend {
    private let bundle: GlossModelBundle
    private var textEmbedder: GlossTextEmbedder?
    private var textLoadTask: Task<Void, any Error>?
    private var imageEmbedder: GlossImageEmbedderMasked?
    private var audioEmbedder: GlossAudioEmbedderMasked?
    private var videoEmbedder: GlossVideoEmbedderMasked?

    init(bundle: GlossModelBundle) {
        self.bundle = bundle
    }

    private func resolve(_ path: String) -> URL { bundle.resolve(path) }

    /// Image pipeline: masked ViT encoder + shared media decoder, built from the manifest sections
    /// the production validator already pinned. Cached on the actor, so a class pipeline that is
    /// not `Sendable` never crosses an isolation boundary.
    private func requireImagePipeline() throws -> GlossImageEmbedderMasked {
        if let imageEmbedder { return imageEmbedder }
        let manifest = bundle.manifest
        guard let image = manifest.image,
              let decoder = manifest.decoder,
              let tokens = manifest.tokens.image else {
            throw OmniSmallBackendError.failure("bundle manifest is missing the image pipeline")
        }
        let pipeline = try GlossImageEmbedderMasked(
            visionModelURL: resolve(image.encoder),
            embedModelURL: resolve(decoder.embed),
            decoderModelURL: resolve(decoder.model),
            resourcesDir: resolve(image.resources),
            tokens: tokens.mediaTokens,
            featureDim: manifest.embeddingDimension,
            patchBuckets: image.patchBuckets,
            padTokenID: manifest.tokens.padID,
            preprocessor: GlossImagePreprocessor(
                minPixels: image.preprocess.minPixels,
                maxPixels: image.preprocess.maxPixels),
            encoderUnits: .cpuAndGPU,
            decoderUnits: nil,
            sequenceBuckets: decoder.sequenceBuckets)
        imageEmbedder = pipeline
        return pipeline
    }

    /// Audio pipeline: runtime-masked audio encoder + shared media decoder, cached like the image
    /// pipeline.
    private func requireAudioPipeline() throws -> GlossAudioEmbedderMasked {
        if let audioEmbedder { return audioEmbedder }
        let manifest = bundle.manifest
        guard let audio = manifest.audio,
              let decoder = manifest.decoder,
              let tokens = manifest.tokens.audio else {
            throw OmniSmallBackendError.failure("bundle manifest is missing the audio pipeline")
        }
        let pipeline = try GlossAudioEmbedderMasked(
            audioModelURL: resolve(audio.encoder),
            embedModelURL: resolve(decoder.embed),
            decoderModelURL: resolve(decoder.model),
            tokens: tokens.mediaTokens,
            featureDim: manifest.embeddingDimension,
            padTokenID: manifest.tokens.padID,
            encoderUnits: .cpuAndGPU,
            decoderUnits: nil,
            sequenceBuckets: decoder.sequenceBuckets)
        audioEmbedder = pipeline
        return pipeline
    }

    private func requireVideoPipeline() throws -> GlossVideoEmbedderMasked {
        if let videoEmbedder { return videoEmbedder }
        let manifest = bundle.manifest
        guard let video = manifest.video,
              let image = manifest.image,
              let decoder = manifest.decoder,
              let tokens = manifest.tokens.video else {
            throw OmniSmallBackendError.failure("bundle manifest is missing the video pipeline")
        }
        let pipeline = try GlossVideoEmbedderMasked(
            visionModelURL: resolve(video.encoder),
            embedModelURL: resolve(decoder.embed),
            decoderModelURL: resolve(decoder.model),
            resourcesDir: resolve(image.resources),
            tokens: tokens.mediaTokens,
            featureDim: manifest.embeddingDimension,
            patchBuckets: video.patchBuckets,
            padTokenID: manifest.tokens.padID,
            encoderUnits: .cpuAndGPU,
            decoderUnits: nil,
            sequenceBuckets: decoder.sequenceBuckets)
        videoEmbedder = pipeline
        return pipeline
    }

    func prepareTexts(
        _ texts: [String],
        role: OmniSmallRole
    ) async throws -> [ValidatedText] {
        try await ensureTextLoaded()
        guard let textEmbedder else {
            throw OmniSmallBackendError.failure("text embedder did not initialize")
        }

        var prepared = [ValidatedText]()
        prepared.reserveCapacity(texts.count)
        for (index, text) in texts.enumerated() {
            try Task.checkCancellation()
            do {
                prepared.append(try ValidatedText(
                    tokenIDs: textEmbedder.tokenIDs(for: text, prompt: role.prompt)))
            } catch let error as OmniSmallBackendError {
                throw OmniSmallBackendError.item(index: index, reason: error.description)
            } catch {
                throw OmniSmallBackendError.item(index: index, reason: String(describing: error))
            }
        }
        return prepared
    }

    func embedTexts(
        _ rows: [ValidatedText],
        dimensions: OmniSmall.Dimensions
    ) async throws -> [[Float]] {
        try await ensureTextLoaded()
        guard let textEmbedder else {
            throw OmniSmallBackendError.failure("text embedder did not initialize")
        }

        do {
            return try textEmbedder.embed(
                tokenIDRows: rows.map(\.tokenIDs),
                dim: dimensions.rawValue)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw OmniSmallBackendError.failure(String(describing: error))
        }
    }

    func embedMedia(
        _ input: OmniSmall.Input,
        role: OmniSmallRole,
        dimensions: OmniSmall.Dimensions
    ) async throws -> [Float] {
        try Task.checkCancellation()
        do {
            switch input {
            case .text:
                throw OmniSmallBackendError.failure("text input reached the media backend")
            case let .image(url):
                let image = try requireImagePipeline()
                return try image.embed(
                    imageURL: url,
                    dim: dimensions.rawValue,
                    prompt: role.prompt)
            case let .imageData(data):
                let image = try requireImagePipeline()
                return try image.embed(
                    imageData: data,
                    dim: dimensions.rawValue,
                    prompt: role.prompt)
            case let .audio(url):
                return try embedAudio(url, role: role, dimensions: dimensions)
            case let .audioData(data):
                return try withTemporaryMediaFile(data, extension: "wav") {
                    try embedAudio($0, role: role, dimensions: dimensions)
                }
            case let .video(url):
                return try requireVideoPipeline().embed(
                    videoURL: url,
                    dim: dimensions.rawValue,
                    prompt: role.prompt)
            case let .videoData(data):
                return try withTemporaryMediaFile(data, extension: "mp4") {
                    try requireVideoPipeline().embed(
                        videoURL: $0,
                        dim: dimensions.rawValue,
                        prompt: role.prompt)
                }
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as GlossImagePreprocessor.ImageError {
            throw OmniSmallBackendError.invalidInput(
                "image input could not be decoded: \(error)")
        } catch let error as GlossMelFrontend.MelError {
            throw OmniSmallBackendError.invalidInput(
                "audio input is invalid: \(error)")
        } catch let error as GlossVideoFile.DecodeError {
            throw OmniSmallBackendError.invalidInput(
                "video input could not be decoded: \(error)")
        } catch let error as VideoCoreMLEncoderMasked.EncoderError {
            throw OmniSmallBackendError.invalidInput(
                "video input is invalid: \(error)")
        } catch let error as OmniSmallBackendError {
            throw error
        } catch {
            throw OmniSmallBackendError.failure(String(describing: error))
        }
    }

    private func embedAudio(
        _ url: URL,
        role: OmniSmallRole,
        dimensions: OmniSmall.Dimensions
    ) throws -> [Float] {
        let audio: [Float]
        do {
            audio = try decodeBoundedAudio(url)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OmniSmallBackendError {
            throw error
        } catch {
            throw OmniSmallBackendError.invalidInput(
                "audio file could not be decoded: \(error)")
        }
        guard !audio.isEmpty else {
            throw OmniSmallBackendError.invalidInput("audio file contains no samples")
        }
        try OmniSmallInputLimits.validateDecodedAudio(sampleCount: audio.count)
        guard audio.allSatisfy(\.isFinite) else {
            throw OmniSmallBackendError.invalidInput("audio contains non-finite samples")
        }
        try Task.checkCancellation()
        return try requireAudioPipeline().embed(
            audio,
            dim: dimensions.rawValue,
            prompt: role.prompt)
    }

    private func withTemporaryMediaFile<T>(
        _ data: Data,
        extension fileExtension: String,
        _ operation: (URL) throws -> T
    ) throws -> T {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-media-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("input.\(fileExtension)")
        try data.write(to: file, options: .atomic)
        return try operation(file)
    }

    /// Reject clearly over-limit files from container metadata before allocating a decoded buffer.
    /// The exact decoded sample count is checked again by the caller after resampling.
    private func decodeBoundedAudio(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let rate = file.processingFormat.sampleRate
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: file.length,
            sampleRate: rate,
            channelCount: file.processingFormat.channelCount)
        return try GlossAudioFile.decode16kMono(url)
    }

    private func ensureTextLoaded() async throws {
        if textEmbedder != nil { return }
        if textLoadTask == nil {
            textLoadTask = Task { try await self.loadText() }
        }
        guard let task = textLoadTask else {
            throw OmniSmallBackendError.failure("text initialization task was not created")
        }
        do {
            try await task.value
            textLoadTask = nil
            try Task.checkCancellation()
        } catch {
            textLoadTask = nil
            throw error
        }
    }

    private func loadText() async throws {
        let manifest = bundle.manifest
        guard let text = manifest.text else {
            throw OmniSmallBackendError.failure("bundle manifest is missing the text tower")
        }
        textEmbedder = try await GlossTextEmbedder(
            multiFunctionModelURL: resolve(text.model),
            tokenizerFolder: resolve(text.tokenizer),
            buckets: text.buckets,
            computeUnits: nil,
            prompts: GlossTextEmbedder.PromptStrings(
                query: manifest.prompts?.query ?? "",
                document: manifest.prompts?.document ?? ""),
            padTokenID: manifest.tokens.padID,
            batchSize: text.batch?.size,
            batchBuckets: text.batch?.buckets ?? [],
            batchSizes: text.batch?.sizes ?? [])
    }
}

enum OmniSmallVectorValidator {
    private static let normTolerance = 0.002

    @discardableResult
    static func validated(
        _ values: [Float],
        dimensions: OmniSmall.Dimensions,
        space: String,
        artifactFingerprint: String
    ) throws -> [Float] {
        guard values.count == dimensions.rawValue else {
            throw OmniSmallError.invalidEmbedding(
                "expected \(dimensions.rawValue) values, found \(values.count)")
        }
        guard values.allSatisfy(\.isFinite) else {
            throw OmniSmallError.invalidEmbedding("vector contains non-finite values")
        }
        let normSquared = values.reduce(0.0) {
            $0 + Double($1) * Double($1)
        }
        let norm = normSquared.squareRoot()
        guard norm.isFinite, norm > 0 else {
            throw OmniSmallError.invalidEmbedding("vector has zero or non-finite norm")
        }
        guard abs(norm - 1) <= normTolerance else {
            throw OmniSmallError.invalidEmbedding(
                "vector must be L2-normalized; norm is \(norm)")
        }
        guard validTaggedDigest(space, prefix: "glossematics:omni-small:sha256:") else {
            throw OmniSmallError.invalidEmbedding("space identity is malformed")
        }
        guard validTaggedDigest(artifactFingerprint, prefix: "sha256:") else {
            throw OmniSmallError.invalidEmbedding("artifact fingerprint is malformed")
        }
        return values
    }

    private static func validTaggedDigest(_ value: String, prefix: String) -> Bool {
        guard value.hasPrefix(prefix) else { return false }
        let digest = value.dropFirst(prefix.count)
        return digest.utf8.count == 64 && digest.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

private struct OmniSmallLoadedBundle {
    let bundle: GlossModelBundle
    let spaces: [OmniSmall.Dimensions: String]
    let artifactFingerprint: String
}

private enum OmniSmallBundleValidator {
    private static let modelID = "jinaai/jina-embeddings-v5-omni-small"
    private static let sourceRevision = "41a20a1e1f56dad91e3a55d52ac6dc13007d67a5"
    private static let nativeDimensions = [32, 64, 128, 256, 512, 1024]
    private static let nativeTextBuckets = [
        32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768,
    ]
    private static let nativeTextBatchPairs = [
        (size: 64, bucket: 32),
        (size: 32, bucket: 64),
        (size: 16, bucket: 128),
        (size: 8, bucket: 256),
        (size: 4, bucket: 512),
    ]
    private static let nativeImageBuckets = [1_024, 1_600, 2_304, 3_072, 4_032, 5_120]
    private static let nativeAudioBuckets = [200, 400, 800, 1_600, 3_200]
    private static let nativeDecoderBuckets = [128, 256, 512, 1_024, 2_048]
    private static let nativeVideoBuckets = [256, 512, 1_024, 2_048]
    private static let requiredAudioFrames = 3_000

    static func load(
        from url: URL,
        dimensions: OmniSmall.Dimensions
    ) async throws -> OmniSmallLoadedBundle {
        guard url.isFileURL else {
            throw OmniSmallError.invalidBundle(
                "bundle URL must be a local file URL")
        }
        let resolvedURL = url.resolvingSymlinksInPath().standardizedFileURL
        guard let values = try? resolvedURL.resourceValues(
            forKeys: [.isDirectoryKey, .isReadableKey]),
              values.isDirectory == true,
              values.isReadable == true else {
            throw OmniSmallError.invalidBundle(
                "bundle URL is missing, unreadable, or not a directory: \(resolvedURL.path)")
        }
        let manifestURL = resolvedURL.appendingPathComponent("manifest.json")
        let manifestValues = try? manifestURL.resourceValues(
            forKeys: [.isRegularFileKey, .isReadableKey, .isSymbolicLinkKey])
        guard manifestValues?.isRegularFile == true,
              manifestValues?.isReadable == true,
              manifestValues?.isSymbolicLink != true else {
            throw OmniSmallError.invalidBundle("manifest.json must be a readable regular file")
        }
        let bundle: GlossModelBundle
        do {
            bundle = try GlossModelBundle(url: resolvedURL)
        } catch {
            throw OmniSmallError.invalidBundle(String(describing: error))
        }
        let manifest = bundle.manifest
        guard manifest.modelID == modelID else {
            throw OmniSmallError.invalidBundle(
                "expected \(modelID), found \(manifest.modelID)")
        }
        guard manifest.source?.repo == modelID,
              manifest.source?.revision == sourceRevision else {
            throw OmniSmallError.invalidBundle(
                "source must be pinned to \(modelID) revision \(sourceRevision)")
        }
        guard manifest.embeddingDimension == 1024,
              manifest.matryoshkaDimensions == nativeDimensions,
              manifest.matryoshkaDimensions.contains(dimensions.rawValue) else {
            throw OmniSmallError.unsupportedCapability(
                "expected embedding dimension 1024 and Matryoshka dimensions \(nativeDimensions)")
        }
        guard manifest.minimumDeployment.macOS == "15.0",
              manifest.minimumDeployment.iOS == "18.0" else {
            throw OmniSmallError.invalidBundle(
                "minimum deployment must be macOS 15.0 and iOS 18.0")
        }
        guard manifest.prompts?.query == "Query: ",
              manifest.prompts?.document == "Document: " else {
            throw OmniSmallError.invalidBundle(
                "retrieval prompts must be the pinned Query/Document conditioning")
        }
        guard manifest.taskPrompts?.isEmpty != false else {
            throw OmniSmallError.invalidBundle(
                "Omni Small must not declare code-model task prompts")
        }
        guard let precision = manifest.precision else {
            throw OmniSmallError.invalidBundle("precision is required")
        }
        let precisionID: String
        switch (precision.weights, precision.activations) {
        case ("float16", "float16"):
            precisionID = "w16a16"
        case ("int8", "float16"):
            precisionID = "w8a16"
        default:
            throw OmniSmallError.invalidBundle(
                "unsupported precision \(precision.weights)/\(precision.activations)")
        }
        let requiredSpaceID = "\(modelID):1024:\(precisionID):native-v1"

        guard let text = manifest.text else {
            throw OmniSmallError.unsupportedCapability("text tower is missing")
        }
        guard text.maxTokens == 32_768,
              text.buckets == nativeTextBuckets,
              !text.requiresAttentionMask else {
            throw OmniSmallError.unsupportedCapability(
                "text buckets must be \(nativeTextBuckets)")
        }
        guard let batch = text.batch,
              batch.pairs.count == nativeTextBatchPairs.count,
              zip(batch.pairs, nativeTextBatchPairs).allSatisfy({ pair in
                  pair.0.size == pair.1.size && pair.0.bucket == pair.1.bucket
              }) else {
            throw OmniSmallError.unsupportedCapability(
                "text batch ladder must retain b64@32, b32@64, b16@128, b8@256, and b4@512")
        }

        guard let image = manifest.image,
              image.preprocess == .init(
                patch: 16,
                merge: 2,
                minPixels: 262_144,
                maxPixels: 1_310_720),
              image.patchBuckets == nativeImageBuckets else {
            throw OmniSmallError.unsupportedCapability(
                "image preprocessing must be 262144...1310720 pixels with buckets \(nativeImageBuckets)")
        }
        guard let audio = manifest.audio,
              audio.sampleRate == 16_000,
              audio.maxSamples == 480_000,
              audio.maxFrames == requiredAudioFrames,
              audio.frameBuckets == nativeAudioBuckets else {
            throw OmniSmallError.unsupportedCapability(
                "audio must declare 16 kHz, 480000 samples, 3000 frames, and buckets \(nativeAudioBuckets)")
        }
        guard let video = manifest.video,
              video.patchBuckets == nativeVideoBuckets else {
            throw OmniSmallError.unsupportedCapability(
                "video must declare patch buckets \(nativeVideoBuckets)")
        }
        guard let decoder = manifest.decoder,
              decoder.sequenceBuckets == nativeDecoderBuckets else {
            throw OmniSmallError.unsupportedCapability(
                "shared media decoder buckets must be \(nativeDecoderBuckets)")
        }
        guard manifest.spaceID == requiredSpaceID else {
            throw OmniSmallError.invalidBundle(
                "spaceID must be \(requiredSpaceID) for the native contract")
        }
        try validateMediaTokens(manifest.tokens)
        try validateCompiledPlatform(manifest.compiled)
        try validateRequiredFiles(
            bundle: bundle,
            directory: text.tokenizer,
            names: ["tokenizer.json", "tokenizer_config.json", "config.json"],
            label: "tokenizer")
        try validateRequiredFiles(
            bundle: bundle,
            directory: image.resources,
            names: ["meta.json", "pos_embed_table.f32", "rope_inv_freq.f32"],
            label: "vision resources")

        let textURL = try compiledURL(bundle, path: text.model, label: "text model")
        let imageURL = try compiledURL(bundle, path: image.encoder, label: "image encoder")
        let audioURL = try compiledURL(bundle, path: audio.encoder, label: "audio encoder")
        let videoURL = try compiledURL(bundle, path: video.encoder, label: "video encoder")
        let embedURL = try compiledURL(bundle, path: decoder.embed, label: "media embed model")
        let decoderURL = try compiledURL(bundle, path: decoder.model, label: "media decoder")

        let textMetadata = try metadata(in: textURL)
        for bucket in text.buckets {
            try validateFunction(
                in: textMetadata,
                model: textURL.lastPathComponent,
                name: "bucket_\(bucket)",
                inputs: [
                    .init(name: "input_ids", dataType: "Int32", shape: [1, bucket]),
                    .init(name: "position_ids", dataType: "Int32", shape: [3, 1, bucket]),
                    .init(name: "selector", dataType: "Float32", shape: [1, bucket]),
                ],
                outputs: [
                    .init(name: "embedding", dataType: "Float32", shape: [1, 1_024])
                ])
        }
        for pair in text.batch?.pairs ?? [] {
            try validateFunction(
                in: textMetadata,
                model: textURL.lastPathComponent,
                name: "bucket_\(pair.bucket)_b\(pair.size)",
                inputs: [
                    .init(name: "input_ids", dataType: "Int32", shape: [pair.size, pair.bucket]),
                    .init(name: "position_ids", dataType: "Int32", shape: [3, pair.size, pair.bucket]),
                    .init(name: "selector", dataType: "Float32", shape: [pair.size, pair.bucket]),
                ],
                outputs: [
                    .init(name: "embedding", dataType: "Float32", shape: [pair.size, 1_024])
                ])
        }

        let imageMetadata = try metadata(in: imageURL)
        for patches in image.patchBuckets {
            guard patches.isMultiple(of: 4) else {
                throw OmniSmallError.invalidBundle(
                    "image patch bucket \(patches) is not divisible by merge area 4")
            }
            try validateFunction(
                in: imageMetadata,
                model: imageURL.lastPathComponent,
                name: "f\(patches)",
                inputs: [
                    .init(name: "pixel_values", dataType: "Float32", shape: [patches, 1_536]),
                    .init(name: "pos_embeds", dataType: "Float32", shape: [patches, 1_024]),
                    .init(name: "rope_cos", dataType: "Float32", shape: [patches, 64]),
                    .init(name: "rope_sin", dataType: "Float32", shape: [patches, 64]),
                    .init(name: "attn_bias", dataType: "Float32", shape: [1, 1, 1, patches]),
                ],
                outputs: [
                    .init(name: "vision_features", dataType: "Float32", shape: [patches / 4, 1_024])
                ])
        }

        let audioMetadata = try metadata(in: audioURL)
        for frames in audio.frameBuckets {
            guard frames.isMultiple(of: 200) else {
                throw OmniSmallError.invalidBundle(
                    "audio frame bucket \(frames) is not divisible by 200")
            }
            let chunks = frames / 200
            let attentionTokens = chunks * 100
            try validateFunction(
                in: audioMetadata,
                model: audioURL.lastPathComponent,
                name: "f\(frames)",
                inputs: [
                    .init(name: "packed_mel", dataType: "Float32", shape: [128, frames]),
                    .init(name: "conv_mask", dataType: "Float32", shape: [chunks, 1, 200]),
                    .init(name: "attn_bias", dataType: "Float32", shape: [1, 1, attentionTokens, attentionTokens]),
                ],
                outputs: [
                    .init(name: "audio_features", dataType: "Float32", shape: [chunks * 50, 1_024])
                ])
        }

        let embedMetadata = try metadata(in: embedURL)
        let decoderMetadata = try metadata(in: decoderURL)
        guard decoder.sequenceBuckets == decoder.sequenceBuckets.sorted(),
              Set(decoder.sequenceBuckets).count == decoder.sequenceBuckets.count else {
            throw OmniSmallError.invalidBundle(
                "media decoder sequence buckets must be sorted and unique")
        }
        for sequence in decoder.sequenceBuckets {
            try validateFunction(
                in: embedMetadata,
                model: embedURL.lastPathComponent,
                name: "f\(sequence)",
                inputs: [
                    .init(name: "input_ids", dataType: "Int32", shape: [1, sequence])
                ],
                outputs: [
                    .init(name: "out", dataType: "Float32", shape: [1, sequence, 1_024])
                ])
            try validateFunction(
                in: decoderMetadata,
                model: decoderURL.lastPathComponent,
                name: "f\(sequence)",
                inputs: [
                    .init(name: "inputs_embeds", dataType: "Float32", shape: [1, sequence, 1_024]),
                    .init(name: "position_ids", dataType: "Int32", shape: [3, 1, sequence]),
                    .init(name: "selector", dataType: "Float32", shape: [1, sequence]),
                ],
                outputs: [
                    .init(name: "embedding", dataType: "Float32", shape: [1, 1_024])
                ])
        }

        let roots = [
            text.model,
            text.tokenizer,
            image.encoder,
            image.resources,
            audio.encoder,
            video.encoder,
            decoder.embed,
            decoder.model,
        ]
        let videoMetadata = try metadata(in: videoURL)
        for patches in video.patchBuckets {
            try validateFunction(
                in: videoMetadata,
                model: videoURL.lastPathComponent,
                name: "f\(patches)",
                inputs: [
                    .init(name: "pixel_values", dataType: "Float32", shape: [patches, 1_536]),
                    .init(name: "pos_embeds", dataType: "Float32", shape: [patches, 1_024]),
                    .init(name: "rope_cos", dataType: "Float32", shape: [patches, 64]),
                    .init(name: "rope_sin", dataType: "Float32", shape: [patches, 64]),
                    .init(name: "attn_bias", dataType: "Float32", shape: [1, 1, patches, patches]),
                ],
                outputs: [
                    .init(name: "vision_features", dataType: "Float32", shape: [patches / 4, 1_024])
                ])
        }
        let artifactFingerprint = try await verifyChecksums(
            bundle: bundle,
            roots: roots,
            declaration: manifest.artifactChecksums)
        var spaces = [OmniSmall.Dimensions: String]()
        for candidate in OmniSmall.Dimensions.allCases {
            let semantic = [
                "contract=native-v1",
                "spaceID=\(requiredSpaceID)",
                "modelID=\(manifest.modelID)",
                "revision=\(sourceRevision)",
                "dimension=\(candidate.rawValue)",
                "prompts=Query: |Document: ",
                "textMaxTokens=32768",
                "image=16,2,262144,1310720,5120",
                "audio=16000,480000,3000,3200",
                // Identical to the SDK daemon's space identity, so text, image, and audio vectors
                // indexed before stay valid. The video file recipe is reported separately
                // (`OmniSmall.videoRecipe`); bump that when decoding, sampling, or resizing changes.
                "decoder=2048",
            ].joined(separator: "\n")
            let digest = SHA256.hash(data: Data(semantic.utf8)).hex
            spaces[candidate] = "glossematics:omni-small:sha256:\(digest)"
        }
        return OmniSmallLoadedBundle(
            bundle: bundle,
            spaces: spaces,
            artifactFingerprint: artifactFingerprint)
    }

    private static func validateMediaTokens(
        _ tokens: GlossModelBundle.TokenArtifacts
    ) throws {
        guard tokens.padID == 151_643 else {
            throw OmniSmallError.invalidBundle("unexpected pad token id")
        }
        guard let image = tokens.image,
              image.prefixIDs == [151_644, 872, 198, 151_652],
              image.suffixIDs == [151_653, 151_645, 198],
              image.placeholderID == 151_655,
              image.queryPrefixIDs == [151_644, 872, 198, 2_859, 25, 220, 151_652],
              image.documentPrefixIDs == [151_644, 872, 198, 7_524, 25, 220, 151_652] else {
            throw OmniSmallError.invalidBundle(
                "image query/document conditioning tokens do not match the pinned model")
        }
        guard let audio = tokens.audio,
              audio.prefixIDs == [151_644, 872, 198, 151_670],
              audio.suffixIDs == [151_671, 151_645, 198],
              audio.placeholderID == 151_669,
              audio.queryPrefixIDs == [151_644, 872, 198, 2_859, 25, 220, 151_670],
              audio.documentPrefixIDs == [151_644, 872, 198, 7_524, 25, 220, 151_670] else {
            throw OmniSmallError.invalidBundle(
                "audio query/document conditioning tokens do not match the pinned model")
        }
        guard let video = tokens.video,
              video.prefixIDs == [151_644, 872, 198, 151_652],
              video.suffixIDs == [151_653, 151_645, 198],
              video.placeholderID == 151_656,
              video.queryPrefixIDs == [151_644, 872, 198, 2_859, 25, 220, 151_652],
              video.documentPrefixIDs == [151_644, 872, 198, 7_524, 25, 220, 151_652] else {
            throw OmniSmallError.invalidBundle(
                "video query/document conditioning tokens do not match the pinned model")
        }
    }

    private static func validateCompiledPlatform(
        _ compiled: GlossModelBundle.CompiledArtifacts?
    ) throws {
        guard let compiled,
              compiled.format == "mlmodelc",
              compiled.tool == "coremlcompiler" else {
            throw OmniSmallError.invalidBundle(
                "production loading requires precompiled mlmodelc artifacts")
        }
#if os(macOS)
        guard compiled.platform == "macOS" else {
            throw OmniSmallError.invalidBundle(
                "compiled artifacts target \(compiled.platform), expected macOS")
        }
#elseif os(iOS)
        guard compiled.platform == "iOS" else {
            throw OmniSmallError.invalidBundle(
                "compiled artifacts target \(compiled.platform), expected iOS")
        }
#endif
    }

    private static func compiledURL(
        _ bundle: GlossModelBundle,
        path: String,
        label: String
    ) throws -> URL {
        let url = try artifactURL(bundle, path: path, label: label)
        let values = try? url.resourceValues(
            forKeys: [.isDirectoryKey, .isReadableKey, .isSymbolicLinkKey])
        guard url.pathExtension == "mlmodelc",
              values?.isDirectory == true,
              values?.isReadable == true,
              values?.isSymbolicLink != true else {
            throw OmniSmallError.invalidBundle(
                "\(label) must be a readable, non-symlink .mlmodelc directory at \(path)")
        }
        return url
    }

    private static func validateRequiredFiles(
        bundle: GlossModelBundle,
        directory: String,
        names: [String],
        label: String
    ) throws {
        let root = try artifactURL(bundle, path: directory, label: label)
        let rootValues = try? root.resourceValues(
            forKeys: [.isDirectoryKey, .isReadableKey, .isSymbolicLinkKey])
        guard rootValues?.isDirectory == true,
              rootValues?.isReadable == true,
              rootValues?.isSymbolicLink != true else {
            throw OmniSmallError.invalidBundle(
                "\(label) must be a readable, non-symlink directory at \(directory)")
        }
        for name in names {
            let file = root.appendingPathComponent(name)
            let values = try? file.resourceValues(
                forKeys: [.isRegularFileKey, .isReadableKey, .isSymbolicLinkKey])
            guard values?.isRegularFile == true,
                  values?.isReadable == true,
                  values?.isSymbolicLink != true else {
                throw OmniSmallError.invalidBundle(
                    "\(label) is missing required file \(directory)/\(name)")
            }
        }
    }

    private static func validateRelativePath(_ path: String, label: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw OmniSmallError.invalidBundle(
                "\(label) path must stay inside the bundle: \(path)")
        }
    }

    private static func artifactURL(
        _ bundle: GlossModelBundle,
        path: String,
        label: String
    ) throws -> URL {
        try validateRelativePath(path, label: label)
        var url = bundle.rootDirectory
        for component in path.split(separator: "/") {
            url.appendPathComponent(String(component))
            let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard values?.isSymbolicLink == false else {
                throw OmniSmallError.invalidBundle(
                    "\(label) contains a symlink or missing component: \(path)")
            }
        }
        return url
    }

    private struct MetadataRoot: Decodable {
        let functions: [MetadataFunction]
    }

    private struct MetadataFunction: Decodable {
        let name: String
        let inputSchema: [MetadataFeature]
        let outputSchema: [MetadataFeature]
    }

    private struct MetadataFeature: Decodable {
        let name: String
        let dataType: String
        let shape: String

        var dimensions: [Int]? {
            guard shape.first == "[", shape.last == "]" else { return nil }
            let body = shape.dropFirst().dropLast()
            if body.isEmpty { return [] }
            let parsed = body.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            return parsed.count == body.split(separator: ",").count ? parsed : nil
        }
    }

    private struct MetadataExpectation {
        let name: String
        let dataType: String
        let shape: [Int]
    }

    private static func metadata(in modelURL: URL) throws -> [MetadataFunction] {
        let metadataURL = modelURL.appendingPathComponent("metadata.json")
        do {
            let roots = try JSONDecoder().decode(
                [MetadataRoot].self,
                from: Data(contentsOf: metadataURL))
            guard let root = roots.first else {
                throw OmniSmallError.invalidBundle(
                    "compiled metadata is empty at \(metadataURL.path)")
            }
            return root.functions
        } catch let error as OmniSmallError {
            throw error
        } catch {
            throw OmniSmallError.invalidBundle(
                "cannot inspect compiled metadata at \(metadataURL.path): \(error)")
        }
    }

    private static func validateFunction(
        in functions: [MetadataFunction],
        model: String,
        name: String,
        inputs: [MetadataExpectation],
        outputs: [MetadataExpectation]
    ) throws {
        guard let function = functions.first(where: { $0.name == name }) else {
            throw OmniSmallError.unsupportedCapability(
                "\(model) has no \(name) function")
        }
        guard function.inputSchema.count == inputs.count,
              function.outputSchema.count == outputs.count else {
            throw OmniSmallError.unsupportedCapability(
                "\(name) must have exactly \(inputs.count) inputs and \(outputs.count) outputs")
        }
        for expected in inputs {
            guard let actual = function.inputSchema.first(where: { $0.name == expected.name }),
                  actual.dataType == expected.dataType,
                  actual.dimensions == expected.shape else {
                throw OmniSmallError.unsupportedCapability(
                    "\(name).\(expected.name) must be \(expected.dataType) \(expected.shape)")
            }
        }
        for expected in outputs {
            guard let actual = function.outputSchema.first(where: { $0.name == expected.name }),
                  actual.dataType == expected.dataType,
                  actual.dimensions == expected.shape else {
                throw OmniSmallError.unsupportedCapability(
                    "\(name).\(expected.name) must be \(expected.dataType) \(expected.shape)")
            }
        }
    }

    private static func verifyChecksums(
        bundle: GlossModelBundle,
        roots: [String],
        declaration: GlossModelBundle.ArtifactChecksums?
    ) async throws -> String {
        guard let declaration else {
            throw OmniSmallError.artifactIntegrity(
                "manifest has no artifactChecksums declaration")
        }
        guard declaration.algorithm == "sha256" else {
            throw OmniSmallError.artifactIntegrity(
                "unsupported checksum algorithm \(declaration.algorithm)")
        }

        var actualPaths = Set<String>()
        for root in roots {
            let rootURL = try artifactURL(bundle, path: root, label: "checksum root")
            for path in try regularFiles(under: rootURL, bundleRoot: bundle.rootDirectory) {
                guard actualPaths.insert(path).inserted else {
                    throw OmniSmallError.artifactIntegrity(
                        "artifact file is covered by multiple roots: \(path)")
                }
            }
        }
        let declaredPaths = Set(declaration.files.keys)
        if let missing = actualPaths.subtracting(declaredPaths).sorted().first {
            throw OmniSmallError.artifactIntegrity(
                "missing checksum for \(missing)")
        }
        if let extra = declaredPaths.subtracting(actualPaths).sorted().first {
            throw OmniSmallError.artifactIntegrity(
                "checksum references an unused or missing file: \(extra)")
        }

        var verified = [String: String]()
        verified.reserveCapacity(actualPaths.count)
        for path in actualPaths.sorted() {
            try Task.checkCancellation()
            guard let expected = declaration.files[path],
                  isLowercaseSHA256(expected) else {
                throw OmniSmallError.artifactIntegrity(
                    "checksum for \(path) is not 64-character lowercase SHA-256")
            }
            let actual = try await hashFile(
                bundle.rootDirectory.appendingPathComponent(path))
            guard actual == expected else {
                throw OmniSmallError.artifactIntegrity(
                    "checksum mismatch for \(path)")
            }
            verified[path] = actual
        }

        var aggregate = SHA256()
        for path in verified.keys.sorted() {
            let pathData = Data(path.utf8)
            var pathLength = UInt64(pathData.count).bigEndian
            withUnsafeBytes(of: &pathLength) {
                aggregate.update(data: Data($0))
            }
            aggregate.update(data: pathData)
            guard let digest = verified[path].flatMap(Data.init(hex:)) else {
                throw OmniSmallError.artifactIntegrity(
                    "verified checksum for \(path) could not be decoded")
            }
            aggregate.update(data: digest)
        }
        return "sha256:\(aggregate.finalize().hex)"
    }

    private static func regularFiles(
        under root: URL,
        bundleRoot: URL
    ) throws -> [String] {
        let standardizedBundle = bundleRoot.standardizedFileURL
        let standardizedRoot = root.standardizedFileURL
        guard standardizedRoot.path.hasPrefix(standardizedBundle.path + "/") else {
            throw OmniSmallError.artifactIntegrity(
                "artifact path escapes the bundle: \(root.path)")
        }
        let rootValues = try standardizedRoot.resourceValues(
            forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isSymbolicLink != true else {
            throw OmniSmallError.artifactIntegrity(
                "symbolic links are not allowed in artifacts: \(root.path)")
        }
        if rootValues.isRegularFile == true {
            return [relativePath(standardizedRoot, to: standardizedBundle)]
        }
        guard rootValues.isDirectory == true else {
            throw OmniSmallError.artifactIntegrity(
                "artifact root does not exist: \(root.path)")
        }

        let keys: [URLResourceKey] = [
            .isRegularFileKey,
            .isDirectoryKey,
            .isSymbolicLinkKey,
        ]
        var enumerationFailure: String?
        guard let enumerator = FileManager.default.enumerator(
            at: standardizedRoot,
            includingPropertiesForKeys: keys,
            options: [],
            errorHandler: { url, error in
                enumerationFailure = "\(url.path): \(error)"
                return false
            }) else {
            throw OmniSmallError.artifactIntegrity(
                "cannot enumerate artifact root \(root.path)")
        }
        var files = [String]()
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: Set(keys))
            if values.isSymbolicLink == true {
                throw OmniSmallError.artifactIntegrity(
                    "symbolic links are not allowed in artifacts: \(url.path)")
            }
            if values.isRegularFile == true {
                if url.lastPathComponent != ".DS_Store" {
                    files.append(relativePath(url, to: standardizedBundle))
                }
            } else if values.isDirectory != true {
                throw OmniSmallError.artifactIntegrity(
                    "unsupported file type in artifacts: \(url.path)")
            }
        }
        if let enumerationFailure {
            throw OmniSmallError.artifactIntegrity(
                "cannot enumerate artifact file \(enumerationFailure)")
        }
        return files
    }

    private static func relativePath(_ url: URL, to root: URL) -> String {
        String(url.standardizedFileURL.path.dropFirst(root.path.count + 1))
    }

    private static func hashFile(_ url: URL) async throws -> String {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw OmniSmallError.artifactIntegrity(
                "cannot read \(url.path): \(error)")
        }
        defer { try? handle.close() }
        var hasher = SHA256()
        do {
            while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
                try Task.checkCancellation()
                hasher.update(data: data)
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw OmniSmallError.artifactIntegrity(
                "cannot hash \(url.path): \(error)")
        }
        return hasher.finalize().hex
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

private extension SHA256.Digest {
    var hex: String {
        map { String(format: "%02x", $0) }.joined()
    }
}

private extension Data {
    init?(hex: String) {
        guard hex.count.isMultiple(of: 2) else { return nil }
        var bytes = [UInt8]()
        bytes.reserveCapacity(hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
            bytes.append(byte)
            index = next
        }
        self.init(bytes)
    }
}
