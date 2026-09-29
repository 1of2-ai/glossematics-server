import Foundation

/// Image, audio, and interleaved-message inputs.
///
/// Media support is a property of the loaded bundle: a text-only bundle advertises only `text`
/// and rejects media explicitly (HTTP 400, `unsupported_modality`) instead of silently dropping
/// or approximating it. Preparation (decode, resample, patchify, log-mel, tokenization, MRoPE
/// positions) runs on the CPU off the accelerator lane; tower and language-model work is
/// scheduled by `TextScheduler`.
struct MediaPipeline: Sendable {
    enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        case invalid(String)
        case internalError(String)

        var description: String {
            switch self {
            case let .unsupported(reason), let .invalid(reason), let .internalError(reason): reason
            }
        }

        var isClientError: Bool {
            switch self {
            case .unsupported, .invalid: true
            case .internalError: false
            }
        }

        var code: String? {
            switch self {
            case .unsupported: "unsupported_modality"
            case .invalid: "invalid_media"
            case .internalError: nil
            }
        }
    }

    /// The longest clip accepted before decoding (the 32768-token context holds ~43 minutes).
    static let maximumAudioSeconds = 2_700.0

    let backend: BidirLMBackend
    let lane: AcceleratorLane
    let metrics: ServerMetrics
    private let mel: GlossMelFrontend?

    init(backend: BidirLMBackend, lane: AcceleratorLane, metrics: ServerMetrics) throws {
        self.backend = backend
        self.lane = lane
        self.metrics = metrics
        mel = backend.hasAudio ? try GlossMelFrontend() : nil
    }

    var modalities: [String] {
        var out = ["text"]
        if backend.hasVision { out.append("image") }
        if backend.hasAudio { out.append("audio") }
        if backend.hasVision || backend.hasAudio { out.append("message") }
        return out
    }

    /// Validate and preprocess one media item into a language-model input.
    func prepare(_ item: EmbeddingInputItem, index: Int) async throws -> ScheduledInput {
        let kind: String
        let parts: [MessagePart]
        switch item {
        case let .image(data): kind = "image"; parts = [.image(data)]
        case let .audio(data): kind = "audio"; parts = [.audio(data)]
        case let .message(content): kind = "message"; parts = content
        default: throw Failure.internalError("input \(index) is not a media item")
        }
        guard modalities.contains(kind) else {
            if !backend.mediaQualified, backend.bundle.manifest.vision != nil || backend.bundle.manifest.audio != nil {
                throw Failure.unsupported(
                    "input \(index): \(kind) inputs are not served with --compute cpu (the media towers are qualified on ane and gpu)")
            }
            throw Failure.unsupported("input \(index): this bundle does not include \(kind) support")
        }
        guard !parts.isEmpty else { throw Failure.invalid("input \(index): a message needs at least one content part") }
        for part in parts {
            switch part {
            case .image where !backend.hasVision:
                throw Failure.unsupported("input \(index): this bundle does not include image support")
            case .audio where !backend.hasAudio:
                throw Failure.unsupported("input \(index): this bundle does not include audio support")
            default: break
            }
        }
        let manifest = backend.bundle.manifest
        let tokenizer = backend.tokenizer
        let mel = self.mel
        return try await Self.offload {
            var built = [MediaSequenceBuilder.Part]()
            for part in parts {
                do {
                    switch part {
                    case let .text(text):
                        built.append(.text(text))
                    case let .image(data):
                        built.append(.media(.image(try BidirLMMediaInputs.image(data, config: manifest.vision!))))
                    case let .audio(data):
                        let samples = try BidirLMMediaInputs.samples16k(data, maximumSeconds: Self.maximumAudioSeconds)
                        built.append(.media(.audio(try BidirLMMediaInputs.audio(samples, frontend: mel!))))
                    }
                } catch let error as BidirLMMediaInputs.Failure {
                    throw Failure.invalid("input \(index): \(error.description)")
                }
            }
            let pending: PendingMediaSequence
            do {
                pending = try MediaSequenceBuilder.build(built, tokenizer: tokenizer,
                                                         tokens: manifest.mediaTokens ?? BidirLMContract.mediaTokens,
                                                         kind: kind)
            } catch let error as MediaSequenceBuilder.Failure {
                throw Failure.invalid("input \(index): \(error.description)")
            }
            if pending.spans.isEmpty { return .sequence(LMSequence(ids: pending.ids)) }
            return .media(pending)
        }
    }

    /// CPU-bound preprocessing runs on a bounded background queue, not the cooperative pool.
    private static let queue: OperationQueue = {
        let q = OperationQueue()
        q.name = "gloss-media-preprocessing"
        q.maxConcurrentOperationCount = max(2, ProcessInfo.processInfo.activeProcessorCount / 2)
        q.qualityOfService = .userInitiated
        return q
    }()

    private static func offload<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.addOperation {
                do { continuation.resume(returning: try work()) } catch { continuation.resume(throwing: error) }
            }
        }
    }
}
