import Foundation
import Glossematics
import Tokenizers

struct ServerConfig: Sendable {
    var bundleURL: URL
    var port: UInt16
    var defaultDimensions: OmniSmall.Dimensions
    var modelName: String?
    var maxBatch: Int
    var maxBodyBytes: Int
}

/// Lazily loads and caches one `OmniSmall` per requested Matryoshka dimension. The production
/// API pins a single dimension per instance, so the OpenAI `dimensions` parameter maps to
/// additional loads of the same bundle; failures are cached so a broken bundle fails fast
/// on every request instead of re-verifying gigabytes of checksums per call.
actor ModelStore {
    private let bundleURL: URL
    let defaultDimensions: OmniSmall.Dimensions
    private var loaded: [OmniSmall.Dimensions: OmniSmall] = [:]
    private var inFlight: [OmniSmall.Dimensions: Task<OmniSmall, any Error>] = [:]
    private(set) var loadFailures: [OmniSmall.Dimensions: String] = [:]

    init(bundleURL: URL, defaultDimensions: OmniSmall.Dimensions) {
        self.bundleURL = bundleURL
        self.defaultDimensions = defaultDimensions
    }

    func model(for dimensions: OmniSmall.Dimensions) async throws -> OmniSmall {
        if let model = loaded[dimensions] { return model }
        if let existing = inFlight[dimensions] { return try await existing.value }
        let task = Task { () throws -> OmniSmall in
            try await OmniSmall.load(from: bundleURL, dimensions: dimensions)
        }
        inFlight[dimensions] = task
        defer { inFlight[dimensions] = nil }
        do {
            let model = try await task.value
            loaded[dimensions] = model
            loadFailures[dimensions] = nil
            return model
        } catch {
            loadFailures[dimensions] = String(describing: error)
            throw error
        }
    }

    func status() -> (loaded: [Int], loading: [Int]) {
        (loaded.keys.map(\.rawValue).sorted(),
         Array(inFlight.keys).map(\.rawValue).sorted())
    }
}

/// Real token counts for the OpenAI `usage` field, via the bundle's own tokenizer. This counts
/// the raw input text; retrieval conditioning adds a small fixed overhead that is not billed
/// here. Media items contribute zero tokens. The tokenizer is loaded once, synchronously, at
/// daemon startup and is immutable afterwards; if it cannot be loaded, counts fall back to a
/// character heuristic.
final class TokenCounter: @unchecked Sendable {
    private let tokenizer: (any Tokenizer)?

    private final class Box: @unchecked Sendable {
        var value: (any Tokenizer)?
    }

    init(bundle: GlossModelBundle) {
        guard let text = bundle.manifest.text else {
            self.tokenizer = nil
            return
        }
        let folder = bundle.resolve(text.tokenizer)
        let box = Box()
        let loaded = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            box.value = try? await AutoTokenizer.from(modelFolder: folder)
            loaded.signal()
        }
        loaded.wait()
        self.tokenizer = box.value
    }

    func count(_ text: String) -> Int {
        guard let tokenizer else {
            // Fall back to the usual ~4 characters per token heuristic.
            return max(1, text.count / 4)
        }
        return tokenizer.encode(text: text, addSpecialTokens: true).count
    }
}

/// Route table + embeddings orchestration. Shared across connections; all mutable state
/// lives in the actor-protected stores above.
struct EmbeddingsService: Sendable {
    let config: ServerConfig
    let bundle: GlossModelBundle
    let store: ModelStore
    let counter: TokenCounter
    let startedAt: Int

    var servedModelName: String {
        config.modelName ?? bundle.manifest.modelID
    }

    private var matryoshka: Set<Int> {
        Set(bundle.capabilities.matryoshkaDimensions)
    }

    func route(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/docs"), ("GET", "/docs/"):
            return docsPage()
        case ("GET", "/"):
            return HTTPResponse(
                status: 302, contentType: "text/plain",
                extraHeaders: [("Location", "/docs")], body: Data())
        case ("GET", "/health"), ("GET", "/healthz"):
            return await health()
        case ("GET", "/v1/models"):
            return models()
        case ("GET", _) where request.path.hasPrefix("/v1/models/"):
            return model(pathComponent: percentDecode(String(request.path.dropFirst("/v1/models/".count))))
        case ("POST", "/v1/embeddings"):
            return await embeddings(request)
        case ("POST", "/v1/models"), ("POST", "/health"), ("POST", "/healthz"):
            return .error(405, APIError.invalidRequest("method not allowed; use GET"))
        case (_, "/v1/embeddings"):
            return .error(405, APIError.invalidRequest("method not allowed; use POST"))
        default:
            return .error(404, APIError.invalidRequest(
                "unknown route: \(request.method) \(request.path)",
                code: "not_found"))
        }
    }

    // MARK: - GET /docs

    private func docsPage() -> HTTPResponse {
        DocsPage.response(.init(
            baseURL: "http://127.0.0.1:\(config.port)",
            modelID: servedModelName,
            defaultDimensions: config.defaultDimensions.rawValue,
            matryoshka: bundle.capabilities.matryoshkaDimensions.sorted(),
            maxBatch: config.maxBatch,
            maxBodyMB: config.maxBodyBytes / 1_048_576,
            spaceID: bundle.manifest.spaceID))
    }

    // MARK: - GET /health

    private func health() async -> HTTPResponse {
        // Polling /health warms the default dimension; a load failure surfaces as "error".
        let model = try? await store.model(for: config.defaultDimensions)
        let status = await store.status()
        let state: String
        if model != nil {
            state = "ok"
        } else if !status.loading.isEmpty {
            state = "loading"
        } else {
            state = "error"
        }
        let payload = HealthResponse(
            status: state,
            model: servedModelName,
            default_dimensions: config.defaultDimensions.rawValue,
            loaded_dimensions: status.loaded,
            loading_dimensions: status.loading,
            space: model?.space,
            port: Int(config.port))
        return .json(state == "error" ? 503 : 200, payload)
    }

    // MARK: - GET /v1/models[/{id}]

    private func models() -> HTTPResponse {
        ModelListResponse(data: [modelObject()])
            .asResponse()
    }

    private func model(pathComponent: String) -> HTTPResponse {
        if pathComponent == servedModelName {
            return modelObject().asResponse()
        }
        return .error(404, APIError.modelNotFound(pathComponent))
    }

    private func modelObject() -> ModelObject {
        ModelObject(id: servedModelName, created: startedAt, owned_by: "glossematics")
    }

    // MARK: - POST /v1/embeddings

    private func embeddings(_ request: HTTPRequest) async -> HTTPResponse {
        // 1) JSON decode.
        let body: EmbeddingsRequestBody
        do {
            body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: request.body)
        } catch {
            return .error(400, APIError.invalidRequest(decodingMessage(error), param: "input"))
        }

        // 2) Model identity.
        if let requested = body.model {
            guard requested == servedModelName else {
                return .error(404, APIError.modelNotFound(requested))
            }
        } else {
            return .error(400, APIError.invalidRequest(
                "you must provide a model parameter", param: "model", code: nil))
        }

        // 3) Input shape.
        guard !body.items.isEmpty else {
            return .error(400, APIError.invalidRequest(
                "input must not be an empty array", param: "input"))
        }
        guard body.items.count <= config.maxBatch else {
            return .error(400, APIError.invalidRequest(
                "input exceeds the maximum of \(config.maxBatch) items per request",
                param: "input"))
        }
        if let badTask = body.invalidTask {
            return .error(400, APIError.invalidRequest(
                "unknown task/role: \"\(badTask)\"; use retrieval.query, query, "
                    + "retrieval.passage, passage, or document",
                param: "task"))
        }

        // 4) encoding_format.
        var base64 = false
        if let format = body.encodingFormat {
            switch format.lowercased() {
            case "float": base64 = false
            case "base64": base64 = true
            default:
                return .error(400, APIError.invalidRequest(
                    "encoding_format must be \"float\" or \"base64\"", param: "encoding_format"))
            }
        }

        // 5) dimensions → Matryoshka check → instance.
        let dimensions: OmniSmall.Dimensions
        if let requested = body.dimensions {
            guard let resolved = OmniSmall.Dimensions(rawValue: requested),
                  matryoshka.contains(requested) else {
                return .error(400, APIError.invalidRequest(
                    "dimensions must be one of \(matryoshka.sorted()) for this model",
                    param: "dimensions"))
            }
            dimensions = resolved
        } else {
            dimensions = config.defaultDimensions
        }

        let model: OmniSmall
        do {
            model = try await store.model(for: dimensions)
        } catch let error as OmniSmallError {
            return .error(500, APIError.apiError(error.description))
        } catch {
            return .error(500, APIError.apiError(String(describing: error)))
        }

        // 6) Role conditioning. Default is document/passage: the recipe for stored corpus
        //    vectors. Queries must ask for it explicitly via task/role.
        let inputs: [OmniSmall.Input]
        do {
            inputs = try body.items.map { item in
                switch item {
                case let .text(text):
                    guard !text.isEmpty else {
                        throw APIError.invalidRequest("input must not contain empty strings", param: "input")
                    }
                    return .text(text)
                case let .image(path):
                    return .image(try mediaURL(path, kind: "image"))
                case let .audio(path):
                    return .audio(try mediaURL(path, kind: "audio"))
                }
            }
        } catch let apiError as APIError {
            return .error(400, apiError)
        } catch {
            return .error(400, APIError.invalidRequest(String(describing: error), param: "input"))
        }

        // 7) Inference — atomic ordered batch through the production API.
        let values: [[Float]]
        do {
            if body.isQuery {
                values = try await model.embedQueries(inputs).map(\.values)
            } else {
                values = try await model.embedDocuments(inputs).map(\.values)
            }
        } catch let error as OmniSmallError {
            switch error {
            case .invalidInput, .invalidBatchInput:
                return .error(400, APIError.invalidRequest(error.description, param: "input"))
            default:
                return .error(500, APIError.apiError(error.description))
            }
        } catch is CancellationError {
            return .error(500, APIError.apiError("request was cancelled"))
        } catch {
            return .error(500, APIError.apiError(String(describing: error)))
        }

        // 8) Usage counts + payload encoding.
        var promptTokens = 0
        for item in body.items {
            if case let .text(text) = item {
                promptTokens += counter.count(text)
            }
        }

        let data: [EmbeddingObject] = values.enumerated().map { index, vector in
            EmbeddingObject(
                index: index,
                embedding: base64
                    ? .base64(Self.base64LittleEndian(vector))
                    : .floats(vector))
        }
        let response = EmbeddingsResponse(
            data: data,
            model: servedModelName,
            usage: EmbeddingUsage(prompt_tokens: promptTokens, total_tokens: promptTokens))

        let headers: [(String, String)] = [
            ("X-Glossematics-Space", model.space),
            ("X-Glossematics-Dimensions", String(dimensions.rawValue)),
        ]
        return .json(200, response, extraHeaders: headers)
    }

    private func mediaURL(_ path: String, kind: String) throws -> URL {
        guard FileManager.default.fileExists(atPath: path) else {
            throw APIError.invalidRequest("\(kind) file not found: \(path)", param: "input")
        }
        return URL(fileURLWithPath: path, isDirectory: false)
    }

    private func percentDecode(_ text: String) -> String {
        text.removingPercentEncoding ?? text
    }

    private func decodingMessage(_ error: any Error) -> String {
        if let decoding = error as? DecodingError {
            switch decoding {
            case let .valueNotFound(_, context):
                return context.debugDescription
            case let .dataCorrupted(context):
                return context.debugDescription
            case let .keyNotFound(key, _):
                return "missing required parameter: '\(key.stringValue)'"
            case let .typeMismatch(_, context):
                return context.debugDescription
            @unknown default:
                return "malformed JSON body"
            }
        }
        return "malformed JSON body: \(String(describing: error))"
    }
}

extension ModelObject {
    func asResponse() -> HTTPResponse {
        .json(200, self)
    }
}

extension ModelListResponse {
    func asResponse() -> HTTPResponse {
        .json(200, self)
    }
}

extension EmbeddingsService {
    /// Float32 array → little-endian bytes → base64 (OpenAI base64 encoding_format contract).
    static func base64LittleEndian(_ values: [Float]) -> String {
        let values = values
        guard !values.isEmpty else { return "" }
        return values.withUnsafeBytes { raw in
            Data([UInt8](raw)).base64EncodedString()
        }
    }
}
