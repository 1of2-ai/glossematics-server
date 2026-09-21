import Foundation

// OpenAI-compatible wire schema for POST /v1/embeddings, plus the error envelope shared by
// every route. Unknown request fields are ignored for forward compatibility, matching the
// OpenAI clients' behavior.

struct APIError: Encodable, Sendable, Error {
    struct Payload: Encodable, Sendable {
        var message: String
        var type: String
        var param: String?
        var code: String?
    }

    var error: Payload

    static func invalidRequest(
        _ message: String, param: String? = nil, code: String? = nil
    ) -> APIError {
        APIError(error: Payload(message: message, type: "invalid_request_error", param: param, code: code))
    }

    static func modelNotFound(_ model: String) -> APIError {
        APIError(error: Payload(
            message: "The model '\(model)' does not exist.",
            type: "invalid_request_error", param: "model", code: "model_not_found"))
    }

    static func apiError(_ message: String) -> APIError {
        APIError(error: Payload(message: message, type: "api_error", param: nil, code: nil))
    }

    static func serviceUnavailable(_ message: String) -> APIError {
        APIError(error: Payload(message: message, type: "service_unavailable", param: nil, code: nil))
    }
}

/// One element of the `input` field: plain text (the OpenAI contract), or a local extension
/// object pointing at an on-disk image/audio file for this bundle's media towers.
enum EmbeddingInputItem: Sendable {
    case text(String)
    case image(path: String)
    case audio(path: String)
}

private struct InputItemObject: Decodable {
    var text: String?
    var image_path: String?
    var audio_path: String?
    var path: String?
    var kind: String?
}

struct EmbeddingsRequestBody: Decodable, Sendable {
    var items: [EmbeddingInputItem]
    var model: String?
    var dimensions: Int?
    var encodingFormat: String?
    /// Local retrieval-role extension: "retrieval.query" / "retrieval.passage" (jina-compatible),
    /// with "query" / "document" accepted as aliases. Defaults to passage (document) semantics,
    /// the recipe for indexed corpora.
    var task: String?
    var role: String?
    var user: String?

    enum Keys: String, CodingKey {
        case input, model, dimensions, user
        case encodingFormat = "encoding_format"
        case task, role
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: Keys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        dimensions = try container.decodeIfPresent(Int.self, forKey: .dimensions)
        encodingFormat = try container.decodeIfPresent(String.self, forKey: .encodingFormat)
        task = try container.decodeIfPresent(String.self, forKey: .task)
        role = try container.decodeIfPresent(String.self, forKey: .role)
        user = try container.decodeIfPresent(String.self, forKey: .user)

        guard container.contains(.input) else {
            throw DecodingError.valueNotFound(
                InputValue.self,
                .init(codingPath: decoder.codingPath + [Keys.input],
                      debugDescription: "missing required parameter: 'input'"))
        }
        let value = try container.decode(InputValue.self, forKey: .input)
        items = value.items
    }

    var isQuery: Bool {
        for candidate in [task, role].compactMap({ $0 }) {
            switch candidate.lowercased() {
            case "retrieval.query", "query":
                return true
            case "retrieval.passage", "passage", "document":
                return false
            default:
                continue
            }
        }
        return false
    }

    /// An explicitly provided task/role that matches no known value is a client error;
    /// absent values default to document/passage semantics.
    var invalidTask: String? {
        for candidate in [task, role].compactMap({ $0 }) {
            switch candidate.lowercased() {
            case "retrieval.query", "query", "retrieval.passage", "passage", "document":
                continue
            default:
                return candidate
            }
        }
        return nil
    }
}

private struct InputValue: Decodable {
    var items: [EmbeddingInputItem]

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self.items = [.text(text)]
            return
        }
        self.items = try container.decode([InputItemValue].self).map(\.item)
    }
}

private struct InputItemValue: Decodable {
    var item: EmbeddingInputItem

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self.item = .text(text)
            return
        }
        let object = try container.decode(InputItemObject.self)
        if let text = object.text {
            self.item = .text(text)
            return
        }
        if let imagePath = object.image_path {
            self.item = .image(path: imagePath)
            return
        }
        if let audioPath = object.audio_path {
            self.item = .audio(path: audioPath)
            return
        }
        if let path = object.path {
            switch object.kind?.lowercased() {
            case "image":
                self.item = .image(path: path)
                return
            case "audio":
                self.item = .audio(path: path)
                return
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "path objects require kind: \"image\" or \"audio\"")
            }
        }
        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "input objects must set text, image_path, audio_path, or path+kind")
    }
}

// MARK: - Responses

struct EmbeddingUsage: Encodable, Sendable {
    var prompt_tokens: Int
    var total_tokens: Int
}

enum EmbeddingPayload: Encodable, Sendable {
    case floats([Float])
    case base64(String)

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .floats(values):
            try container.encode(values)
        case let .base64(text):
            try container.encode(text)
        }
    }
}

struct EmbeddingObject: Encodable, Sendable {
    var object = "embedding"
    var index: Int
    var embedding: EmbeddingPayload
}

struct EmbeddingsResponse: Encodable, Sendable {
    var object = "list"
    var data: [EmbeddingObject]
    var model: String
    var usage: EmbeddingUsage
}

struct ModelObject: Encodable, Sendable {
    var id: String
    var object = "model"
    var created: Int
    var owned_by: String
}

struct ModelListResponse: Encodable, Sendable {
    var object = "list"
    var data: [ModelObject]
}

struct HealthResponse: Encodable, Sendable {
    var status: String
    var model: String
    var default_dimensions: Int
    var loaded_dimensions: [Int]
    var loading_dimensions: [Int]
    var space: String?
    var port: Int
}
