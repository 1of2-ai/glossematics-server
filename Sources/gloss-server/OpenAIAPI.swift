import Foundation

struct APIError: Encodable, Sendable, Error, CustomStringConvertible {
    var description: String { error.message }
    struct Payload: Encodable, Sendable {
        var message: String
        var type: String
        var param: String?
        var code: String?
    }
    var error: Payload

    static func invalidRequest(_ message: String, param: String? = nil, code: String? = nil) -> APIError {
        .init(error: .init(message: message, type: "invalid_request_error", param: param, code: code))
    }
    static func modelNotFound(_ model: String) -> APIError {
        .init(error: .init(message: "The model '\(model)' does not exist.", type: "invalid_request_error", param: "model", code: "model_not_found"))
    }
    static func apiError(_ message: String) -> APIError {
        .init(error: .init(message: message, type: "api_error", param: nil, code: nil))
    }
    static func serviceUnavailable(_ message: String) -> APIError {
        .init(error: .init(message: message, type: "service_unavailable", param: nil, code: nil))
    }
}

enum EmbeddingInputItem: Sendable {
    case text(String)
    case tokenIDs([Int32])
    /// Explicit data-URL extensions; OpenAI's embeddings endpoint does not define media inputs.
    case image(Data)
    case audio(Data)
    /// Accepted by the decoder only so the service can reject it with a clear error.
    case video(Data)
    /// One user turn interleaving text, images, and audio; produces a single embedding.
    case message([MessagePart])
}

enum MessagePart: Sendable {
    case text(String)
    case image(Data)
    case audio(Data)
}

private struct DynamicKey: CodingKey, Hashable {
    let stringValue: String
    init?(stringValue: String) { self.stringValue = stringValue }
    let intValue: Int? = nil
    init?(intValue: Int) { return nil }
}

struct EmbeddingsRequestBody: Decodable, Sendable {
    var items: [EmbeddingInputItem]
    var model: String?
    var dimensions: Int?
    var encodingFormat: String?
    var user: String?

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: DynamicKey.self)
        let allowed = Set(["input", "model", "dimensions", "encoding_format", "user"])
        for key in c.allKeys where !allowed.contains(key.stringValue) {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: c,
                debugDescription: "Unrecognized request argument supplied: \(key.stringValue)")
        }

        let inputKey = DynamicKey(stringValue: "input")!
        guard c.contains(inputKey) else {
            throw DecodingError.keyNotFound(
                inputKey,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "missing required parameter: 'input'"))
        }
        model = try c.decodeIfPresent(String.self, forKey: DynamicKey(stringValue: "model")!)
        dimensions = try c.decodeIfPresent(Int.self, forKey: DynamicKey(stringValue: "dimensions")!)
        encodingFormat = try c.decodeIfPresent(
            String.self,
            forKey: DynamicKey(stringValue: "encoding_format")!)
        user = try c.decodeIfPresent(String.self, forKey: DynamicKey(stringValue: "user")!)
        items = try c.decode(InputValue.self, forKey: inputKey).items
    }
}

private struct InputValue: Decodable {
    var items: [EmbeddingInputItem]

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let text = try? c.decode(String.self) { items = [.text(text)]; return }
        if let ids = try? c.decode([Int].self) {
            items = [.tokenIDs(try Self.tokenIDs(ids, decoder: decoder))]
            return
        }
        if let rows = try? c.decode([[Int]].self) {
            items = try rows.map { .tokenIDs(try Self.tokenIDs($0, decoder: decoder)) }
            return
        }
        if let single = try? c.decode(InputItemValue.self) { items = [single.item]; return }
        if let values = try? c.decode([InputItemValue].self) {
            items = values.map(\.item)
            return
        }
        throw DecodingError.typeMismatch(
            EmbeddingInputItem.self,
            .init(
                codingPath: decoder.codingPath,
                debugDescription: "input must be a string, an array of strings, token IDs, or an input content object"))
    }

    private static func tokenIDs(
        _ values: [Int],
        decoder: any Decoder
    ) throws -> [Int32] {
        try values.map {
            guard let value = Int32(exactly: $0), value >= 0 else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "token IDs must be non-negative 32-bit integers"))
            }
            return value
        }
    }
}

private struct InputItemValue: Decodable {
    var item: EmbeddingInputItem

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let text = try? c.decode(String.self) { item = .text(text); return }

        let object = try decoder.container(keyedBy: DynamicKey.self)
        let typeKey = DynamicKey(stringValue: "type")!
        guard object.contains(typeKey) else {
            throw DecodingError.keyNotFound(
                typeKey,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "input content objects require a type"))
        }

        let type = try object.decode(String.self, forKey: typeKey)
        switch type {
        case "input_text":
            try Self.requireOnly(["type", "text"], in: object)
            let textKey = DynamicKey(stringValue: "text")!
            guard let text = try object.decodeIfPresent(String.self, forKey: textKey) else {
                throw DecodingError.keyNotFound(
                    textKey,
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "input_text requires text"))
            }
            item = .text(text)
        case "input_image":
            try Self.requireOnly(["type", "image_url"], in: object)
            let imageURLKey = DynamicKey(stringValue: "image_url")!
            guard let imageURL = try object.decodeIfPresent(String.self, forKey: imageURLKey) else {
                throw DecodingError.keyNotFound(
                    imageURLKey,
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "input_image requires image_url"))
            }
            item = .image(try MediaDataURL.decode(
                imageURL,
                decoder: decoder,
                field: "input_image.image_url",
                mediaTypes: ["image/jpeg", "image/png", "image/webp"],
                maximumBytes: 20 * 1_048_576))
        case "input_audio":
            try Self.requireOnly(["type", "audio_url"], in: object)
            let audioURLKey = DynamicKey(stringValue: "audio_url")!
            guard let audioURL = try object.decodeIfPresent(String.self, forKey: audioURLKey) else {
                throw DecodingError.keyNotFound(
                    audioURLKey,
                    .init(codingPath: decoder.codingPath, debugDescription: "input_audio requires audio_url"))
            }
            item = .audio(try MediaDataURL.decode(
                audioURL,
                decoder: decoder,
                field: "input_audio.audio_url",
                mediaTypes: ["audio/wav", "audio/x-wav", "audio/wave"],
                maximumBytes: 20 * 1_048_576))
        case "input_video":
            try Self.requireOnly(["type", "video_url"], in: object)
            let videoURLKey = DynamicKey(stringValue: "video_url")!
            guard let videoURL = try object.decodeIfPresent(String.self, forKey: videoURLKey) else {
                throw DecodingError.keyNotFound(
                    videoURLKey,
                    .init(codingPath: decoder.codingPath, debugDescription: "input_video requires video_url"))
            }
            item = .video(try MediaDataURL.decode(
                videoURL,
                decoder: decoder,
                field: "input_video.video_url",
                mediaTypes: ["video/mp4"],
                maximumBytes: 32 * 1_048_576))
        case "message":
            try Self.requireOnly(["type", "role", "content"], in: object)
            let roleKey = DynamicKey(stringValue: "role")!
            if let role = try object.decodeIfPresent(String.self, forKey: roleKey), role != "user" {
                throw DecodingError.dataCorruptedError(
                    forKey: roleKey, in: object, debugDescription: "message role must be \"user\"")
            }
            let contentKey = DynamicKey(stringValue: "content")!
            let parts = try object.decode([InputItemValue].self, forKey: contentKey)
            guard !parts.isEmpty else {
                throw DecodingError.dataCorruptedError(
                    forKey: contentKey, in: object, debugDescription: "message content must not be empty")
            }
            item = .message(try parts.map { part in
                switch part.item {
                case let .text(text): return .text(text)
                case let .image(data): return .image(data)
                case let .audio(data): return .audio(data)
                default:
                    throw DecodingError.dataCorruptedError(
                        forKey: contentKey, in: object,
                        debugDescription: "message content supports input_text, input_image, and input_audio")
                }
            })
        default:
            throw DecodingError.dataCorruptedError(
                forKey: typeKey,
                in: object,
                debugDescription: "unsupported input content type: \(type)")
        }
    }

    private static func requireOnly(
        _ allowed: Set<String>,
        in container: KeyedDecodingContainer<DynamicKey>
    ) throws {
        for key in container.allKeys where !allowed.contains(key.stringValue) {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "Unrecognized input content argument supplied: \(key.stringValue)")
        }
    }
}

private enum MediaDataURL {
    static func decode(
        _ raw: String,
        decoder: any Decoder,
        field: String,
        mediaTypes: Set<String>,
        maximumBytes: Int
    ) throws -> Data {
        guard raw.hasPrefix("data:"),
              let comma = raw.firstIndex(of: ",") else {
            throw invalid(decoder, "\(field) must be a base64 data URL")
        }

        let metadata = String(raw[raw.index(raw.startIndex, offsetBy: 5)..<comma]).lowercased()
        let parts = metadata.split(separator: ";", omittingEmptySubsequences: false)
        guard parts.count == 2,
              mediaTypes.contains(String(parts[0])),
              parts[1] == "base64" else {
            throw invalid(
                decoder,
                "\(field) must use a supported media type followed by ;base64")
        }

        let payload = String(raw[raw.index(after: comma)...])
        let maximumBase64Characters = ((maximumBytes + 2) / 3) * 4
        guard !payload.isEmpty,
              payload.count <= maximumBase64Characters,
              payload.rangeOfCharacter(from: .whitespacesAndNewlines) == nil,
              let data = Data(base64Encoded: payload),
              data.count <= maximumBytes else {
            throw invalid(
                decoder,
                "\(field) contains invalid base64 or exceeds \(maximumBytes / 1_048_576) MiB")
        }
        return data
    }

    private static func invalid(_ decoder: any Decoder, _ message: String) -> DecodingError {
        .dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: message))
    }
}

struct EmbeddingUsage: Encodable, Sendable { var prompt_tokens: Int; var total_tokens: Int }
enum EmbeddingPayload: Encodable, Sendable {
    case floats([Float]); case base64(String)
    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self { case let .floats(v): try c.encode(v); case let .base64(s): try c.encode(s) }
    }
}
struct EmbeddingObject: Encodable, Sendable { var object = "embedding"; var index: Int; var embedding: EmbeddingPayload }
struct EmbeddingsResponse: Encodable, Sendable { var object = "list"; var data: [EmbeddingObject]; var model: String; var usage: EmbeddingUsage }
struct ModelObject: Encodable, Sendable { var id: String; var object = "model"; var created: Int; var owned_by: String }
struct ModelListResponse: Encodable, Sendable { var object = "list"; var data: [ModelObject] }

struct LiveResponse: Encodable, Sendable { var status: String; var version: String; var model: String; var port: Int }
struct HealthResponse: Encodable, Sendable {
    var status: String
    var ready: Bool
    var version: String
    var model: String
    var dimensions: Int
    var space: String
    var compute: String
    var core_ml_compute_units: String
    var placement_verified: Bool
    var placement_strict_ane: Bool?
    var placement_cost_share: [String: Double]?
    var load_seconds: Double?
    var max_tokens: Int
    var modalities: [String]
    var port: Int
    var uptime_seconds: Int
    var ready_at: Int?
    var queue_requests: Int
    var queue_items: Int
    var scheduled_requests: Int
    var pending_short_rows: Int
    var pending_long_documents: Int
    var active_long_tokens: Int?
    var active_long_progress: Double?
    var active_media_kind: String?
    var active_media_progress: Double?
    var programs_resident: Int
    var program_budget: Int
    var program_sets_resident: [String]
    var program_set_loads: Int
    var program_set_evictions: Int
    var keep_warm_seconds: Int
    var last_keep_warm_at: Int?
    var last_keep_warm_error: String?
    var keep_warm_failures: Int
    var fixture: Bool
    var error: String?
}
