import Foundation

/// The model family a bundle belongs to, detected from its `manifest.json` before any family's
/// full contract validation runs. One process serves one bundle; the family selects the pipeline
/// (tokenizer, runtime, scheduler, request semantics, docs) behind the shared HTTP shell.
enum ModelFamily: String, Sendable, CaseIterable {
    /// BidirLM Omni 2.5B: `format` is `bidirlm-omni-ane-v2` (complete encoders) or
    /// `bidirlm-omni-streamed-v2` (staged ANE release). 2048 dimensions, no retrieval prompts.
    case bidirlm = "bidirlm-omni"
    /// jina-embeddings-v5-omni-small: `formatVersion` 2. Matryoshka 32-1024 dimensions,
    /// query/document retrieval prompts, text, image, audio, and video towers.
    case jinaOmniSmall = "jina-embeddings-v5-omni-small"

    struct DetectionError: Error, CustomStringConvertible {
        let description: String
    }

    static let supportedSummary =
        "BidirLM Omni (manifest format bidirlm-omni-*) and jina-embeddings-v5-omni-small (formatVersion 2)"

    static func detect(bundle url: URL) throws -> ModelFamily {
        let manifestURL = url.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw DetectionError(description: "bundle has no readable manifest.json at \(manifestURL.path)")
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DetectionError(description: "\(manifestURL.path) is not a JSON object")
        }
        return try detect(manifest: object)
    }

    static func detect(manifest object: [String: Any]) throws -> ModelFamily {
        let format = object["format"] as? String
        let modelID = object["modelID"] as? String ?? ""
        if let format, format.hasPrefix("bidirlm-omni") { return .bidirlm }
        if format == nil, object["formatVersion"] as? Int == 2 {
            if modelID == "jinaai/jina-embeddings-v5-omni-small" { return .jinaOmniSmall }
            throw DetectionError(description:
                "Jina bundle for \(modelID.isEmpty ? "an unknown model" : modelID) is not supported; this server runs jina-embeddings-v5-omni-small only")
        }
        throw DetectionError(description:
            "unsupported bundle (manifest format \(format ?? "none"), model \(modelID.isEmpty ? "unknown" : modelID)); this server serves \(supportedSummary)")
    }
}
