import Foundation

/// A different numerical and scheduling contract from the legacy W8A16 stack bundle.
/// v1's mean-scaled summaries must never be interpreted as v2 normalized contexts.
struct StreamedTextManifest: Decodable, Sendable {
    struct Geometry: Decodable, Sendable {
        let chunk: Int
        let keyBlock: Int
        let capacity: Int
        enum CodingKeys: String, CodingKey { case chunk, capacity; case keyBlock = "key_block" }
    }
    struct Embeddings: Decodable, Sendable { let file: String; let shape: [Int] }
    /// A stage is either a single-function package ("main") or a named function of the
    /// shared-weight multifunction release artifact.
    struct Stage: Decodable, Sendable {
        let compiled: String
        let function: String?
        var functionName: String { function ?? "main" }
    }
    struct Tables: Decodable, Sendable { let cos: String; let sin: String; let inverseFrequency: String }

    let format: String
    let revision: String
    let layers: Int
    let geometry: Geometry
    let keyBuckets: [Int]
    let weightPrecision: String
    let rmsFP32: Bool
    let deepstackLayers: Int
    let padTokenID: Int32
    let mropeSection: [Int]
    let embeddings: Embeddings
    let pooling: String
    let attentionSchedule: String
    let attentionSummary: String
    let nativeTables: Tables
    let stages: [String: Stage]

    enum CodingKeys: String, CodingKey {
        case format, revision, layers, geometry, embeddings, pooling, stages
        case keyBuckets = "key_buckets", weightPrecision = "weight_precision", rmsFP32 = "rms_fp32"
        case deepstackLayers = "deepstack_layers", padTokenID = "pad_token_id", mropeSection = "mrope_section"
        case attentionSchedule = "attention_schedule", attentionSummary = "attention_summary", nativeTables = "native_tables"
    }

    static let format = "bidirlm-streamed-v2"
    static let serverFormat = "bidirlm-omni-streamed-v2"
    static let spaceID = "\(BidirLMContract.modelID):\(BidirLMContract.revision):2048:fp16:ane-streamed-normalized-v2"
    /// Release recipe: int8 weights shared bit-for-bit with the complete CPU/GPU functions.
    static let spaceIDW8 = "\(BidirLMContract.modelID):\(BidirLMContract.revision):2048:w8a16:ane-streamed-v2"

    /// Each weight recipe is a separate embedding space; vectors must never be mixed.
    var expectedSpaceID: String { weightPrecision == "w8" ? Self.spaceIDW8 : Self.spaceID }
    var expectedPrecision: BidirLMManifest.Precision {
        weightPrecision == "w8" ? .init(weights: "int8", activations: "float16")
                                : .init(weights: "float16", activations: "float16")
    }

    func validate() throws {
        func require(_ value: Bool, _ reason: String) throws {
            if !value { throw BidirLMBundle.Failure.invalid(reason) }
        }
        try require(format == Self.format && revision == BidirLMContract.revision, "unsupported streamed format or source")
        try require(layers == 28 && geometry.chunk == 512 && geometry.keyBlock == 4096 && geometry.capacity == 32768,
                    "streamed runtime requires all 28 layers, C512/B4096/32768")
        try require(keyBuckets == [512, 4096, 8192, 16384, 32768], "unsupported streamed key buckets")
        try require((weightPrecision == "fp16" || weightPrecision == "w8") && !rmsFP32,
                    "streamed runtime requires the FP16 or W8A16 ANE recipe")
        try require(attentionSchedule == "summary_then_merge" && attentionSummary == "max_sum_normalized_output",
                    "streamed attention must use max/sum/normalized-output summaries")
        try require(pooling == "masked_mean_l2_float64_host", "unsupported streamed pooling")
        try require(embeddings.shape == [151936, 2048] && padTokenID == BidirLMContract.padTokenID,
                    "unexpected streamed token table")
        try require(deepstackLayers == 3 && mropeSection == [24, 20, 20],
                    "unexpected streamed MRoPE or DeepStack geometry")
        let expected = Set(["head", "summary_b512", "summary_b4096", "merge_n2", "merge_n4", "merge_n8"]
                           + (0..<28).map { "boundary_\($0)" })
        try require(Set(stages.keys) == expected, "streamed artifact has missing or unexpected stages")
        for path in stages.values.map(\.compiled) + [embeddings.file, nativeTables.cos, nativeTables.sin, nativeTables.inverseFrequency] {
            try require(!path.isEmpty && !path.hasPrefix("/") && !path.split(separator: "/").contains(".."),
                        "streamed asset path must remain inside its bundle")
        }
    }

    func validateFiles(root: URL) throws {
        for (file, size) in [(embeddings.file, 151936 * 2048 * 2),
                             (nativeTables.cos, 128 * 32768 * 2), (nativeTables.sin, 128 * 32768 * 2),
                             (nativeTables.inverseFrequency, 64 * 4)] {
            let attributes = try FileManager.default.attributesOfItem(atPath: root.appendingPathComponent(file).path)
            guard attributes[.size] as? Int == size else {
                throw BidirLMBundle.Failure.invalid("streamed asset \(file) must contain \(size) bytes")
            }
        }
        let hidden = [1, 2048, 1, 512], q = [1, 8, 128, 1024], kv = [1, 8, 128, 512], rope = [1, 1, 128, 512]
        for (name, stage) in stages {
            let functions = try BidirLMBundle.metadata(root.appendingPathComponent(stage.compiled))
            guard let fn = functions[stage.functionName] else {
                throw BidirLMBundle.Failure.invalid("\(name) has no function \(stage.functionName)")
            }
            var inputs: [String: [Int]], outputs: [String: [Int]]
            if name == "head" {
                inputs = ["hidden": hidden, "cos": rope, "sin": rope]
                outputs = ["query": q, "key": kv, "value": kv]
            } else if name.hasPrefix("boundary_"), let layer = Int(name.dropFirst("boundary_".count)) {
                inputs = ["hidden": hidden, "attention": q]
                outputs = ["hidden_out": hidden]
                if layer < 27 {
                    inputs["cos"] = rope; inputs["sin"] = rope
                    outputs.merge(["query": q, "key": kv, "value": kv]) { _, new in new }
                } else { outputs["normalized"] = hidden }
                if layer < deepstackLayers { inputs["deepstack"] = hidden }
            } else if name.hasPrefix("summary_b"), let block = Int(name.dropFirst("summary_b".count)) {
                inputs = ["query": q, "valid": [1, block, 1, 1]]
                for h in 0..<8 { inputs["key_\(h)"] = [1, block, 1, 128]; inputs["value_\(h)"] = [1, 128, 1, block] }
                outputs = ["maximum": [8, 1, 1, 1024], "mass": [8, 1, 1, 1024], "context": q]
            } else if name.hasPrefix("merge_n"), let blocks = Int(name.dropFirst("merge_n".count)) {
                inputs = [:]
                for b in 0..<blocks {
                    inputs["maximum_\(b)"] = [8, 1, 1, 1024]; inputs["mass_\(b)"] = [8, 1, 1, 1024]
                    inputs["context_\(b)"] = q
                }
                outputs = ["attention": q]
            } else { throw BidirLMBundle.Failure.invalid("unknown streamed stage \(name)") }
            try fn.require(inputs: inputs, outputs: outputs, label: name)
        }
    }
}
