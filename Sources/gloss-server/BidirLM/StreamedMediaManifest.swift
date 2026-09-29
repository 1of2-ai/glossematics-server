import Foundation

/// Staged W8A16 vision and audio towers for the Neural Engine (`streamed_media.json`, embedded in
/// the bundle manifest as `streamedMedia`). Mirrors `bidirlm_coreml/streamed_media.py`: per tower a
/// head, 23 layer boundaries and a tail (vision: three separate mergers; audio: a convolutional
/// front end), plus the language encoder's normalized key-block summaries and merges at 16 x 64.
struct StreamedMediaManifest: Decodable, Sendable {
    struct Stage: Decodable, Sendable {
        let compiled: String
        let function: String?
        var functionName: String { function ?? "main" }
    }

    struct Tower: Decodable, Sendable {
        let keys: [Int]
        let stages: [String: Stage]
        let deepstackBlocks: [Int]?

        enum CodingKeys: String, CodingKey {
            case keys, stages
            case deepstackBlocks = "deepstack_blocks"
        }
    }

    let format: String
    let revision: String
    let weightPrecision: String
    let weightTensors: String
    let chunk: Int
    let keyBlock: Int
    let layers: Int
    let hidden: Int
    let heads: Int
    let headDim: Int
    let outDim: Int
    let attentionSummary: String
    let vision: Tower
    let audio: Tower

    enum CodingKeys: String, CodingKey {
        case format, revision, chunk, layers, hidden, heads, vision, audio
        case weightPrecision = "weight_precision", weightTensors = "weight_tensors", keyBlock = "key_block"
        case headDim = "head_dim", outDim = "out_dim", attentionSummary = "attention_summary"
    }

    static let format = "bidirlm-streamed-media-v1"
    static let visionKeys = [512, 1024, 2048, 4096]
    static let audioKeys = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    static func block(keys: Int) -> Int { min(keys, 4096) }

    /// Attention stages for one key bucket: one summary per 4096-key block, then a merge.
    static func attentionStages(keys: Int) -> [String] {
        let block = block(keys: keys)
        return ["summary_b\(block)"] + (keys > block ? ["merge_n\(keys / block)"] : [])
    }

    static func towerStages(_ family: String, layers: Int = 24) -> [String] {
        var names = family == "audio" ? ["front"] : []
        names += ["head"] + (0..<(layers - 1)).map { "boundary_\($0)" } + ["tail"]
        if family == "vision" { names += ["merger_deepstack_0", "merger_deepstack_1", "merger"] }
        return names
    }

    static func expectedStages(_ family: String, keys: [Int]) -> Set<String> {
        Set(towerStages(family) + keys.flatMap(attentionStages))
    }

    var stageCount: Int { vision.stages.count + audio.stages.count }

    func tower(_ family: String) -> Tower { family == "vision" ? vision : audio }

    func validate(revision expected: String) throws {
        func require(_ value: Bool, _ reason: String) throws {
            if !value { throw BidirLMBundle.Failure.invalid(reason) }
        }
        try require(format == Self.format && revision == expected, "unsupported staged media format or source")
        try require(weightPrecision == "w8" && weightTensors == "complete_release_shared",
                    "staged media must share the release's W8 tensors")
        try require(chunk == 512 && keyBlock == 4096 && layers == 24 && hidden == 1024 && heads == 16
                     && headDim == 64 && outDim == 2048, "unexpected staged media geometry")
        try require(attentionSummary == "max_sum_normalized_output", "media attention must use normalized summaries")
        try require(vision.keys == Self.visionKeys && audio.keys == Self.audioKeys, "unsupported media key buckets")
        try require(vision.deepstackBlocks == [8, 16], "unexpected vision DeepStack blocks")
        for family in ["vision", "audio"] {
            let t = tower(family)
            try require(Set(t.stages.keys) == Self.expectedStages(family, keys: t.keys),
                        "\(family) staged tower has missing or unexpected stages")
            for stage in t.stages.values {
                try require(!stage.compiled.isEmpty && !stage.compiled.hasPrefix("/")
                            && !stage.compiled.split(separator: "/").contains(".."),
                            "staged media path must remain inside its bundle")
            }
        }
    }

    /// Every stage's compiled signature: names, FP16, exact shapes.
    func validateFiles(root: URL) throws {
        let C = chunk, H = hidden, D = outDim
        let hiddenShape = [1, H, 1, C], q = [1, heads, headDim, C], rope = [1, 1, headDim, C]
        for family in ["vision", "audio"] {
            for (name, stage) in tower(family).stages {
                let functions = try BidirLMBundle.metadata(root.appendingPathComponent(stage.compiled))
                guard let fn = functions[stage.functionName] else {
                    throw BidirLMBundle.Failure.invalid("\(family)_\(name) has no function \(stage.functionName)")
                }
                var inputs: [String: [Int]], outputs: [String: [Int]]
                if name == "front" {
                    inputs = ["mel": [20, 1, 128, 200], "mask1": [20, 1, 1, 100], "mask2": [20, 1, 1, 50]]
                    outputs = ["hidden_out": [1, H, 1, 500]]
                } else if name == "head" {
                    inputs = family == "vision"
                        ? ["pixels": [1, 1536, 1, C], "positions": hiddenShape, "cos": rope, "sin": rope]
                        : ["hidden": hiddenShape]
                    outputs = ["query": q, "key": q, "value": q]
                    if family == "vision" { outputs["hidden_out"] = hiddenShape }
                } else if name.hasPrefix("boundary_") {
                    inputs = ["attention": q, "hidden": hiddenShape]
                    if family == "vision" { inputs["cos"] = rope; inputs["sin"] = rope }
                    outputs = ["hidden_out": hiddenShape, "query": q, "key": q, "value": q]
                } else if name == "tail" {
                    inputs = ["attention": q, "hidden": hiddenShape]
                    outputs = family == "vision" ? ["hidden_out": hiddenShape] : ["features": [1, D, 1, C]]
                } else if name.hasPrefix("merger") {
                    inputs = ["hidden": hiddenShape]
                    outputs = ["features": [1, D, 1, C / 4]]
                } else if name.hasPrefix("summary_b"), let block = Int(name.dropFirst("summary_b".count)) {
                    inputs = ["query": q, "valid": [1, block, 1, 1]]
                    for h in 0..<heads { inputs["key_\(h)"] = [1, block, 1, headDim]; inputs["value_\(h)"] = [1, headDim, 1, block] }
                    outputs = ["maximum": [heads, 1, 1, C], "mass": [heads, 1, 1, C], "context": q]
                } else if name.hasPrefix("merge_n"), let blocks = Int(name.dropFirst("merge_n".count)) {
                    inputs = [:]
                    for b in 0..<blocks {
                        inputs["maximum_\(b)"] = [heads, 1, 1, C]; inputs["mass_\(b)"] = [heads, 1, 1, C]
                        inputs["context_\(b)"] = q
                    }
                    outputs = ["attention": q]
                } else {
                    throw BidirLMBundle.Failure.invalid("unknown staged media stage \(name)")
                }
                try fn.require(inputs: inputs, outputs: outputs, label: "\(family)_\(name)")
            }
        }
    }

    /// (compiled path, function) for every stage, for placement audits.
    var functions: [(path: String, function: String)] {
        ["vision", "audio"].flatMap { family in
            tower(family).stages.sorted { $0.key < $1.key }.map { ($0.value.compiled, $0.value.functionName) }
        }
    }

    func programs(_ family: String, _ names: [String]) -> [ProgramResidency.Function] {
        names.map { .init(path: tower(family).stages[$0]!.compiled, name: tower(family).stages[$0]!.functionName) }
    }

    /// Residency sets: each tower's learned stages, the audio front end, and one set per key bucket.
    var programSets: [(String, [ProgramResidency.Function])] {
        var sets = [(String, [ProgramResidency.Function])]()
        for family in ["vision", "audio"] {
            let learned = Self.towerStages(family).filter { $0 != "front" }
            if family == "audio" { sets.append(("audio.front", programs(family, ["front"]))) }
            sets.append(("\(family).tower", programs(family, learned)))
            for keys in tower(family).keys {
                sets.append(("\(family).attn.\(keys)", programs(family, Self.attentionStages(keys: keys))))
            }
        }
        return sets
    }
}
