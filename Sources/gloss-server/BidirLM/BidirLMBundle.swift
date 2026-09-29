import CoreML
import CryptoKit
import Foundation

/// A validated BidirLM bundle. Validation runs before any tokenizer or Core ML model is loaded:
/// pinned model identity and geometry, sealed qualification, every compiled function's I/O
/// signature, token-table size, and a SHA-256 over every file.
struct BidirLMBundle: Sendable {
    let root: URL
    let manifest: BidirLMManifest
    /// SHA-256 over the verified checksum map: identifies these exact artifact bytes.
    let artifactFingerprint: String

    enum Failure: Error, CustomStringConvertible, Equatable {
        case invalid(String)
        case integrity(String)

        var description: String {
            switch self {
            case let .invalid(reason): "BidirLM bundle is invalid: \(reason)"
            case let .integrity(reason): "BidirLM bundle integrity check failed: \(reason)"
            }
        }
    }

    static func load(from url: URL, allowFixture: Bool = false) throws -> BidirLMBundle {
        let root = url.resolvingSymlinksInPath().standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw Failure.invalid("bundle is not a directory: \(root.path)")
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        let manifest: BidirLMManifest
        do {
            manifest = try JSONDecoder().decode(BidirLMManifest.self, from: Data(contentsOf: manifestURL))
        } catch {
            throw Failure.invalid("manifest.json could not be decoded: \(error)")
        }
        try validateContract(manifest, allowFixture: allowFixture)
        let fingerprint = try verifyChecksums(root: root, manifest: manifest)
        try validateFiles(root: root, manifest: manifest)
        return BidirLMBundle(root: root, manifest: manifest, artifactFingerprint: fingerprint)
    }

    // MARK: - contract

    static func validateContract(_ m: BidirLMManifest, allowFixture: Bool) throws {
        if m.format == StreamedTextManifest.serverFormat {
            try validateStreamedContract(m)
            return
        }
        func require(_ condition: Bool, _ message: String) throws {
            if !condition { throw Failure.invalid(message) }
        }
        try require(m.format == BidirLMContract.format, "format must be \(BidirLMContract.format)")
        try require(m.streamed == nil, "legacy stack bundles cannot contain a streamed recipe")
        try require(m.modelID == BidirLMContract.modelID, "expected \(BidirLMContract.modelID), found \(m.modelID)")
        try require(m.revision == BidirLMContract.revision, "source must be pinned to revision \(BidirLMContract.revision)")
        try require(m.embeddingDimension == BidirLMContract.dimension, "embedding dimension must be 2048")
        try require(m.pooling == "masked_mean_l2", "pooling must be masked_mean_l2")
        try require(m.padTokenID == BidirLMContract.padTokenID, "unexpected pad token id")
        try require(m.precision == BidirLMContract.precision,
                    "bundle must be W8A16 (int8 weights, float16 activations), found \(m.precision.weights)/\(m.precision.activations)")
        try require(m.spaceID == BidirLMContract.spaceID, "spaceID must be \(BidirLMContract.spaceID)")
        try require(m.tokenEmbeddings.shape == [BidirLMContract.vocabulary, BidirLMContract.hidden]
                     && m.tokenEmbeddings.dtype == "float16", "token table must be 151936 x 2048 float16")
        let t = m.text
        try require(t.maxTokens == BidirLMContract.maxTokens, "text maxTokens must be 32768")
        try require(t.chunks == BidirLMContract.chunks && t.stackLayers == BidirLMContract.stackLayers,
                    "text chunks must be [64, 512] with 7-layer stacks")
        try require(t.keyBlock == BidirLMContract.keyBlock && t.longKeys == BidirLMContract.longKeys,
                    "long-context key geometry does not match the runtime")
        try require(t.poolWidth == BidirLMContract.poolWidth, "pool width must be 64")
        try require(t.layers == BidirLMContract.layers && t.groupModels.count == BidirLMContract.stacks,
                    "text tower must have 28 layers in 4 stack groups")
        try require(t.hidden == BidirLMContract.hidden && t.heads == BidirLMContract.heads
                     && t.kvHeads == BidirLMContract.kvHeads && t.headDim == BidirLMContract.headDim,
                    "text attention geometry does not match the runtime")
        try require(t.userPrefixIDs == BidirLMContract.userPrefixIDs
                     && t.userSuffixIDs == BidirLMContract.userSuffixIDs,
                    "chat template token IDs do not match the pinned tokenizer")
        try require(t.maskNegative <= -10_000 && t.maskNegative >= -65_504, "mask value must be a finite FP16 negative")
        if m.vision != nil || m.audio != nil {
            try require(m.mediaTokens == BidirLMContract.mediaTokens, "media token IDs do not match the pinned tokenizer")
            if !m.isFixture {
                try require(m.qualification.parity["media-ane"]?.pass == true,
                            "bundle has media towers but lacks passing media parity")
            }
        }
        if let v = m.vision {
            try require(v.chunk == BidirLMContract.encoderChunk && v.hidden == BidirLMContract.encoderHidden
                         && v.heads == BidirLMContract.encoderHeads && v.headDim == BidirLMContract.encoderHeadDim,
                        "vision encoder geometry does not match the runtime")
            try require(v.patchSize == 16 && v.temporalPatchSize == 2 && v.mergeSize == 2
                         && v.patchFeatures == BidirLMContract.patchFeatures, "vision patch geometry does not match the runtime")
            try require(v.minPixels == 65_536 && v.maxPixels == 1_048_576
                         && v.imageMean == [0.5, 0.5, 0.5] && v.imageStd == [0.5, 0.5, 0.5],
                        "image preprocessing does not match the pinned processor")
            try require(v.ropeTheta == 10_000 && v.positionTableShape == BidirLMContract.positionTableShape,
                        "vision position geometry does not match the runtime")
            try require(v.keys == BidirLMContract.visionKeys && Set(v.attentionModels.keys) == Set(v.keys.map(String.init))
                         && v.keyBlock == BidirLMContract.keyBlock && v.singleBlockMaxKeys == 4096,
                        "vision attention functions do not match the runtime")
            try require(v.layers == BidirLMContract.visionLayers, "vision tower must have 24 blocks")
            try require(v.deepstackBlocks == BidirLMContract.visionDeepstackBlocks
                         && v.deepstackBlocks.count == t.deepstackLayers, "vision DeepStack layout does not match the text tower")
        }
        if let a = m.audio {
            try require(a.chunk == BidirLMContract.encoderChunk && a.hidden == BidirLMContract.encoderHidden
                         && a.heads == BidirLMContract.encoderHeads && a.headDim == BidirLMContract.encoderHeadDim,
                        "audio encoder geometry does not match the runtime")
            try require(a.sampleRate == 16_000 && a.melBins == BidirLMContract.melBins && a.nFFT == 400 && a.hopLength == 160,
                        "audio features do not match the pinned Whisper feature extractor")
            try require(a.chunkFrames == BidirLMContract.audioChunkFrames && a.tokensPerChunk == BidirLMContract.audioTokensPerChunk
                         && a.frontBatch == BidirLMContract.audioFrontBatch, "audio front-end geometry does not match the runtime")
            try require(a.keys == BidirLMContract.audioKeys && Set(a.attentionModels.keys) == Set(a.keys.map(String.init))
                         && a.keyBlock == BidirLMContract.keyBlock && a.singleBlockMaxKeys == 4096,
                        "audio attention functions do not match the runtime")
            try require(a.layers == BidirLMContract.audioLayers, "audio tower must have 24 layers")
        }
        if m.isFixture {
            try require(allowFixture, "Core ML dummy fixture requires --allow-dummy")
        } else {
            try require(m.neuralCompute == "ane-strict", "bundle does not declare strict ANE placement")
            try require(m.qualification.sealed, "bundle is not sealed after parity qualification")
            try require(m.qualification.aneStrictAll, "bundle failed the build-time strict ANE audit")
            try require(m.qualification.parity["ane"]?.pass == true, "bundle lacks passing ANE parity")
        }
    }

    private static func validateStreamedContract(_ m: BidirLMManifest) throws {
        guard let s = m.streamed else { throw Failure.invalid("streamed bundle has no language contract") }
        try s.validate()
        guard m.modelID == BidirLMContract.modelID, m.revision == s.revision,
              m.embeddingDimension == 2048, m.pooling == "masked_mean_l2", m.padTokenID == s.padTokenID,
              m.precision == s.expectedPrecision,
              m.spaceID == s.expectedSpaceID,
              m.tokenEmbeddings.file == s.embeddings.file, m.tokenEmbeddings.shape == s.embeddings.shape,
              m.tokenEmbeddings.dtype == "float16" else {
            throw Failure.invalid("streamed model identity, precision, or embedding-space contract mismatch")
        }
        let t = m.text
        guard t.maxTokens == 32768, t.layers == 28, t.hidden == 2048, t.heads == 16, t.kvHeads == 8, t.headDim == 128,
              t.chunks == [512], t.stackLayers == 0, t.groupModels.isEmpty, t.attentionModels.isEmpty,
              t.keyBlock == 4096, t.longKeys == s.keyBuckets, t.poolWidth == 1,
              t.deepstackLayers == s.deepstackLayers, t.mropeSection == s.mropeSection,
              t.userPrefixIDs == BidirLMContract.userPrefixIDs, t.userSuffixIDs == BidirLMContract.userSuffixIDs else {
            throw Failure.invalid("streamed text metadata does not match its runtime")
        }
        guard m.fixture == nil else { throw Failure.invalid("streamed bundles cannot be fixtures") }
        // Media towers are served only as the staged W8A16 functions of the same release, with
        // their own compiled programs over the shared weights and the host geometry in vision/audio.
        var mediaStages = 0
        if m.vision != nil || m.audio != nil || m.streamedMedia != nil {
            guard let media = m.streamedMedia, let v = m.vision, let a = m.audio, m.mediaTokens != nil,
                  s.weightPrecision == "w8" else {
                throw Failure.invalid("streamed media requires the staged W8A16 towers, geometry, and media tokens")
            }
            try media.validate(revision: s.revision)
            guard v.layers == media.layers, a.layers == media.layers, v.chunk == media.chunk, a.chunk == media.chunk,
                  v.keys == media.vision.keys, a.keys == media.audio.keys, v.deepstackBlocks == media.vision.deepstackBlocks,
                  v.deepstackBlocks.count <= t.deepstackLayers,
                  v.hidden == media.hidden, a.hidden == media.hidden, v.heads == media.heads, v.headDim == media.headDim,
                  a.frontBatch == 20, a.chunkFrames == 200, a.tokensPerChunk == 25, a.melBins == 128,
                  v.patchFeatures == 1536, v.mergeSize == 2 else {
                throw Failure.invalid("staged media geometry does not match the host preprocessing contract")
            }
            mediaStages = media.stageCount
        }
        guard m.neuralCompute == "ane-strict", m.qualification.sealed,
              m.qualification.aneStrictAll, m.qualification.aneStrictFunctions == s.stages.count + mediaStages,
              m.qualification.parity["ane"]?.pass == true else {
            throw Failure.invalid("streamed bundle has not passed and sealed native ANE qualification")
        }
    }

    // MARK: - files

    private static func validateFiles(root: URL, manifest m: BidirLMManifest) throws {
        let table = root.appendingPathComponent(m.tokenEmbeddings.file)
        let size = (try? FileManager.default.attributesOfItem(atPath: table.path)[.size] as? Int) ?? -1
        guard size == BidirLMContract.vocabulary * BidirLMContract.hidden * 2 else {
            throw Failure.invalid("token table has \(size) bytes")
        }
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            guard FileManager.default.fileExists(atPath: root.appendingPathComponent(m.tokenizer)
                .appendingPathComponent(name).path) else {
                throw Failure.invalid("tokenizer is missing \(name)")
            }
        }
        if let streamed = m.streamed {
            try streamed.validateFiles(root: root)
            try m.streamedMedia?.validateFiles(root: root)
            if let v = m.vision {
                let size = (try? FileManager.default.attributesOfItem(
                    atPath: root.appendingPathComponent(v.positionTable).path)[.size] as? Int) ?? -1
                guard size == v.positionTableShape.reduce(1, *) * 4 else {
                    throw Failure.invalid("vision position table has \(size) bytes")
                }
            }
            return
        }
        let C = 512, H = BidirLMContract.hidden, P = BidirLMContract.poolWidth
        let q = [1, 8, 128, 2 * C], kv = [1, 8, 128, C]
        let ds = m.text.deepstackLayers
        for (group, path) in m.text.groupModels.enumerated() {
            let functions = try metadata(root.appendingPathComponent(path))
            let first = group * m.text.stackLayers, last = first + m.text.stackLayers
            for name in BidirLMContract.groupFunctions(group) {
                guard let fn = functions[name] else { throw Failure.invalid("\(path) has no \(name) function") }
                var inputs: [String: [Int]], outputs: [String: [Int]]
                if name.hasPrefix("stack_c") {
                    let W = Int(name.dropFirst("stack_c".count))!
                    inputs = ["hidden": [1, H, 1, W], "cos": [1, 1, 128, W], "sin": [1, 1, 128, W], "bias": [1, 1, W, 2 * W]]
                    for i in first..<last where i < ds { inputs["deepstack_\(i)"] = [1, H, 1, W] }
                    if last >= BidirLMContract.layers {
                        inputs["pool"] = [1, W, P]
                        inputs["carry"] = [1, H, P]
                        outputs = ["carry_out": [1, H, P], "embedding": [1, H, P]]
                    } else {
                        outputs = ["hidden_out": [1, H, 1, W]]
                    }
                } else if name == "head_c512" {
                    inputs = ["hidden": [1, H, 1, C], "cos": [1, 1, 128, C], "sin": [1, 1, 128, C]]
                    outputs = ["query": q, "key": kv, "value": kv]
                } else if name == "tail_c512" {
                    inputs = ["attention": q, "hidden": [1, H, 1, C], "pool": [1, C, P], "carry": [1, H, P]]
                    outputs = ["carry_out": [1, H, P], "embedding": [1, H, P]]
                } else {
                    let layer = Int(name.dropFirst("mid_c512_l".count))!
                    inputs = ["attention": q, "hidden": [1, H, 1, C], "cos": [1, 1, 128, C], "sin": [1, 1, 128, C]]
                    if layer < ds { inputs["deepstack_in"] = [1, H, 1, C] }
                    outputs = ["hidden_out": [1, H, 1, C], "query": q, "key": kv, "value": kv]
                }
                try fn.require(inputs: inputs, outputs: outputs, label: "\(path):\(name)")
            }
        }
        let expectedAttention = Set(BidirLMContract.attentionFunctions.map {
            BidirLMContract.attentionName(chunk: $0.chunk, keys: $0.keys, packed: $0.packed)
        })
        guard Set(m.text.attentionModels.keys) == expectedAttention else {
            throw Failure.invalid("attention models do not match the runtime's function list")
        }
        for spec in BidirLMContract.attentionFunctions {
            let name = BidirLMContract.attentionName(chunk: spec.chunk, keys: spec.keys, packed: spec.packed)
            guard let path = m.text.attentionModels[name],
                  let fn = try metadata(root.appendingPathComponent(path))["main"] else {
                throw Failure.invalid("attention model \(name) is missing")
            }
            let block = min(spec.keys, BidirLMContract.keyBlock)
            var inputs: [String: [Int]] = ["query": q]
            for b in 0..<(spec.keys / block) {
                inputs["key_\(b)"] = [1, 8, 128, block]
                inputs["value_\(b)"] = [1, 8, 128, block]
                inputs["bias_\(b)"] = [1, 1, block, 1]
            }
            try fn.require(inputs: inputs, outputs: ["attention": q], label: name)
        }
        try validateMediaFiles(root: root, manifest: m)
    }

    private static func validateMediaFiles(root: URL, manifest m: BidirLMManifest) throws {
        let C = BidirLMContract.encoderChunk, H = BidirLMContract.encoderHidden, D = BidirLMContract.hidden
        let heads = [1, BidirLMContract.encoderHeads, BidirLMContract.encoderHeadDim]
        let hidden = [1, H, 1, C]
        func function(_ path: String, _ name: String = "main") throws -> FunctionSignature {
            guard let fn = try metadata(root.appendingPathComponent(path))[name] else {
                throw Failure.invalid("\(path) has no \(name) function")
            }
            return fn
        }
        func tower(_ path: String, layers: Int, rope: Bool, head: [String: [Int]], headHidden: Bool,
                   deepstackBlocks: [Int], tail: [Int]) throws {
            let ropeInputs: [String: [Int]] = rope
                ? ["cos": [1, 1, BidirLMContract.encoderHeadDim, C], "sin": [1, 1, BidirLMContract.encoderHeadDim, C]] : [:]
            let qkv = ["query": heads + [C], "key": heads + [C], "value": heads + [C]]
            try function(path, "head").require(
                inputs: head.merging(ropeInputs) { a, _ in a },
                outputs: headHidden ? qkv.merging(["hidden_out": hidden]) { a, _ in a } : qkv, label: "\(path):head")
            for i in 0..<(layers - 1) {
                var outputs = qkv
                outputs["hidden_out"] = hidden
                if deepstackBlocks.contains(i) { outputs["deepstack"] = [1, D, 1, C / 4] }
                try function(path, "mid_l\(i)").require(
                    inputs: ["attention": heads + [C], "hidden": hidden].merging(ropeInputs) { a, _ in a },
                    outputs: outputs, label: "\(path):mid_l\(i)")
            }
            try function(path, "tail").require(inputs: ["attention": heads + [C], "hidden": hidden],
                                               outputs: ["features": tail], label: "\(path):tail")
        }
        func attention(_ models: [String: String], block: (Int) -> Int) throws {
            for (keysText, path) in models {
                guard let keys = Int(keysText) else { throw Failure.invalid("bad attention bucket \(keysText)") }
                let b = block(keys)
                var inputs: [String: [Int]] = ["query": heads + [C]]
                for i in 0..<(keys / b) {
                    inputs["key_\(i)"] = heads + [b]
                    inputs["value_\(i)"] = heads + [b]
                    inputs["bias_\(i)"] = [1, 1, b, 1]
                }
                try function(path).require(inputs: inputs, outputs: ["attention": heads + [C]], label: path)
            }
        }
        if let v = m.vision {
            let table = root.appendingPathComponent(v.positionTable)
            let size = (try? FileManager.default.attributesOfItem(atPath: table.path)[.size] as? Int) ?? -1
            guard size == v.positionTableShape.reduce(1, *) * 4 else {
                throw Failure.invalid("vision position table has \(size) bytes")
            }
            try tower(v.towerModel, layers: v.layers, rope: true,
                      head: ["pixels": [1, v.patchFeatures, 1, C], "positions": hidden], headHidden: true,
                      deepstackBlocks: v.deepstackBlocks, tail: [1, D, 1, C / 4])
            try attention(v.attentionModels) { $0 <= v.singleBlockMaxKeys ? $0 : v.keyBlock }
        }
        if let a = m.audio {
            try function(a.frontModel).require(
                inputs: ["mel": [a.frontBatch, 1, a.melBins, a.chunkFrames],
                         "mask1": [a.frontBatch, 1, 1, a.chunkFrames / 2], "mask2": [a.frontBatch, 1, 1, a.chunkFrames / 4]],
                outputs: ["hidden_out": [1, H, 1, a.frontBatch * a.tokensPerChunk]], label: a.frontModel)
            try tower(a.towerModel, layers: a.layers, rope: false, head: ["hidden": hidden], headHidden: false,
                      deepstackBlocks: [], tail: [1, D, 1, C])
            try attention(a.attentionModels) { $0 <= a.singleBlockMaxKeys ? $0 : a.keyBlock }
        }
    }

    /// Every (package, function) pair the runtime loads, for placement audits.
    static func functions(_ m: BidirLMManifest) -> [(path: String, function: String)] {
        if let streamed = m.streamed {
            return streamed.stages.sorted { $0.key < $1.key }.map { ($0.value.compiled, $0.value.functionName) }
                + (m.streamedMedia?.functions ?? [])
        }
        var out = [(String, String)]()
        for (group, path) in m.text.groupModels.enumerated() {
            out += BidirLMContract.groupFunctions(group).map { (path, $0) }
        }
        out += m.text.attentionModels.sorted { $0.key < $1.key }.map { ($0.value, "main") }
        if let v = m.vision {
            out += BidirLMContract.towerFunctions(layers: v.layers).map { (v.towerModel, $0) }
            out += v.attentionModels.sorted { Int($0.key)! < Int($1.key)! }.map { ($0.value, "main") }
        }
        if let a = m.audio {
            out.append((a.frontModel, "main"))
            out += BidirLMContract.towerFunctions(layers: a.layers).map { (a.towerModel, $0) }
            out += a.attentionModels.sorted { Int($0.key)! < Int($1.key)! }.map { ($0.value, "main") }
        }
        return out
    }

    /// Program sets the runtime loads on demand (everything except the resident text stacks),
    /// with the names `ProgramResidency` uses for them.
    static func onDemandSets(_ m: BidirLMManifest) -> [(String, [ProgramResidency.Function])] {
        // All 34 streamed language stages are pinned as one set; staged media loads on demand.
        if m.streamed != nil { return m.streamedMedia?.programSets ?? [] }
        var sets = [(String, [ProgramResidency.Function])]()
        var long = [ProgramResidency.Function]()
        for (group, path) in m.text.groupModels.enumerated() {
            long += BidirLMContract.groupFunctions(group).filter { !$0.hasPrefix("stack_") }.map { .init(path: path, name: $0) }
        }
        sets.append((BidirLMTextEncoder.longSet, long))
        for spec in BidirLMContract.attentionFunctions {
            let name = BidirLMContract.attentionName(chunk: spec.chunk, keys: spec.keys, packed: spec.packed)
            if let path = m.text.attentionModels[name] { sets.append(("text.attn.\(spec.keys)", [.init(path: path, name: nil)])) }
        }
        for (family, tower, layers, attention) in [("vision", m.vision?.towerModel, m.vision?.layers, m.vision?.attentionModels),
                                                   ("audio", m.audio?.towerModel, m.audio?.layers, m.audio?.attentionModels)] {
            guard let tower, let layers, let attention else { continue }
            sets.append(("\(family).tower", BidirLMContract.towerFunctions(layers: layers).map { .init(path: tower, name: $0) }))
            for (keys, path) in attention.sorted(by: { Int($0.key)! < Int($1.key)! }) {
                sets.append(("\(family).attn.\(keys)", [.init(path: path, name: nil)]))
            }
        }
        if let a = m.audio { sets.append(("audio.front", [.init(path: a.frontModel, name: nil)])) }
        return sets
    }

    struct FunctionSignature {
        let inputs: [String: (String, [Int])]
        let outputs: [String: (String, [Int])]

        func require(inputs expectedInputs: [String: [Int]], outputs expectedOutputs: [String: [Int]],
                     label: String) throws {
            guard Set(inputs.keys) == Set(expectedInputs.keys),
                  Set(outputs.keys) == Set(expectedOutputs.keys) else {
                throw Failure.invalid("\(label) has inputs \(inputs.keys.sorted()) / outputs \(outputs.keys.sorted())")
            }
            for (name, shape) in expectedInputs where inputs[name]! != ("Float16", shape) {
                throw Failure.invalid("\(label).\(name) must be Float16 \(shape), found \(inputs[name]!)")
            }
            for (name, shape) in expectedOutputs where outputs[name]! != ("Float16", shape) {
                throw Failure.invalid("\(label).\(name) must be Float16 \(shape), found \(outputs[name]!)")
            }
        }
    }

    private struct MetadataRoot: Decodable {
        struct Feature: Decodable { let name: String; let dataType: String; let shape: String }
        struct Function: Decodable { let name: String; let inputSchema: [Feature]; let outputSchema: [Feature] }
        let functions: [Function]?
        let inputSchema: [Feature]?
        let outputSchema: [Feature]?
    }

    static func metadata(_ compiled: URL) throws -> [String: FunctionSignature] {
        guard compiled.pathExtension == "mlmodelc" else {
            throw Failure.invalid("\(compiled.lastPathComponent) must be a precompiled .mlmodelc")
        }
        let url = compiled.appendingPathComponent("metadata.json")
        guard let roots = try? JSONDecoder().decode([MetadataRoot].self, from: Data(contentsOf: url)),
              let root = roots.first else {
            throw Failure.invalid("cannot read compiled metadata for \(compiled.lastPathComponent)")
        }
        func parse(_ features: [MetadataRoot.Feature]) -> [String: (String, [Int])] {
            var out = [String: (String, [Int])]()
            for f in features {
                let dims = f.shape.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                out[f.name] = (f.dataType, dims)
            }
            return out
        }
        if let functions = root.functions {
            return Dictionary(uniqueKeysWithValues: functions.map {
                ($0.name, FunctionSignature(inputs: parse($0.inputSchema), outputs: parse($0.outputSchema)))
            })
        }
        return ["main": FunctionSignature(inputs: parse(root.inputSchema ?? []), outputs: parse(root.outputSchema ?? []))]
    }

    // MARK: - checksums

    private static func verifyChecksums(root: URL, manifest: BidirLMManifest) throws -> String {
        guard let declared = manifest.checksums else { throw Failure.integrity("manifest has no checksums") }
        guard declared.algorithm == "sha256" else { throw Failure.integrity("unsupported checksum algorithm") }
        var actual = Set<String>()
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey]) else {
            throw Failure.integrity("cannot enumerate bundle")
        }
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey])
            if values.isSymbolicLink == true { throw Failure.integrity("symbolic links are not allowed: \(url.path)") }
            guard values.isRegularFile == true else {
                if values.isDirectory == true { continue }
                throw Failure.integrity("unsupported file type: \(url.path)")
            }
            let relative = String(url.standardizedFileURL.path.dropFirst(root.path.count + 1))
            if relative == "manifest.json" || url.lastPathComponent == ".DS_Store" { continue }
            actual.insert(relative)
        }
        let expected = Set(declared.files.keys)
        if let missing = actual.subtracting(expected).sorted().first {
            throw Failure.integrity("file without checksum: \(missing)")
        }
        if let absent = expected.subtracting(actual).sorted().first {
            throw Failure.integrity("checksummed file is missing: \(absent)")
        }
        var aggregate = SHA256()
        for path in actual.sorted() {
            let digest = try hash(root.appendingPathComponent(path))
            guard digest == declared.files[path] else { throw Failure.integrity("checksum mismatch for \(path)") }
            aggregate.update(data: Data(path.utf8))
            aggregate.update(data: Data(digest.utf8))
        }
        return "sha256:" + aggregate.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 8 << 20), !data.isEmpty { hasher.update(data: data) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
