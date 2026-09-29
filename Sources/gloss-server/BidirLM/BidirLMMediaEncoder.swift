import CoreML
import Foundation

/// Chunked vision and audio towers (mirrors `bidirlm_coreml/ane_media_runtime.py`).
///
/// Both towers are pre-norm transformer encoders over 512-token chunks. Each tower package
/// holds `head` (layer 0 projections; for vision also the patch embedding), one `mid_l{i}` per
/// layer boundary (layer i's output projection and MLP, then layer i+1's projections; the vision
/// DeepStack mergers after blocks 8 and 16 are extra `deepstack` outputs), and `tail` (the last
/// layer plus the patch merger or audio output projection). Attention over every key of the item
/// runs in a weightless bucket function between them. Tower programs are loaded on demand
/// through `ProgramResidency`, and work is exposed as resumable runs of one chunk operation per
/// step so the scheduler can interleave a long clip with short text requests.
final class BidirLMMediaEncoder: @unchecked Sendable {
    enum Failure: Error, CustomStringConvertible {
        case unavailable(String)
        case invalidInput(String)
        case invalidOutput(String)

        var description: String {
            switch self {
            case let .unavailable(reason), let .invalidInput(reason), let .invalidOutput(reason): reason
            }
        }
    }

    static let chunk = BidirLMContract.encoderChunk
    static let hidden = BidirLMContract.encoderHidden
    static let heads = BidirLMContract.encoderHeads
    static let headDim = BidirLMContract.encoderHeadDim
    static let outDim = BidirLMContract.hidden

    /// One tower's programs for one item: head, mids, tail, and the item's attention bucket.
    struct Programs {
        let head: MLModel
        let mids: [MLModel]
        let tail: MLModel
        let attention: MLModel
        let keys: Int
        let block: Int
    }

    let residency: ProgramResidency
    /// Staged W8A16 towers of a streamed release; nil for the legacy chunked programs.
    let staged: StreamedMediaManifest?
    let vision: BidirLMManifest.Vision?
    let audio: BidirLMManifest.Audio?
    /// Learned vision position table (2304 x 1024, row-major), interpolated on the host.
    let positionTable: [Float]

    init(residency: ProgramResidency, manifest: BidirLMManifest) throws {
        self.residency = residency
        staged = manifest.streamed != nil ? manifest.streamedMedia : nil
        vision = manifest.vision
        audio = manifest.audio
        if let v = manifest.vision {
            let data = try Data(contentsOf: residency.root.appendingPathComponent(v.positionTable))
            positionTable = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            guard positionTable.count == v.positionTableShape.reduce(1, *) else {
                throw Failure.unavailable("vision position table has \(positionTable.count) values")
            }
        } else {
            positionTable = []
        }
    }

    /// Acquire a tower's programs and the attention bucket covering `tokens` keys.
    fileprivate func programs(family: String, tower path: String, layers: Int, attention: [String: String],
                              keys available: [Int], keyBlock: Int, singleBlockMaxKeys: Int,
                              tokens: Int) throws -> Programs {
        guard let keys = available.sorted().first(where: { $0 >= tokens }),
              let attentionPath = attention[String(keys)] else {
            throw Failure.invalidInput("\(tokens) \(family) tokens exceed the tower's \(available.max() ?? 0)-key attention")
        }
        let functions = BidirLMContract.towerFunctions(layers: layers).map { ProgramResidency.Function(path: path, name: $0) }
        let towerSet = "\(family).tower", attentionSet = "\(family).attn.\(keys)"
        let tower = try residency.acquire(towerSet, functions, inUse: [attentionSet])
        let bucket = try residency.acquire(attentionSet, [.init(path: attentionPath, name: nil)], inUse: [towerSet])
        let ordered = functions.map { tower[$0]! }
        return Programs(head: ordered.first!, mids: Array(ordered.dropFirst().dropLast()), tail: ordered.last!,
                        attention: bucket.values.first!, keys: keys,
                        block: keys <= singleBlockMaxKeys ? keys : keyBlock)
    }

    /// Acquire a staged tower (learned stages) and the attention set for the item's key bucket.
    fileprivate func stagedPrograms(family: String, tokens: Int) throws -> StagedEncoderCore.Programs {
        guard let media = staged else { throw Failure.unavailable("this bundle has no staged \(family) tower") }
        let tower = media.tower(family)
        guard let keys = tower.keys.sorted().first(where: { $0 >= tokens }) else {
            throw Failure.invalidInput("\(tokens) \(family) tokens exceed the tower's \(tower.keys.max() ?? 0)-key attention")
        }
        let learned = StreamedMediaManifest.towerStages(family).filter { $0 != "front" }
        let attention = StreamedMediaManifest.attentionStages(keys: keys)
        let towerSet = "\(family).tower", attentionSet = "\(family).attn.\(keys)"
        let towerFunctions = media.programs(family, learned), attentionFunctions = media.programs(family, attention)
        let loadedTower = try residency.acquire(towerSet, towerFunctions, inUse: [attentionSet])
        let loadedAttention = try residency.acquire(attentionSet, attentionFunctions, inUse: [towerSet])
        func model(_ stage: String) -> MLModel {
            if let i = learned.firstIndex(of: stage) { return loadedTower[towerFunctions[i]]! }
            return loadedAttention[attentionFunctions[attention.firstIndex(of: stage)!]]!
        }
        let block = StreamedMediaManifest.block(keys: keys)
        var deepstack = [Int: MLModel]()
        for (k, layer) in (tower.deepstackBlocks ?? []).enumerated() { deepstack[layer] = model("merger_deepstack_\(k)") }
        return .init(head: model("head"), boundaries: (0..<(media.layers - 1)).map { model("boundary_\($0)") },
                     tail: model("tail"), summary: model("summary_b\(block)"),
                     merge: keys > block ? model("merge_n\(keys / block)") : nil,
                     deepstackMergers: deepstack, merger: family == "vision" ? model("merger") : nil,
                     keys: keys, block: block)
    }

    /// Features for one media item: rows replacing its placeholder tokens, plus DeepStack rows
    /// (images only), all row-major FP16 with 2048 columns.
    struct Features: Sendable {
        let rows: [Float16]
        let deepstack: [[Float16]]
        var count: Int { rows.count / BidirLMMediaEncoder.outDim }
    }

    /// A resumable tower execution.
    protocol Run: AnyObject, Sendable {
        /// Advance by one step; true once `features` is available.
        func step() throws -> Bool
        var features: Features? { get }
        var totalSteps: Int { get }
        var completedSteps: Int { get }
    }

    func startImage(_ image: PreparedImage) throws -> any Run {
        guard let vision else { throw Failure.unavailable("this bundle has no vision tower") }
        return try VisionRun(encoder: self, config: vision, image: image)
    }

    func startAudio(_ clip: PreparedAudio) throws -> any Run {
        guard let audio else { throw Failure.unavailable("this bundle has no audio tower") }
        return try AudioRun(encoder: self, config: audio, clip: clip)
    }

    /// Run to completion (tests and warm-up).
    func features(of run: any Run) throws -> Features {
        while try !run.step() { try Task.checkCancellation() }
        return run.features!
    }

    // MARK: - shared encoder core

    /// Layer-by-layer execution over hidden chunks. The first N steps run `head` per chunk; then
    /// a step is one chunk's attention for layer i followed by that chunk's `mid_l{i}` (or
    /// `tail`). Keys and values alternate between two buffers so layer i+1's projections never
    /// overwrite keys that layer i's remaining chunks still attend to.
    final class EncoderCore: TowerCore {
        let programs: Programs
        let chunks: Int
        let tokens: Int
        private let layers: Int
        private var heads: [[String: MLMultiArray]]
        private var hidden: [MLMultiArray?]
        private let rope: [(MLMultiArray, MLMultiArray)]?
        private let keys: [[MLMultiArray]]
        private let values: [[MLMultiArray]]
        private let biases: [MLMultiArray]
        private var queries: [MLMultiArray?]
        private var headed = 0
        private(set) var layer = 0
        private var chunk = 0
        /// Tail outputs per chunk once finished.
        private(set) var outputs: [MLMultiArray?]
        /// DeepStack merger outputs per layer (after that layer), per chunk.
        private(set) var deepstack: [Int: [MLMultiArray]] = [:]

        var steps: Int { chunks * (1 + layers) }
        var finished: Bool { layer == layers }

        init(programs: Programs, layers: Int, heads: [[String: MLMultiArray]], tokens: Int,
             rope: [(MLMultiArray, MLMultiArray)]?) throws {
            self.programs = programs
            self.layers = layers
            self.heads = heads
            self.tokens = tokens
            self.rope = rope
            chunks = heads.count
            hidden = Array(repeating: nil, count: chunks)
            queries = Array(repeating: nil, count: chunks)
            outputs = Array(repeating: nil, count: chunks)
            let width = programs.block
            let shape = [1, BidirLMMediaEncoder.heads, BidirLMMediaEncoder.headDim, width]
            keys = try (0..<2).map { _ in try (0..<(programs.keys / width)).map { _ in try HalfArrays.zeros(shape) } }
            values = try (0..<2).map { _ in try (0..<(programs.keys / width)).map { _ in try HalfArrays.zeros(shape) } }
            let negative = Float16(BidirLMContract.maskNegative)
            biases = try (0..<(programs.keys / width)).map { b in
                let array = try HalfArrays.zeros([1, 1, width, 1])
                let ptr = HalfArrays.pointer(array)
                for i in 0..<width where b * width + i >= tokens { ptr[i] = negative }
                return array
            }
        }

        private func store(_ out: MLFeatureProvider, buffer: Int, chunk c: Int) throws {
            let C = BidirLMMediaEncoder.chunk, block = programs.block
            let shape = [1, BidirLMMediaEncoder.heads, BidirLMMediaEncoder.headDim, C]
            queries[c] = try HalfArrays.output(out, "query", shape)
            let start = c * C
            HalfArrays.copyColumns(from: try HalfArrays.output(out, "key", shape),
                                   into: keys[buffer][start / block], columnOffset: start % block)
            HalfArrays.copyColumns(from: try HalfArrays.output(out, "value", shape),
                                   into: values[buffer][start / block], columnOffset: start % block)
        }

        /// One chunk operation.
        func step() throws {
            precondition(!finished)
            let C = BidirLMMediaEncoder.chunk, H = BidirLMMediaEncoder.hidden
            if headed < chunks {
                let c = headed
                let out = try BidirLMMediaEncoder.predict(programs.head, heads[c])
                hidden[c] = out.featureValue(for: "hidden_out")?.multiArrayValue ?? heads[c]["hidden"]
                if hidden[c] == nil { throw Failure.invalidOutput("tower head produced no hidden state") }
                try store(out, buffer: 0, chunk: c)
                heads[c] = [:]
                headed += 1
                return
            }
            let current = layer % 2, next = (layer + 1) % 2
            var feeds: [String: MLMultiArray] = ["query": queries[chunk]!]
            for b in 0..<keys[current].count {
                feeds["key_\(b)"] = keys[current][b]
                feeds["value_\(b)"] = values[current][b]
                feeds["bias_\(b)"] = biases[b]
            }
            let att = try HalfArrays.output(try BidirLMMediaEncoder.predict(programs.attention, feeds), "attention",
                                            [1, BidirLMMediaEncoder.heads, BidirLMMediaEncoder.headDim, C])
            if layer == layers - 1 {
                let out = try BidirLMMediaEncoder.predict(programs.tail, ["attention": att, "hidden": hidden[chunk]!])
                guard let features = out.featureValue(for: "features")?.multiArrayValue else {
                    throw Failure.invalidOutput("tower tail produced no features")
                }
                outputs[chunk] = features
                hidden[chunk] = nil
            } else {
                var midFeeds: [String: MLMultiArray] = ["attention": att, "hidden": hidden[chunk]!]
                if let rope { midFeeds["cos"] = rope[chunk].0; midFeeds["sin"] = rope[chunk].1 }
                let out = try BidirLMMediaEncoder.predict(programs.mids[layer], midFeeds)
                hidden[chunk] = try HalfArrays.output(out, "hidden_out", [1, H, 1, C])
                try store(out, buffer: next, chunk: chunk)
                if let ds = out.featureValue(for: "deepstack")?.multiArrayValue {
                    deepstack[layer, default: []].append(ds)
                }
            }
            chunk += 1
            if chunk == chunks { chunk = 0; layer += 1 }
        }
    }

    /// Staged towers (`streamed_media`): the same chunk schedule as `EncoderCore`, but keys and
    /// values live in per-head contraction layouts (`StreamedKVBank`) and attention is the language
    /// encoder's normalized key-block summaries, merged across blocks. Vision mergers are separate
    /// functions: the DeepStack mergers run on block 8/16 outputs, the final merger on the tail.
    final class StagedEncoderCore: TowerCore {
        struct Programs {
            let head: MLModel
            let boundaries: [MLModel]
            let tail: MLModel
            let summary: MLModel
            let merge: MLModel?
            let deepstackMergers: [Int: MLModel]
            let merger: MLModel?
            let keys: Int
            let block: Int
        }

        let programs: Programs
        let chunks: Int
        private let layers: Int
        private var heads: [[String: MLMultiArray]]
        private var hidden: [MLMultiArray?]
        private var queries: [MLMultiArray?]
        private let rope: [(MLMultiArray, MLMultiArray)]?
        private let banks: [StreamedKVBank]
        private let keyScratch: HalfSurface
        private let valueScratch: HalfSurface
        private var headed = 0
        private(set) var layer = 0
        private var chunk = 0
        private(set) var outputs: [MLMultiArray?]
        private(set) var deepstack: [Int: [MLMultiArray]] = [:]

        var steps: Int { chunks * (1 + layers) }
        var finished: Bool { layer == layers }

        init(programs: Programs, layers: Int, heads: [[String: MLMultiArray]], tokens: Int,
             rope: [(MLMultiArray, MLMultiArray)]?) throws {
            self.programs = programs
            self.layers = layers
            self.heads = heads
            self.rope = rope
            chunks = heads.count
            guard layers == programs.boundaries.count + 1, chunks * BidirLMMediaEncoder.chunk <= programs.keys,
                  tokens > 0, tokens <= chunks * BidirLMMediaEncoder.chunk else {
                throw Failure.invalidInput("staged tower geometry does not match its inputs")
            }
            hidden = Array(repeating: nil, count: chunks)
            queries = Array(repeating: nil, count: chunks)
            outputs = Array(repeating: nil, count: chunks)
            let valid = (0..<programs.keys).map { $0 < tokens }
            banks = try (0..<2).map { _ in
                try StreamedKVBank(keys: programs.keys, block: programs.block, valid: valid,
                                   heads: BidirLMMediaEncoder.heads, dim: BidirLMMediaEncoder.headDim)
            }
            let q = [1, BidirLMMediaEncoder.heads, BidirLMMediaEncoder.headDim, BidirLMMediaEncoder.chunk]
            keyScratch = try HalfSurface(q)
            valueScratch = try HalfSurface(q)
        }

        private var queryShape: [Int] { [1, BidirLMMediaEncoder.heads, BidirLMMediaEncoder.headDim, BidirLMMediaEncoder.chunk] }

        private func store(_ out: MLFeatureProvider, bank: StreamedKVBank, chunk c: Int) throws {
            queries[c] = try HalfArrays.output(out, "query", queryShape)
            try keyScratch.copy(from: try HalfArrays.output(out, "key", queryShape))
            try valueScratch.copy(from: try HalfArrays.output(out, "value", queryShape))
            try bank.store(start: c * BidirLMMediaEncoder.chunk, key: keyScratch, value: valueScratch)
        }

        /// Every key of the item: one summary per 4096-key block, merged by mass when there are several.
        private func attention(_ bank: StreamedKVBank, query: MLMultiArray) throws -> MLMultiArray {
            let blocks = programs.keys / programs.block, H = BidirLMMediaEncoder.heads, C = BidirLMMediaEncoder.chunk
            var mergeFeeds = [String: MLMultiArray]()
            for b in 0..<blocks {
                var feeds: [String: MLMultiArray] = ["query": query, "valid": bank.masks[b].array]
                for h in 0..<H {
                    feeds["key_\(h)"] = bank.keys[b][h].array
                    feeds["value_\(h)"] = bank.values[b][h].array
                }
                let out = try BidirLMMediaEncoder.predict(programs.summary, feeds)
                if blocks == 1 { return try HalfArrays.output(out, "context", queryShape) }
                mergeFeeds["maximum_\(b)"] = try HalfArrays.output(out, "maximum", [H, 1, 1, C])
                mergeFeeds["mass_\(b)"] = try HalfArrays.output(out, "mass", [H, 1, 1, C])
                mergeFeeds["context_\(b)"] = try HalfArrays.output(out, "context", queryShape)
            }
            guard let merge = programs.merge else { throw Failure.unavailable("multi-block attention has no merge") }
            return try HalfArrays.output(try BidirLMMediaEncoder.predict(merge, mergeFeeds), "attention", queryShape)
        }

        private func features(_ model: MLModel, hidden: MLMultiArray) throws -> MLMultiArray {
            let out = try BidirLMMediaEncoder.predict(model, ["hidden": hidden])
            guard let features = out.featureValue(for: "features")?.multiArrayValue else {
                throw Failure.invalidOutput("tower merger produced no features")
            }
            return features
        }

        func step() throws {
            precondition(!finished)
            let C = BidirLMMediaEncoder.chunk, H = BidirLMMediaEncoder.hidden
            if headed < chunks {
                let c = headed
                let out = try BidirLMMediaEncoder.predict(programs.head, heads[c])
                hidden[c] = out.featureValue(for: "hidden_out")?.multiArrayValue ?? heads[c]["hidden"]
                if hidden[c] == nil { throw Failure.invalidOutput("tower head produced no hidden state") }
                try store(out, bank: banks[0], chunk: c)
                heads[c] = [:]
                headed += 1
                return
            }
            let att = try attention(banks[layer % 2], query: queries[chunk]!)
            if layer == layers - 1 {
                let out = try BidirLMMediaEncoder.predict(programs.tail, ["attention": att, "hidden": hidden[chunk]!])
                if let merger = programs.merger {
                    outputs[chunk] = try features(merger, hidden: try HalfArrays.output(out, "hidden_out", [1, H, 1, C]))
                } else {
                    guard let features = out.featureValue(for: "features")?.multiArrayValue else {
                        throw Failure.invalidOutput("tower tail produced no features")
                    }
                    outputs[chunk] = features
                }
                hidden[chunk] = nil
            } else {
                var feeds: [String: MLMultiArray] = ["attention": att, "hidden": hidden[chunk]!]
                if let rope { feeds["cos"] = rope[chunk].0; feeds["sin"] = rope[chunk].1 }
                let out = try BidirLMMediaEncoder.predict(programs.boundaries[layer], feeds)
                let state = try HalfArrays.output(out, "hidden_out", [1, H, 1, C])
                hidden[chunk] = state
                try store(out, bank: banks[(layer + 1) % 2], chunk: chunk)
                if let merger = programs.deepstackMergers[layer] {
                    deepstack[layer, default: []].append(try features(merger, hidden: state))
                }
            }
            chunk += 1
            if chunk == chunks { chunk = 0; layer += 1 }
        }
    }

    // MARK: - vision

    final class VisionRun: Run, @unchecked Sendable {
        private let core: any TowerCore
        private let patches: Int
        private let deepstackBlocks: [Int]
        private(set) var features: Features?
        private(set) var completedSteps = 0
        let totalSteps: Int

        init(encoder: BidirLMMediaEncoder, config v: BidirLMManifest.Vision, image: PreparedImage) throws {
            let n = image.gridH * image.gridW
            guard image.gridH > 0, image.gridW > 0, image.gridH % v.mergeSize == 0, image.gridW % v.mergeSize == 0,
                  image.pixels.count == n * v.patchFeatures else {
                throw Failure.invalidInput("image patches do not match the grid")
            }
            let C = BidirLMMediaEncoder.chunk, H = BidirLMMediaEncoder.hidden
            let chunks = (n + C - 1) / C
            let rope = try VisionTables.rope(gridH: image.gridH, gridW: image.gridW, merge: v.mergeSize,
                                             theta: v.ropeTheta, chunks: chunks)
            let side = Int(Double(v.positionTableShape[0]).squareRoot())
            var heads = [[String: MLMultiArray]]()
            for c in 0..<chunks {
                let pixels = try HalfArrays.zeros([1, v.patchFeatures, 1, C])
                let positions = try HalfArrays.zeros([1, H, 1, C])
                let pp = HalfArrays.pointer(pixels), qp = HalfArrays.pointer(positions)
                let count = min(C, n - c * C)
                let table = VisionTables.positions(encoder.positionTable, side: side, hidden: H, gridH: image.gridH,
                                                   gridW: image.gridW, merge: v.mergeSize, range: (c * C)..<(c * C + count))
                for t in 0..<count {
                    let src = (c * C + t) * v.patchFeatures
                    for f in 0..<v.patchFeatures { pp[f * C + t] = Float16(image.pixels[src + f]) }
                    for ch in 0..<H { qp[ch * C + t] = Float16(table[t * H + ch]) }
                }
                heads.append(["pixels": pixels, "positions": positions, "cos": rope[c].0, "sin": rope[c].1])
            }
            if encoder.staged != nil {
                core = try StagedEncoderCore(programs: encoder.stagedPrograms(family: "vision", tokens: chunks * C),
                                             layers: v.layers, heads: heads, tokens: n, rope: rope)
            } else {
                let programs = try encoder.programs(
                    family: "vision", tower: v.towerModel, layers: v.layers, attention: v.attentionModels, keys: v.keys,
                    keyBlock: v.keyBlock, singleBlockMaxKeys: v.singleBlockMaxKeys, tokens: chunks * C)
                core = try EncoderCore(programs: programs, layers: v.layers, heads: heads, tokens: n, rope: rope)
            }
            patches = n
            deepstackBlocks = v.deepstackBlocks
            totalSteps = core.steps
        }

        func step() throws -> Bool {
            if features != nil { return true }
            try core.step()
            completedSteps += 1
            guard core.finished else { return false }
            let merged = patches / 4, C = BidirLMMediaEncoder.chunk
            var rows = [Float16]()
            for out in core.outputs {
                guard let out else { throw Failure.invalidOutput("vision tail output is missing") }
                try HalfArrays.check(out, [1, BidirLMMediaEncoder.outDim, 1, C / 4], "features")
                rows += try HalfArrays.rows(out, count: C / 4)
            }
            var deepstack = [[Float16]]()
            for block in deepstackBlocks {
                guard let chunks = core.deepstack[block], chunks.count == core.chunks else {
                    throw Failure.invalidOutput("vision DeepStack features after block \(block) are missing")
                }
                var ds = [Float16]()
                for out in chunks { ds += try HalfArrays.rows(out, count: C / 4) }
                deepstack.append(Array(ds[0..<(merged * BidirLMMediaEncoder.outDim)]))
            }
            features = Features(rows: Array(rows[0..<(merged * BidirLMMediaEncoder.outDim)]), deepstack: deepstack)
            return true
        }
    }

    // MARK: - audio

    final class AudioRun: Run, @unchecked Sendable {
        private let encoder: BidirLMMediaEncoder
        private let config: BidirLMManifest.Audio
        private let mel: [Float]
        private let frames: Int
        private let melChunks: [Int]          // valid frames per 200-frame chunk
        private var tokenRows: [Float16] = [] // (tokens x 1024) front-end output
        private var frontIndex = 0
        private let frontCalls: Int
        private let front: MLModel
        let tokens: Int
        private var core: (any TowerCore)?
        private let chunks: Int
        private(set) var features: Features?
        private(set) var completedSteps = 0
        let totalSteps: Int

        init(encoder: BidirLMMediaEncoder, config a: BidirLMManifest.Audio, clip: PreparedAudio) throws {
            guard clip.frames > 0, clip.mel.count == a.melBins * clip.frames else {
                throw Failure.invalidInput("mel features do not match the frame count")
            }
            self.encoder = encoder
            config = a
            mel = clip.mel
            frames = clip.frames
            let count = (clip.frames + a.chunkFrames - 1) / a.chunkFrames
            melChunks = (0..<count).map { min(a.chunkFrames, clip.frames - $0 * a.chunkFrames) }
            tokens = AudioGeometry.tokens(frames: clip.frames)
            guard let maxKeys = a.keys.max(), tokens <= maxKeys else {
                throw Failure.invalidInput("\(tokens) audio tokens exceed the audio tower's maximum")
            }
            frontCalls = (count + a.frontBatch - 1) / a.frontBatch
            chunks = (tokens + BidirLMMediaEncoder.chunk - 1) / BidirLMMediaEncoder.chunk
            let frontFunctions: [ProgramResidency.Function] = encoder.staged.map { $0.programs("audio", ["front"]) }
                ?? [.init(path: a.frontModel, name: nil)]
            front = try encoder.residency.acquire("audio.front", frontFunctions).values.first!
            totalSteps = frontCalls + chunks * (1 + a.layers)
        }

        func step() throws -> Bool {
            if features != nil { return true }
            let C = BidirLMMediaEncoder.chunk, H = BidirLMMediaEncoder.hidden
            let a = config
            defer { completedSteps += 1 }
            if frontIndex < frontCalls {
                let B = a.frontBatch, F = a.chunkFrames, M = a.melBins
                let input = try HalfArrays.zeros([B, 1, M, F])
                let ptr = HalfArrays.pointer(input)
                for j in 0..<B {
                    let i = frontIndex * B + j
                    guard i < melChunks.count else { break }
                    for m in 0..<M {
                        let src = m * frames + i * F
                        let dst = (j * M + m) * F
                        for f in 0..<melChunks[i] { ptr[dst + f] = Float16(mel[src + f]) }
                    }
                }
                // The source pads a clip's chunks to its longest chunk; a lone chunk shorter than
                // 200 frames is not padded, so its convolutions must see zeros past the end.
                let mask1 = try HalfArrays.zeros([B, 1, 1, F / 2]), mask2 = try HalfArrays.zeros([B, 1, 1, F / 4])
                let m1 = HalfArrays.pointer(mask1), m2 = HalfArrays.pointer(mask2)
                for i in 0..<(B * F / 2) { m1[i] = 1 }
                for i in 0..<(B * F / 4) { m2[i] = 1 }
                if melChunks.count == 1, melChunks[0] < F {
                    let l1 = (melChunks[0] - 1) / 2 + 1, l2 = (l1 - 1) / 2 + 1
                    for i in l1..<(F / 2) { m1[i] = 0 }
                    for i in l2..<(F / 4) { m2[i] = 0 }
                }
                let out = try HalfArrays.output(
                    try BidirLMMediaEncoder.predict(front, ["mel": input, "mask1": mask1, "mask2": mask2]),
                                                "hidden_out", [1, H, 1, B * a.tokensPerChunk])
                let rows = try HalfArrays.rows(out, count: B * a.tokensPerChunk)
                for j in 0..<B {
                    let i = frontIndex * B + j
                    guard i < melChunks.count else { break }
                    let valid = AudioGeometry.tokens(frames: melChunks[i])
                    let start = j * a.tokensPerChunk * H
                    tokenRows += rows[start..<(start + valid * H)]
                }
                frontIndex += 1
                if frontIndex == frontCalls {
                    guard tokenRows.count == tokens * H else {
                        throw Failure.invalidOutput("audio front end produced \(tokenRows.count / H) tokens, expected \(tokens)")
                    }
                    var heads = [[String: MLMultiArray]]()
                    for c in 0..<chunks {
                        let x = try HalfArrays.zeros([1, H, 1, C])
                        let xp = HalfArrays.pointer(x)
                        for t in 0..<min(C, tokens - c * C) {
                            let src = (c * C + t) * H
                            for ch in 0..<H { xp[ch * C + t] = tokenRows[src + ch] }
                        }
                        heads.append(["hidden": x])
                    }
                    tokenRows = []
                    if encoder.staged != nil {
                        core = try StagedEncoderCore(programs: encoder.stagedPrograms(family: "audio", tokens: chunks * C),
                                                     layers: a.layers, heads: heads, tokens: tokens, rope: nil)
                    } else {
                        let programs = try encoder.programs(
                            family: "audio", tower: a.towerModel, layers: a.layers, attention: a.attentionModels,
                            keys: a.keys, keyBlock: a.keyBlock, singleBlockMaxKeys: a.singleBlockMaxKeys, tokens: chunks * C)
                        core = try EncoderCore(programs: programs, layers: a.layers, heads: heads, tokens: tokens, rope: nil)
                    }
                }
                return false
            }
            guard let core else { throw Failure.invalidOutput("audio run has no encoder state") }
            try core.step()
            guard core.finished else { return false }
            var output = [Float16]()
            for out in core.outputs {
                guard let out else { throw Failure.invalidOutput("audio tail output is missing") }
                try HalfArrays.check(out, [1, BidirLMMediaEncoder.outDim, 1, C], "features")
                output += try HalfArrays.rows(out, count: C)
            }
            features = Features(rows: Array(output[0..<(tokens * BidirLMMediaEncoder.outDim)]), deepstack: [])
            return true
        }
    }

    static func predict(_ model: MLModel, _ feeds: [String: MLMultiArray]) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feeds.mapValues { MLFeatureValue(multiArray: $0) }))
    }
}

/// One tower's layer-by-layer execution (legacy chunked programs or staged W8A16 functions).
protocol TowerCore: AnyObject {
    var chunks: Int { get }
    var steps: Int { get }
    var finished: Bool { get }
    var outputs: [MLMultiArray?] { get }
    var deepstack: [Int: [MLMultiArray]] { get }
    func step() throws
}

/// Host inputs for one image: normalized patches in the processor's 2x2 merge-window order.
struct PreparedImage: Sendable {
    /// (gridH * gridW) x 1536, row-major.
    let pixels: [Float]
    let gridH: Int
    let gridW: Int
    var tokens: Int { gridH * gridW / 4 }
}

/// Host inputs for one audio clip: log-mel features (128 x frames, row-major).
struct PreparedAudio: Sendable {
    let mel: [Float]
    let frames: Int
    var tokens: Int { AudioGeometry.tokens(frames: frames) }
}

enum AudioGeometry {
    /// Three stride-2 convolutions (kernel 3, padding 1): floor((L - 1) / 2) + 1, three times.
    static func tokens(frames: Int) -> Int {
        var length = frames
        for _ in 0..<3 { length = (length - 1) / 2 + 1 }
        return frames > 0 ? length : 0
    }
}

/// Host-side vision tables, evaluated in float64 (mirrors `ane_media.py`).
enum VisionTables {
    /// Bilinear interpolation of the learned `side x side` table to the patch grid, in 2x2
    /// merge-window order (`fast_pos_embed_interpolate`). Returns rows for `range` (row-major).
    static func positions(_ table: [Float], side: Int, hidden: Int, gridH: Int, gridW: Int, merge: Int,
                          range: Range<Int>) -> [Double] {
        func linspace(_ count: Int) -> [Double] {
            count == 1 ? [0] : (0..<count).map { Double($0) * Double(side - 1) / Double(count - 1) }
        }
        let hs = linspace(gridH), ws = linspace(gridW)
        var out = [Double](repeating: 0, count: range.count * hidden)
        let blocksW = gridW / merge
        for (row, p) in range.enumerated() {
            let (r, c) = mergeOrderCoordinates(p, blocksW: blocksW, merge: merge)
            let h = hs[r], w = ws[c]
            let hf = Int(h), wf = Int(w)
            let hc = min(hf + 1, side - 1), wc = min(wf + 1, side - 1)
            let dh = h - Double(hf), dw = w - Double(wf)
            let corners = [(hf * side + wf, (1 - dh) * (1 - dw)), (hf * side + wc, (1 - dh) * dw),
                           (hc * side + wf, dh * (1 - dw)), (hc * side + wc, dh * dw)]
            for (index, weight) in corners where weight != 0 {
                let base = index * hidden
                for ch in 0..<hidden { out[row * hidden + ch] += Double(table[base + ch]) * weight }
            }
        }
        return out
    }

    /// (row, column) of patch `p` in merge-window order: blocks row-major, 2x2 inside a block.
    static func mergeOrderCoordinates(_ p: Int, blocksW: Int, merge: Int) -> (Int, Int) {
        let inner = merge * merge
        let block = p / inner, within = p % inner
        return ((block / blocksW) * merge + within / merge, (block % blocksW) * merge + within % merge)
    }

    /// 2-D rotary tables (1, 1, 64, 512) per chunk: 16 frequencies for rows, then 16 for
    /// columns, duplicated (rotate-half layout).
    static func rope(gridH: Int, gridW: Int, merge: Int, theta: Double, chunks: Int) throws -> [(MLMultiArray, MLMultiArray)] {
        let C = BidirLMMediaEncoder.chunk, D = BidirLMMediaEncoder.headDim
        let quarter = D / 4
        let inv = (0..<quarter).map { 1.0 / pow(theta, Double(2 * $0) / Double(D / 2)) }
        let n = gridH * gridW
        var tables = [(MLMultiArray, MLMultiArray)]()
        for c in 0..<chunks {
            let cos = try HalfArrays.zeros([1, 1, D, C]), sin = try HalfArrays.zeros([1, 1, D, C])
            let cp = HalfArrays.pointer(cos), sp = HalfArrays.pointer(sin)
            for t in 0..<C {
                let p = c * C + t
                guard p < n else { break }
                let (r, col) = mergeOrderCoordinates(p, blocksW: gridW / merge, merge: merge)
                for i in 0..<(2 * quarter) {
                    let angle = i < quarter ? Double(r) * inv[i] : Double(col) * inv[i - quarter]
                    let cv = Float16(Foundation.cos(angle)), sv = Float16(Foundation.sin(angle))
                    cp[i * C + t] = cv; cp[(i + 2 * quarter) * C + t] = cv
                    sp[i * C + t] = sv; sp[(i + 2 * quarter) * C + t] = sv
                }
            }
            tables.append((cos, sin))
        }
        return tables
    }
}
