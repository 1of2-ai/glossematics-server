import CoreML
import Foundation

/// Host plan for one language-model execution. Mirrors `plan_for` in `bidirlm_coreml/ane_text.py`.
struct BidirLMTextPlan: Equatable, Sendable {
    let chunk: Int
    let chunks: Int
    let keys: Int
    let packed: Bool

    enum Failure: Error, CustomStringConvertible, Equatable {
        case empty
        case tooLong(Int, Int)
        case tooMany(Int, Int)
        case longMustBeAlone

        var description: String {
            switch self {
            case .empty: "every sequence needs at least one token"
            case let .tooLong(n, max): "sequence has \(n) tokens; maximum is \(max)"
            case let .tooMany(n, max): "\(n) sequences exceed the \(max)-sequence pack"
            case .longMustBeAlone: "sequences longer than one chunk are encoded alone"
            }
        }
    }

    static func make(lengths: [Int]) throws -> BidirLMTextPlan {
        guard !lengths.isEmpty, lengths.allSatisfy({ $0 > 0 }) else { throw Failure.empty }
        if let longest = lengths.max(), longest > BidirLMContract.maxTokens {
            throw Failure.tooLong(longest, BidirLMContract.maxTokens)
        }
        guard lengths.count <= BidirLMContract.poolWidth else {
            throw Failure.tooMany(lengths.count, BidirLMContract.poolWidth)
        }
        let total = lengths.reduce(0, +)
        for chunk in BidirLMContract.chunks where total <= chunk {
            return .init(chunk: chunk, chunks: 1, keys: chunk, packed: true)
        }
        guard lengths.count == 1 else { throw Failure.longMustBeAlone }
        let chunk = BidirLMContract.chunks.last!
        let chunks = (total + chunk - 1) / chunk
        let keys = BidirLMContract.longKeys.first { $0 >= chunks * chunk }!
        return .init(chunk: chunk, chunks: chunks, keys: keys, packed: false)
    }
}

/// One language-model input: a complete, already-templated token sequence, optionally with media
/// feature rows replacing placeholder tokens, 3-D MRoPE positions, and DeepStack rows.
struct LMSequence: Sendable {
    struct Replacement: Sendable {
        let start: Int
        /// Row-major FP16 feature rows (count x 2048).
        let rows: [Float16]
    }

    var ids: [Int32]
    var replacements: [Replacement] = []
    /// (t, h, w) per token; nil means text positions 0..<n on all three axes.
    var positions: [(Int32, Int32, Int32)]? = nil
    /// Visual token positions receiving DeepStack rows after layers 0...(k-1).
    var deepstackPositions: [Int] = []
    /// Per DeepStack layer, row-major FP16 rows (deepstackPositions.count x 2048).
    var deepstack: [[Float16]] = []

    init(ids: [Int32]) { self.ids = ids }

    var count: Int { ids.count }
    var hasDeepstack: Bool { !deepstack.isEmpty }

    func position(_ index: Int) -> (Int32, Int32, Int32) {
        if let positions { return positions[index] }
        let p = Int32(index)
        return (p, p, p)
    }
}

/// Chunked BidirLM language model over the compiled bundle functions.
///
/// Packed inputs (at most 512 tokens, up to 64 sequences) run through four resident stack
/// programs of seven whole layers each. Longer inputs run layer by layer over 512-token chunks:
/// `head_c512` projects layer 0, each `mid_c512_l{i}` finishes layer i and projects layer i+1,
/// and `tail_c512` finishes layer 27 and pools; attention against all keys runs in a weightless
/// bucket function between them. Long-input functions are loaded on demand through
/// `ProgramResidency`, which bounds this process's share of the Neural Engine's program slots.
///
/// Neural work runs in Core ML; the host performs only token-table lookup, media-row placement,
/// RoPE table evaluation (float64), mask/pooling matrix construction, and tensor copies. Not
/// thread-safe: the owning backend serializes calls on the accelerator lane.
final class BidirLMTextEncoder: TextModelEncoder, @unchecked Sendable {
    enum Failure: Error, CustomStringConvertible {
        case tokenOutOfRange(Int32)
        case invalidOutput(String)
        case invalidSequence(String)

        var description: String {
            switch self {
            case let .tokenOutOfRange(id): "token id \(id) is outside the vocabulary"
            case let .invalidOutput(reason), let .invalidSequence(reason): reason
            }
        }
    }

    static let stackSet = "text.stacks"
    static let longSet = "text.long"

    private let residency: ProgramResidency
    private let text: BidirLMManifest.Text
    private let table: Data
    private let vocabulary: Int
    private let padID: Int32
    private let stacks: [[Int: MLModel]]           // [group][chunk]
    private let inverseFrequency: [Double]
    /// MRoPE axis (0 = t, 1 = h, 2 = w) for each of the 64 rotary frequencies (interleaved layout).
    private let ropeAxis: [Int]
    fileprivate let hidden: Int
    fileprivate let negative: Float16
    fileprivate let deepstackLayers: Int
    private var zeroRows: [String: MLMultiArray] = [:]

    init(residency: ProgramResidency, manifest: BidirLMManifest) throws {
        self.residency = residency
        text = manifest.text
        hidden = manifest.text.hidden
        padID = manifest.padTokenID
        negative = Float16(manifest.text.maskNegative)
        vocabulary = manifest.tokenEmbeddings.shape[0]
        deepstackLayers = manifest.text.deepstackLayers
        table = try Data(contentsOf: residency.root.appendingPathComponent(manifest.tokenEmbeddings.file),
                         options: .alwaysMapped)
        let half = manifest.text.headDim / 2
        inverseFrequency = (0..<half).map {
            1.0 / pow(manifest.text.ropeTheta, Double(2 * $0) / Double(manifest.text.headDim))
        }
        var axis = [Int](repeating: 0, count: half)
        let section = manifest.text.mropeSection
        for (dim, offset) in [(1, 1), (2, 2)] where section.count == 3 {
            var i = offset
            while i < section[dim] * 3 { axis[i] = dim; i += 3 }
        }
        ropeAxis = axis
        var wanted = [ProgramResidency.Function]()
        for path in manifest.text.groupModels {
            for chunk in manifest.text.chunks { wanted.append(.init(path: path, name: "stack_c\(chunk)")) }
        }
        let models = try residency.acquire(Self.stackSet, wanted, pinned: true)
        stacks = manifest.text.groupModels.map { path in
            Dictionary(uniqueKeysWithValues: manifest.text.chunks.map { ($0, models[.init(path: path, name: "stack_c\($0)")]!) })
        }
    }

    /// Long-input functions of every group, in execution order: head, mids 0...26, tail.
    fileprivate func longFunctions() -> [ProgramResidency.Function] {
        var out = [ProgramResidency.Function]()
        for (g, path) in text.groupModels.enumerated() {
            for name in BidirLMContract.groupFunctions(g) where !name.hasPrefix("stack_") {
                out.append(.init(path: path, name: name))
            }
        }
        return out
    }

    fileprivate func acquireLong(keys: Int) throws -> (head: MLModel, mids: [MLModel], tail: MLModel, attention: MLModel) {
        let functions = longFunctions()
        let attentionName = BidirLMContract.attentionName(chunk: 512, keys: keys, packed: false)
        guard let attentionPath = text.attentionModels[attentionName] else {
            throw Failure.invalidOutput("bundle has no \(attentionName) attention model")
        }
        let attentionSet = "text.attn.\(keys)"
        let long = try residency.acquire(Self.longSet, functions, inUse: [attentionSet])
        let attention = try residency.acquire(attentionSet, [.init(path: attentionPath, name: nil)], inUse: [Self.longSet])
        let ordered = functions.map { long[$0]! }
        return (ordered.first!, Array(ordered.dropFirst().dropLast()), ordered.last!, attention.values.first!)
    }

    /// Embed complete, already-templated token sequences. Returns L2-normalized vectors.
    func encode(_ rows: [[Int32]]) throws -> [[Float]] {
        try encode(sequences: rows.map { LMSequence(ids: $0) })
    }

    /// Embed one plan-compatible group of sequences (packed), or one long sequence.
    func encode(sequences: [LMSequence]) throws -> [[Float]] {
        for sequence in sequences { try validate(sequence) }
        let plan = try BidirLMTextPlan.make(lengths: sequences.map(\.count))
        if plan.packed {
            let embedding = try runPacked(sequences, plan)
            return try (0..<sequences.count).map { try normalizedColumn(embedding, $0) }
        }
        let run = try LongRun(encoder: self, sequence: sequences[0], plan: plan)
        while try !run.step() { try Task.checkCancellation() }
        return [try run.result()]
    }

    /// Start a resumable long execution (more than 512 tokens). The scheduler advances it one
    /// chunk operation at a time so short requests interleave with a multi-minute 32K document.
    func startLong(_ sequence: LMSequence) throws -> any TextModelRun {
        try validate(sequence)
        let plan = try BidirLMTextPlan.make(lengths: [sequence.count])
        guard !plan.packed else { throw Failure.invalidOutput("sequence fits one chunk; use encode") }
        return try LongRun(encoder: self, sequence: sequence, plan: plan)
    }

    func startLong(_ row: [Int32]) throws -> any TextModelRun { try startLong(LMSequence(ids: row)) }

    private func validate(_ s: LMSequence) throws {
        for id in s.ids where id < 0 || Int(id) >= vocabulary { throw Failure.tokenOutOfRange(id) }
        for r in s.replacements {
            guard r.rows.count.isMultiple(of: hidden), r.start >= 0,
                  r.start + r.rows.count / hidden <= s.count else {
                throw Failure.invalidSequence("media rows do not fit the token sequence")
            }
        }
        if let positions = s.positions, positions.count != s.count {
            throw Failure.invalidSequence("position count does not match the token sequence")
        }
        guard s.deepstack.isEmpty || s.deepstack.count == deepstackLayers else {
            throw Failure.invalidSequence("expected \(deepstackLayers) DeepStack feature sets")
        }
        for rows in s.deepstack where rows.count != s.deepstackPositions.count * hidden {
            throw Failure.invalidSequence("DeepStack rows do not match their positions")
        }
        guard s.deepstackPositions.allSatisfy({ (0..<s.count).contains($0) }) else {
            throw Failure.invalidSequence("DeepStack position outside the sequence")
        }
    }

    /// Cached zero DeepStack rows, one array per (width, layer). Never pass one array to two
    /// inputs of the same call: measured, the GPU backend (MPSGraph) aborts on aliased buffers.
    fileprivate func zeros(_ width: Int, layer: Int) throws -> MLMultiArray {
        let key = "\(width).\(layer)"
        if let cached = zeroRows[key] { return cached }
        let array = try HalfArrays.zeros([1, hidden, 1, width])
        zeroRows[key] = array
        return array
    }

    // MARK: - packed (one chunk, one or more sequences)

    private func runPacked(_ sequences: [LMSequence], _ plan: BidirLMTextPlan) throws -> MLMultiArray {
        let C = plan.chunk
        let x0 = try HalfArrays.zeros([1, hidden, 1, C])
        let pool = try HalfArrays.zeros([1, C, BidirLMContract.poolWidth])
        let poolPtr = HalfArrays.pointer(pool)
        var owner = [Int](repeating: -1, count: C)
        var positions = [(Int32, Int32, Int32)](repeating: (0, 0, 0), count: C)
        var ids = [Int32](repeating: padID, count: C)
        var offset = 0
        let useDeepstack = sequences.contains(where: \.hasDeepstack)
        let deepstackArrays = useDeepstack
            ? try (0..<deepstackLayers).map { _ in try HalfArrays.zeros([1, hidden, 1, C]) } : []
        for (p, s) in sequences.enumerated() {
            let weight = Float16(1.0 / Double(s.count))
            for i in 0..<s.count {
                ids[offset + i] = s.ids[i]
                positions[offset + i] = s.position(i)
                owner[offset + i] = p
                poolPtr[(offset + i) * BidirLMContract.poolWidth + p] = weight
            }
            offset += s.count
        }
        fillEmbeddings(x0, ids: ids, columnOffset: 0)
        offset = 0
        for s in sequences {
            place(s, into: x0, deepstack: deepstackArrays, tokenOffset: offset, chunkStart: 0, width: C)
            offset += s.count
        }
        let bias = try HalfArrays.zeros([1, 1, C, 2 * C])
        let biasPtr = HalfArrays.pointer(bias)
        for k in 0..<C {
            for t in 0..<C {
                let value: Float16 = owner[k] >= 0 && owner[k] == owner[t] ? 0 : negative
                biasPtr[k * 2 * C + t] = value
                biasPtr[k * 2 * C + C + t] = value
            }
        }
        let (cos, sin) = try rope(positions)
        var x = x0
        for (g, stack) in stacks.enumerated() {
            try Task.checkCancellation()
            var feeds: [String: MLMultiArray] = ["hidden": x, "cos": cos, "sin": sin, "bias": bias]
            for i in (g * text.stackLayers)..<((g + 1) * text.stackLayers) where i < deepstackLayers {
                feeds["deepstack_\(i)"] = useDeepstack ? deepstackArrays[i] : try zeros(C, layer: i)
            }
            if g == stacks.count - 1 {
                feeds["pool"] = pool
                feeds["carry"] = try HalfArrays.zeros([1, hidden, BidirLMContract.poolWidth])
                return try HalfArrays.output(try predict(stack[C]!, feeds), "embedding",
                                             [1, hidden, BidirLMContract.poolWidth])
            }
            x = try HalfArrays.output(try predict(stack[C]!, feeds), "hidden_out", [1, hidden, 1, C])
        }
        throw Failure.invalidOutput("execution produced no embedding")
    }

    // MARK: - long (one sequence, many chunks)

    /// State of one long sequence between steps. The first N steps project layer 0 for each
    /// chunk; after that a step is one chunk's attention for layer i followed by that chunk's
    /// `mid_c512_l{i}` (or `tail_c512` for the last layer): N + 28 N steps in total. Keys and
    /// values alternate between two buffers so layer i+1's projections never overwrite keys that
    /// layer i's remaining chunks still attend to.
    final class LongRun: TextModelRun, @unchecked Sendable {
        let plan: BidirLMTextPlan
        let tokens: Int
        private unowned let encoder: BidirLMTextEncoder
        private let head: MLModel
        private let mids: [MLModel]
        private let tail: MLModel
        private let attention: MLModel
        private var xs: [MLMultiArray]
        private let tables: [(MLMultiArray, MLMultiArray)]
        private let deepstack: [[MLMultiArray]]          // [layer][chunk]
        private let keys: [[MLMultiArray]]               // [buffer][block]
        private let values: [[MLMultiArray]]
        private let biases: [MLMultiArray]
        private let block: Int
        private var queries: [MLMultiArray?]
        private var carry: MLMultiArray
        private(set) var embedding: MLMultiArray?
        private var headed = 0
        private var layer = 0
        private var chunk = 0

        var totalSteps: Int { plan.chunks * (1 + BidirLMContract.layers) }
        private(set) var completedSteps = 0

        fileprivate init(encoder e: BidirLMTextEncoder, sequence s: LMSequence, plan: BidirLMTextPlan) throws {
            encoder = e
            self.plan = plan
            tokens = s.count
            let C = plan.chunk, N = plan.chunks, S = plan.keys, n = s.count
            (head, mids, tail, attention) = try e.acquireLong(keys: S)
            block = min(S, e.text.keyBlock)
            var ids = [Int32](repeating: e.padID, count: N * C)
            var positions = [(Int32, Int32, Int32)](repeating: (0, 0, 0), count: N * C)
            for i in 0..<n { ids[i] = s.ids[i]; positions[i] = s.position(i) }
            for i in n..<(N * C) { let p = Int32(i); positions[i] = (p, p, p) }
            var chunks = [MLMultiArray]()
            var dsChunks = [[MLMultiArray]](repeating: [], count: e.deepstackLayers)
            for c in 0..<N {
                let x = try HalfArrays.zeros([1, e.hidden, 1, C])
                e.fillEmbeddings(x, ids: Array(ids[(c * C)..<((c + 1) * C)]), columnOffset: 0)
                let ds = s.hasDeepstack
                    ? try (0..<e.deepstackLayers).map { _ in try HalfArrays.zeros([1, e.hidden, 1, C]) }
                    : try (0..<e.deepstackLayers).map { try e.zeros(C, layer: $0) }
                e.place(s, into: x, deepstack: s.hasDeepstack ? ds : [], tokenOffset: 0, chunkStart: c * C, width: C)
                chunks.append(x)
                for (k, array) in ds.enumerated() { dsChunks[k].append(array) }
            }
            xs = chunks
            deepstack = dsChunks
            tables = try (0..<N).map { c in try e.rope(Array(positions[(c * C)..<((c + 1) * C)])) }
            let width = min(S, e.text.keyBlock)
            let shape = [1, e.text.kvHeads, e.text.headDim, width]
            keys = try (0..<2).map { _ in try (0..<(S / width)).map { _ in try HalfArrays.zeros(shape) } }
            values = try (0..<2).map { _ in try (0..<(S / width)).map { _ in try HalfArrays.zeros(shape) } }
            biases = try (0..<(S / width)).map { b -> MLMultiArray in
                let array = try HalfArrays.zeros([1, 1, width, 1])
                let ptr = HalfArrays.pointer(array)
                for i in 0..<width where b * width + i >= n { ptr[i] = e.negative }
                return array
            }
            queries = Array(repeating: nil, count: N)
            carry = try HalfArrays.zeros([1, e.hidden, BidirLMContract.poolWidth])
        }

        private func store(_ out: MLFeatureProvider, buffer: Int, chunk c: Int) throws {
            let C = plan.chunk
            let heads = [1, encoder.text.kvHeads, encoder.text.headDim]
            queries[c] = try HalfArrays.output(out, "query", heads + [2 * C])
            let start = c * C
            HalfArrays.copyColumns(from: try HalfArrays.output(out, "key", heads + [C]),
                                   into: keys[buffer][start / block], columnOffset: start % block)
            HalfArrays.copyColumns(from: try HalfArrays.output(out, "value", heads + [C]),
                                   into: values[buffer][start / block], columnOffset: start % block)
        }

        /// Advance one chunk operation. Returns true when the embedding is ready.
        func step() throws -> Bool {
            if embedding != nil { return true }
            let e = encoder, C = plan.chunk, N = plan.chunks
            defer { completedSteps += 1 }
            if headed < N {
                let c = headed
                try store(try e.predict(head, ["hidden": xs[c], "cos": tables[c].0, "sin": tables[c].1]), buffer: 0, chunk: c)
                headed += 1
                return false
            }
            let current = layer % 2, next = (layer + 1) % 2
            var feeds: [String: MLMultiArray] = ["query": queries[chunk]!]
            for b in 0..<keys[current].count {
                feeds["key_\(b)"] = keys[current][b]
                feeds["value_\(b)"] = values[current][b]
                feeds["bias_\(b)"] = biases[b]
            }
            let heads = [1, e.text.kvHeads, e.text.headDim]
            let att = try HalfArrays.output(try e.predict(attention, feeds), "attention", heads + [2 * C])
            if layer == BidirLMContract.layers - 1 {
                let pool = try HalfArrays.zeros([1, C, BidirLMContract.poolWidth])
                let ptr = HalfArrays.pointer(pool)
                // Chunk-scaled sums (1/C is a normal FP16 value); L2 normalization removes
                // the common scale, so the result equals the masked mean over all tokens.
                let real = max(0, min(C, tokens - chunk * C))
                for t in 0..<real { ptr[t * BidirLMContract.poolWidth] = Float16(1.0 / Double(C)) }
                let out = try e.predict(tail, ["attention": att, "hidden": xs[chunk], "pool": pool, "carry": carry])
                carry = try HalfArrays.output(out, "carry_out", [1, e.hidden, BidirLMContract.poolWidth])
                if chunk == N - 1 {
                    embedding = try HalfArrays.output(out, "embedding", [1, e.hidden, BidirLMContract.poolWidth])
                }
            } else {
                var midFeeds: [String: MLMultiArray] = ["attention": att, "hidden": xs[chunk],
                                                        "cos": tables[chunk].0, "sin": tables[chunk].1]
                if layer < e.deepstackLayers { midFeeds["deepstack_in"] = deepstack[layer][chunk] }
                let out = try e.predict(mids[layer], midFeeds)
                xs[chunk] = try HalfArrays.output(out, "hidden_out", [1, e.hidden, 1, C])
                try store(out, buffer: next, chunk: chunk)
            }
            chunk += 1
            if chunk == N { chunk = 0; layer += 1 }
            return embedding != nil
        }

        /// The normalized embedding once `step()` has returned true.
        func result() throws -> [Float] {
            guard let embedding else { throw Failure.invalidOutput("long run is not finished") }
            return try encoder.normalizedColumn(embedding, 0)
        }
    }

    // MARK: - host tensors

    fileprivate func predict(_ model: MLModel, _ feeds: [String: MLMultiArray]) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feeds.mapValues { MLFeatureValue(multiArray: $0) }))
    }

    /// Write token-table rows for `ids` into columns of a (1, hidden, 1, C) array.
    fileprivate func fillEmbeddings(_ array: MLMultiArray, ids: [Int32], columnOffset: Int) {
        let C = array.shape[3].intValue
        let out = HalfArrays.pointer(array)
        table.withUnsafeBytes { raw in
            let rows = raw.bindMemory(to: Float16.self)
            for (t, id) in ids.enumerated() {
                let base = Int(id) * hidden
                for ch in 0..<hidden { out[ch * C + columnOffset + t] = rows[base + ch] }
            }
        }
    }

    /// Overwrite media placeholder columns and fill DeepStack columns for sequence `s`, whose
    /// token 0 sits at packed column `tokenOffset`, restricted to tokens in
    /// [chunkStart, chunkStart + width) of the sequence.
    fileprivate func place(_ s: LMSequence, into x: MLMultiArray, deepstack: [MLMultiArray],
                       tokenOffset: Int, chunkStart: Int, width: Int) {
        let out = HalfArrays.pointer(x)
        for r in s.replacements {
            let count = r.rows.count / hidden
            for j in 0..<count {
                let token = r.start + j
                guard token >= chunkStart, token < chunkStart + width else { continue }
                let column = tokenOffset + token - chunkStart
                for ch in 0..<hidden { out[ch * width + column] = r.rows[j * hidden + ch] }
            }
        }
        guard !deepstack.isEmpty, s.hasDeepstack else { return }
        for (k, rows) in s.deepstack.enumerated() {
            let ptr = HalfArrays.pointer(deepstack[k])
            for (j, token) in s.deepstackPositions.enumerated() {
                guard token >= chunkStart, token < chunkStart + width else { continue }
                let column = tokenOffset + token - chunkStart
                for ch in 0..<hidden { ptr[ch * width + column] = rows[j * hidden + ch] }
            }
        }
    }

    /// MRoPE tables (1, 1, 128, C) evaluated in float64 with the interleaved axis layout. For
    /// text all three axes are equal and this reduces to standard RoPE.
    fileprivate func rope(_ positions: [(Int32, Int32, Int32)]) throws -> (MLMultiArray, MLMultiArray) {
        let C = positions.count, dim = text.headDim, half = dim / 2
        let cos = try HalfArrays.zeros([1, 1, dim, C]), sin = try HalfArrays.zeros([1, 1, dim, C])
        let cp = HalfArrays.pointer(cos), sp = HalfArrays.pointer(sin)
        for (t, p) in positions.enumerated() {
            let axes = [Double(p.0), Double(p.1), Double(p.2)]
            for i in 0..<half {
                let angle = axes[ropeAxis[i]] * inverseFrequency[i]
                let c = Float16(Foundation.cos(angle)), s = Float16(Foundation.sin(angle))
                cp[i * C + t] = c; cp[(i + half) * C + t] = c
                sp[i * C + t] = s; sp[(i + half) * C + t] = s
            }
        }
        return (cos, sin)
    }

    fileprivate func normalizedColumn(_ embedding: MLMultiArray, _ column: Int) throws -> [Float] {
        let values = try HalfArrays.column(embedding, column)
        var norm: Double = 0
        for v in values { norm += Double(v) * Double(v) }
        norm = norm.squareRoot()
        guard norm.isFinite, abs(norm - 1) <= 0.01 else {
            throw Failure.invalidOutput("embedding column \(column) has norm \(norm)")
        }
        return values.map { Float(Double($0) / norm) }
    }
}
