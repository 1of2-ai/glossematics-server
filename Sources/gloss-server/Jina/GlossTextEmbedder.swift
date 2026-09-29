import CoreML
import Foundation

/// The native batch functions execute their entire fixed shape, including padded rows. On the
/// production M4 Max bundle, a warmed b64@32 call takes ~240 ms regardless of whether it has 2
/// or 64 real rows, while a single-row bucket_32 call takes ~10 ms. The other crossovers were
/// measured the same way; conservative thresholds avoid making sparse requests pay for padding.
/// Unknown batch geometries use a full batch rather than assuming this profile applies to them.
enum NativeTextBatchPolicy {
    static func minimumRows(for capacity: Int, bucket: Int) -> Int {
        switch (capacity, bucket) {
        case (64, 32): 24
        case (32, 64): 24
        case (16, 128): 16
        case (8, 256): 6
        case (4, 512): 3
        default: max(1, capacity)
        }
    }
}

/// High-level text embedding API: raw text -> task prefix -> tokenize -> Core ML encode
/// -> L2-normalized embedding (optionally Matryoshka-truncated).
///
/// Two modes:
///  - single fixed-shape model (`init(modelURL:...)`): one bucket.
///  - multi-function model (`init(multiFunctionModelURL:...)`): functions `bucket_<S>` sharing
///    weights; the smallest fitting bucket is selected per text and lazily loaded.
internal final class GlossTextEmbedder {
    /// Which task prefix to prepend. The actual strings come from ``PromptStrings`` (bundles carry
    /// them in the manifest's `prompts` section); this enum is only a selector.
    public enum Prompt: Sendable {
        case query, document, none
    }

    /// The task prompt prefixes. Defaults are the jina-embeddings v5 family recipe (identical
    /// across the family today); bundles override them from `manifest.prompts`.
    public struct PromptStrings: Sendable {
        public var query: String
        public var document: String

        public init(query: String = "Query: ", document: String = "Document: ") {
            self.query = query
            self.document = document
        }
    }

    public let tokenizer: GlossTokenizer
    public let prompts: PromptStrings
    /// Task-specific instruction pairs (code models) keyed by task name; `embed(_, task:)` selects
    /// them. `nil`/empty for bundles without tasks. The main `prompts` pair is the default task.
    public let taskPrompts: [String: PromptStrings]
    public let padTokenID: Int32
    /// Batch-N function geometry from the bundle manifest (`text.batch`); nil = no batch functions
    /// in the package, in which case `embed(texts:)` falls back to the single-row loop.
    ///
    /// Two shapes are supported: a legacy single size (`batchSize` over `batchBuckets`) and a
    /// ladder (`batchSizes[i]` over `batchBuckets[i]`). ``batchPairs`` is the resolved list of
    /// (batchSize, bucket) functions the runtime actually selects among.
    public let batchSize: Int?
    public let batchBuckets: [Int]
    public let batchSizes: [Int]
    private let batchPairs: [(size: Int, bucket: Int)]

    enum EmbeddingPlanStep: Equatable {
        /// Batch-N encode over the listed original row indices (ascending), padded to `size`.
        case batch(rows: [Int], size: Int, bucket: Int)
        /// Single-row encode for the original row index.
        case single(index: Int, bucket: Int)
    }

    private let compiledURL: URL
    /// `nil` = ADAPTIVE placement (measured on small): ANE for buckets ≤128, GPU for ≥256. The text
    /// decoder is shallow (accurate on either), and has a latency crossover — ANE 7.5/12.3 ms vs GPU
    /// 9.6/15.8 ms at 32/128, but ANE 29/84 ms vs GPU 25/47 ms at 256/512.
    private let forcedUnits: MLComputeUnits?
    private let isMultiFunction: Bool
    private let buckets: [Int]              // sorted ascending
    /// Init-enforced largest bucket; bucket lookup never force-unwraps.
    private let maximumBucket: Int
    /// Loaded functions by bucket (single-row) or batch key. Each key loads exactly once and never
    /// while holding a lock other keys need, so a slow `bucket_32768` load cannot stall `bucket_32`.
    private let cache = KeyedLoadCache<Int, CoreMLTextEncoder>()

    /// Latency-optimal unit for a length bucket (ANE ≤128, GPU ≥256), unless forced.
    func units(forBucket b: Int) -> MLComputeUnits { forcedUnits ?? (b <= 128 ? .cpuAndNeuralEngine : .cpuAndGPU) }

    /// Largest sequence length this embedder can handle (longer is truncated keep-first).
    public var maxTokens: Int { maximumBucket }
    public var availableBuckets: [Int] { buckets }

    /// Single fixed-shape model.
    public convenience init(
        modelURL: URL, tokenizerFolder: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        prompts: PromptStrings = PromptStrings(),
        padTokenID: Int32 = 0
    ) async throws {
        let enc = try CoreMLTextEncoder(modelURL: modelURL, computeUnits: computeUnits, padTokenID: padTokenID)
        try await self.init(_compiledURL: enc.compiledURL, tokenizerFolder: tokenizerFolder,
                            forcedUnits: computeUnits, isMultiFunction: false,
                            buckets: [enc.seqLen], preloaded: [enc.seqLen: enc],
                            prompts: prompts, padTokenID: padTokenID)
    }

    /// Multi-function model: functions named `bucket_<S>` for each S in `buckets`.
    /// `computeUnits: nil` (default) = ADAPTIVE placement (ANE for short buckets, GPU for long);
    /// pass a value to force all buckets onto one unit.
    ///
    /// Batch geometry: pass either a legacy single size (`batchSize` over `batchBuckets`) or a
    /// ladder (`batchSizes[i]` over `batchBuckets[i]`, e.g. sizes `[64, 32, 16, 8, 4]` at buckets
    /// `[32, 64, 128, 256, 512]`). When both are present the ladder wins.
    public convenience init(
        multiFunctionModelURL: URL, tokenizerFolder: URL,
        buckets: [Int] = [32, 64, 128, 256, 512],
        computeUnits: MLComputeUnits? = nil,
        prompts: PromptStrings = PromptStrings(),
        padTokenID: Int32 = 0,
        batchSize: Int? = nil,
        batchBuckets: [Int] = [],
        batchSizes: [Int] = [],
        taskPrompts: [String: PromptStrings] = [:]
    ) async throws {
        let compiled: URL
        if multiFunctionModelURL.pathExtension == "mlmodelc" {
            compiled = multiFunctionModelURL
        } else {
            compiled = try await MLModel.compileModel(at: multiFunctionModelURL)
        }
        try await self.init(_compiledURL: compiled, tokenizerFolder: tokenizerFolder,
                            forcedUnits: computeUnits, isMultiFunction: true,
                            buckets: buckets.sorted(), preloaded: [:],
                            prompts: prompts, padTokenID: padTokenID,
                            batchSize: batchSize, batchBuckets: batchBuckets,
                            batchSizes: batchSizes, taskPrompts: taskPrompts)
    }

    private init(
        _compiledURL: URL, tokenizerFolder: URL, forcedUnits: MLComputeUnits?,
        isMultiFunction: Bool, buckets: [Int], preloaded: [Int: CoreMLTextEncoder],
        prompts: PromptStrings, padTokenID: Int32,
        batchSize: Int? = nil, batchBuckets: [Int] = [], batchSizes: [Int] = [],
        taskPrompts: [String: PromptStrings] = [:]
    ) async throws {
        self.tokenizer = try await GlossTokenizer(modelFolder: tokenizerFolder)
        guard let maximumBucket = buckets.max() else {
            throw CoreMLTextEncoder.EncoderError.badModel("bucket set must not be empty")
        }
        self.compiledURL = _compiledURL
        self.forcedUnits = forcedUnits
        self.isMultiFunction = isMultiFunction
        self.buckets = buckets
        self.maximumBucket = maximumBucket
        for (bucket, encoder) in preloaded { cache.prime(encoder, for: bucket) }
        self.prompts = prompts
        self.taskPrompts = taskPrompts
        self.padTokenID = padTokenID
        self.batchSize = batchSize
        self.batchBuckets = batchBuckets.sorted()
        self.batchSizes = batchSizes
        self.batchPairs = Self.resolveBatchPairs(batchSize: batchSize, batchSizes: batchSizes,
                                                 batchBuckets: batchBuckets.sorted())
    }

    /// Resolve the manifest's batch geometry into (batchSize, bucket) functions, ladder-preferred.
    private static func resolveBatchPairs(batchSize: Int?, batchSizes: [Int],
                                          batchBuckets: [Int]) -> [(size: Int, bucket: Int)] {
        if !batchSizes.isEmpty, batchSizes.count == batchBuckets.count {
            return zip(batchSizes, batchBuckets).map { (size: $0, bucket: $1) }
        }
        if let batchSize {
            return batchBuckets.map { (size: batchSize, bucket: $0) }
        }
        return []
    }

    /// Plan the whole batch by grouping rows on batch functions rather than on input position.
    /// Every row joins the widest batch whose bucket fits it (b64@32, b32@64, b16@128, b8@256,
    /// b4@512), wherever it sits in the input, so interleaved short and long rows still amortize
    /// batch calls. Rows longer than every batch bucket fall back to the same single-row bucket
    /// policy as `embed(_:)`. Execution order is free because the caller reassembles results by
    /// original index; each row appears in exactly one step.
    static func embeddingPlan(
        tokenCounts: [Int],
        buckets: [Int],
        batchPairs: [(size: Int, bucket: Int)]
    ) -> [EmbeddingPlanStep] {
        guard let largestSingleBucket = buckets.max() else { return [] }

        func singleBucket(for tokenCount: Int) -> Int {
            buckets.first { $0 >= tokenCount } ?? largestSingleBucket
        }

        let pairs = batchPairs
            .filter { $0.size > 0 && $0.bucket > 0 }
            .sorted {
                if $0.size == $1.size { return $0.bucket < $1.bucket }
                return $0.size > $1.size
            }
        guard tokenCounts.count > 1,
              let largestBatchBucket = pairs.map(\.bucket).max() else {
            return tokenCounts.indices.map {
                .single(index: $0, bucket: singleBucket(for: tokenCounts[$0]))
            }
        }

        // Assign every batchable row to the first pair (largest size) whose bucket fits it.
        var grouped = [Int: [Int]]()          // pair bucket -> original row indices, ascending
        var groupOrder = [Int]()              // pair buckets in first-appearance order
        var singles = [(index: Int, bucket: Int)]()
        for (index, count) in tokenCounts.enumerated() {
            guard count <= largestBatchBucket,
                  let pair = pairs.first(where: { $0.bucket >= count }) else {
                singles.append((index, singleBucket(for: count)))
                continue
            }
            if grouped[pair.bucket] == nil { groupOrder.append(pair.bucket) }
            grouped[pair.bucket, default: []].append(index)
        }

        var plan = [EmbeddingPlanStep]()
        plan.reserveCapacity(groupOrder.count + singles.count)
        for bucket in groupOrder {
            guard let pair = pairs.first(where: { $0.bucket == bucket }),
                  let rows = grouped[bucket] else { continue }
            var start = 0
            while start < rows.count {
                let end = min(start + pair.size, rows.count)
                let chunk = Array(rows[start..<end])
                if chunk.count >= NativeTextBatchPolicy.minimumRows(
                    for: pair.size, bucket: pair.bucket) {
                    plan.append(.batch(rows: chunk, size: pair.size, bucket: pair.bucket))
                } else {
                    plan.append(contentsOf: chunk.map {
                        .single(index: $0, bucket: singleBucket(for: tokenCounts[$0]))
                    })
                }
                start = end
            }
        }
        plan.append(contentsOf: singles.map { .single(index: $0.index, bucket: $0.bucket) })
        return plan
    }

    private func encoder(forTokenCount n: Int) throws -> (CoreMLTextEncoder, Int) {
        let bucket = buckets.first { $0 >= n } ?? maximumBucket
        return (try singleRowEncoder(bucket: bucket), bucket)
    }

    /// The single-row function for `bucket` (`bucket_<S>`), lazily loaded exactly once.
    func singleRowEncoder(bucket: Int) throws -> CoreMLTextEncoder {
        try cache.value(for: bucket) {
            let fn = isMultiFunction ? "bucket_\(bucket)" : nil
            return try CoreMLTextEncoder(modelURL: compiledURL, computeUnits: units(forBucket: bucket),
                                         functionName: fn, padTokenID: padTokenID)
        }
    }

    /// The batch function for a fixed bucket (bucket_<S>_b<N>), lazily loaded. Placement follows the
    /// same adaptive policy — batch functions only exist where that placement already wins.
    private func batchEncoder(forBucket bucket: Int) throws -> (CoreMLTextEncoder, Int, Int) {
        let encoder = try batchRowEncoder(bucket: bucket)
        return (encoder, encoder.batchSize, bucket)
    }

    /// The batch function for `bucket` (`bucket_<S>_b<N>`), lazily loaded exactly once.
    func batchRowEncoder(bucket: Int) throws -> CoreMLTextEncoder {
        guard let pair = batchPairs.first(where: { $0.bucket == bucket }) else {
            throw CoreMLTextEncoder.EncoderError.badModel("no batch function for bucket \(bucket)")
        }
        let bs = pair.size
        let key = -(bucket * 1000 + bs)   // separate cache namespace from the single-row encoders
        return try cache.value(for: key) {
            let fn = isMultiFunction ? "bucket_\(bucket)_b\(bs)" : nil
            return try CoreMLTextEncoder(modelURL: compiledURL, computeUnits: units(forBucket: bucket),
                                         functionName: fn, padTokenID: padTokenID)
        }
    }

    /// The (size, bucket) batch functions this bundle declares, ascending by bucket — what
    /// startup verification must exercise.
    var batchGeometry: [(size: Int, bucket: Int)] { batchPairs.sorted { $0.bucket < $1.bucket } }

    /// Embed text. `dim` truncates to a Matryoshka dimension (re-normalized); nil = full vector.
    /// `task` selects a task-specific instruction pair (code models) when the bundle carries
    /// `taskPrompts`; nil uses the default `prompts` pair.
    public func embed(_ text: String, prompt: Prompt = .none, task: String? = nil,
                      dim: Int? = nil) throws -> [Float] {
        let ids = tokenIDs(for: text, prompt: prompt, task: task)
        return try embed(tokenIds: ids, dim: dim)
    }

    /// Tokenize with the same role/task conditioning used by inference. Internal production
    /// façades use this to enforce length limits before the low-level truncation policy can apply.
    func tokenIDs(for text: String, prompt: Prompt, task: String? = nil) -> [Int32] {
        tokenizer.encode(prefix(for: prompt, task: task) + text)
    }

    /// Single-row inference from already-tokenized input. Bulk routing uses this for rows that do
    /// not fit a batch function, avoiding a second tokenization while preserving `embed(_:)`.
    private func embed(tokenIds: [Int32], dim: Int?) throws -> [Float] {
        try Task.checkCancellation()
        var ids = tokenIds
        let (enc, bucket) = try encoder(forTokenCount: ids.count)
        if ids.count > bucket { ids = Array(ids.prefix(bucket)) }
        let full = try enc.encode(tokenIds: ids)
        try Task.checkCancellation()
        if let d = dim { return matryoshka(full, dim: d) }
        return full
    }

    /// Embed many texts with order preserved. When the bundle ships batch functions
    /// (`manifest.text.batch`), rows are grouped by batch function (one weight stream per call
    /// instead of per row — measured ~3.2× throughput on short buckets); otherwise this falls back
    /// to the single-row loop. Rows are independent (per-row mask/selector), so results match
    /// `embed(_:)` row-for-row. `dim` applies Matryoshka truncation per row.
    ///
    /// Grouping is by row, not by position: every row joins the widest batch whose bucket fits it
    /// (e.g. b64@bucket 32, b32@64, b16@128, b8@256, b4@512), wherever it sits in the input, so
    /// interleaved short and long rows still amortize batch calls. Rows longer than every batch
    /// bucket use the matching single-row bucket instead; only rows longer than `maxTokens` are
    /// truncated.
    public func embed(texts: [String], prompt: Prompt = .none, task: String? = nil,
                      dim: Int? = nil) throws -> [[Float]] {
        guard !batchPairs.isEmpty, texts.count > 1 else {
            return try texts.map { try embed($0, prompt: prompt, task: task, dim: dim) }
        }
        let allIds = texts.map { tokenIDs(for: $0, prompt: prompt, task: task) }
        return try embed(tokenIDRows: allIds, dim: dim)
    }

    /// Inference for rows already tokenized with ``tokenIDs(for:prompt:task:)``. This keeps
    /// production validation and inference on one tokenization pass without widening public API.
    /// Steps execute in plan order; results are assigned by original index, so the output order is
    /// the input order regardless of grouping. The completeness guard turns a planning defect into
    /// a thrown error rather than a silent misalignment.
    func embed(tokenIDRows allIds: [[Int32]], dim: Int? = nil) throws -> [[Float]] {
        guard !batchPairs.isEmpty, allIds.count > 1 else {
            return try allIds.map { try embed(tokenIds: $0, dim: dim) }
        }
        let plan = Self.embeddingPlan(
            tokenCounts: allIds.map(\.count),
            buckets: buckets,
            batchPairs: batchPairs)
        var out = [[Float]](repeating: [], count: allIds.count)

        for step in plan {
            try Task.checkCancellation()
            switch step {
            case let .single(index, _):
                out[index] = try embed(tokenIds: allIds[index], dim: dim)
            case let .batch(rows, size, bucket):
                let (enc, _, _) = try batchEncoder(forBucket: bucket)
                var chunk = rows.map { Array(allIds[$0]) }
                while chunk.count < size { chunk.append([padTokenID]) }   // filler row; output discarded
                let embs = try enc.encodeBatch(tokenIds: chunk)
                try Task.checkCancellation()
                guard embs.count == chunk.count else {
                    throw CoreMLTextEncoder.EncoderError.badModel(
                        "batch function returned \(embs.count) rows for \(chunk.count) inputs")
                }
                for (offset, row) in rows.enumerated() {
                    let e = embs[offset]
                    out[row] = dim.map { matryoshka(e, dim: $0) } ?? e
                }
            }
        }
        // An empty embedding is impossible for a valid row, so it marks an unassigned index.
        guard !out.contains(where: { $0.isEmpty }) else {
            throw CoreMLTextEncoder.EncoderError.badModel(
                "batch plan did not cover every input row")
        }
        return out
    }

    private func prefix(for prompt: Prompt, task: String? = nil) -> String {
        let strings: PromptStrings
        if let task, let tp = taskPrompts[task] {
            strings = tp
        } else {
            strings = prompts
        }
        switch prompt {
        case .query: return strings.query
        case .document: return strings.document
        case .none: return ""
        }
    }
}
