import CoreML
import Foundation

protocol TextModelRun: AnyObject, Sendable {
    var tokens: Int { get }
    var completedSteps: Int { get }
    var totalSteps: Int { get }
    func step() throws -> Bool
    func result() throws -> [Float]
}

protocol TextModelEncoder: AnyObject, Sendable {
    func encode(sequences: [LMSequence]) throws -> [[Float]]
    func startLong(_ sequence: LMSequence) throws -> any TextModelRun
}

/// Full-depth, layer-major execution of the normalized-summary artifact. Each run owns its
/// buffers, so short requests can interleave without overwriting a suspended long request.
/// Host work is table lookup, RoPE, K/V layout copies, and FP64 pooling; neural work stays in
/// Core ML. CPU_AND_NE is a policy, not evidence of actual ANE execution.
final class StreamedTextEncoder: TextModelEncoder, @unchecked Sendable {
    static let programSet = "text.streamed"
    let manifest: StreamedTextManifest
    private let models: [String: MLModel]
    private let table: Data
    private let cosine: Data
    private let sine: Data
    private let inverseFrequency: [Float]
    private(set) var predictions = 0
    private(set) var declinedOutputBackings = 0

    init(residency: ProgramResidency, manifest: StreamedTextManifest) throws {
        try manifest.validate()
        self.manifest = manifest
        let root = residency.root
        table = try Data(contentsOf: root.appendingPathComponent(manifest.embeddings.file), options: .alwaysMapped)
        cosine = try Data(contentsOf: root.appendingPathComponent(manifest.nativeTables.cos), options: .alwaysMapped)
        sine = try Data(contentsOf: root.appendingPathComponent(manifest.nativeTables.sin), options: .alwaysMapped)
        let frequencies = try Data(contentsOf: root.appendingPathComponent(manifest.nativeTables.inverseFrequency))
        inverseFrequency = frequencies.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard table.count == 151936 * 2048 * 2, cosine.count == 128 * 32768 * 2,
              sine.count == cosine.count, inverseFrequency.count == 64 else {
            throw BidirLMBundle.Failure.invalid("streamed host table size mismatch")
        }
        let functions = manifest.stages.mapValues { ProgramResidency.Function(path: $0.compiled, name: $0.function) }
        let loaded = try residency.acquire(Self.programSet, functions.sorted { $0.key < $1.key }.map(\.value), pinned: true)
        models = functions.mapValues { loaded[$0]! }
    }

    func encode(sequences: [LMSequence]) throws -> [[Float]] {
        // v2 does not advertise legacy block-diagonal short-input packing.
        try sequences.map { sequence in
            let run = try makeRun(sequence)
            while try !run.step() { try Task.checkCancellation() }
            return try run.result()
        }
    }

    func startLong(_ sequence: LMSequence) throws -> any TextModelRun { try makeRun(sequence) }

    func makeRun(_ sequence: LMSequence, valid: [Bool]? = nil,
                 observeLayer: ((Int, [HalfSurface]) throws -> Void)? = nil) throws -> Run {
        try Run(encoder: self, sequence: sequence, valid: valid, observeLayer: observeLayer)
    }

    private func predict(_ name: String, _ feeds: [String: MLMultiArray], _ outputs: [String: HalfSurface]) throws {
        guard let model = models[name] else { throw BidirLMBundle.Failure.invalid("missing streamed model \(name)") }
        declinedOutputBackings += try HalfSurface.predict(model, feeds: feeds, outputs: outputs)
        predictions += 1
    }

    final class Run: TextModelRun, @unchecked Sendable {
        private let encoder: StreamedTextEncoder
        let tokens: Int
        let totalSteps: Int
        private(set) var completedSteps = 0
        private let count: Int
        private let chunk: Int
        private let blocks: Int
        private let summaryName: String
        private let banks: [StreamedKVBank]
        private let valid: [Bool]
        private var hidden: [HalfSurface]
        private let queries: [HalfSurface]
        private let rope: [(HalfSurface, HalfSurface)]
        private let deepstack: [[HalfSurface]]
        private let summaries: [[String: HalfSurface]]
        private let mergeFeeds: [String: MLMultiArray]
        private let attention: HalfSurface
        private var hiddenScratch: HalfSurface
        private let keyScratch: HalfSurface
        private let valueScratch: HalfSurface
        private let normalized: HalfSurface
        private let observeLayer: ((Int, [HalfSurface]) throws -> Void)?
        private var pooled = [Double](repeating: 0, count: 2048)
        private var embedding: [Float]?
        private var headed = 0
        private var layer = 0
        private var queryIndex = 0
        private var blockIndex = 0
        private var merged = false

        fileprivate init(encoder e: StreamedTextEncoder, sequence: LMSequence, valid: [Bool]?,
                         observeLayer: ((Int, [HalfSurface]) throws -> Void)?) throws {
            let m = e.manifest, n = sequence.count, c = m.geometry.chunk
            guard n > 0, n <= m.geometry.capacity, sequence.ids.allSatisfy({ $0 >= 0 && $0 < 151936 }) else {
                throw BidirLMBundle.Failure.invalid("streamed sequence length or token ID is out of range")
            }
            let mask = valid ?? [Bool](repeating: true, count: n)
            guard mask.count == n, mask.contains(true) else {
                throw BidirLMBundle.Failure.invalid("streamed mask must contain at least one valid token")
            }
            for replacement in sequence.replacements {
                guard replacement.rows.count.isMultiple(of: 2048), replacement.start >= 0,
                      replacement.start + replacement.rows.count / 2048 <= n,
                      replacement.rows.allSatisfy(\.isFinite) else {
                    throw BidirLMBundle.Failure.invalid("invalid streamed media replacement")
                }
            }
            guard sequence.positions == nil || sequence.positions?.count == n,
                  sequence.deepstack.count <= m.deepstackLayers,
                  sequence.deepstackPositions.allSatisfy({ (0..<n).contains($0) }),
                  sequence.deepstack.allSatisfy({ $0.count == sequence.deepstackPositions.count * 2048 && $0.allSatisfy(\.isFinite) }) else {
                throw BidirLMBundle.Failure.invalid("invalid streamed positions or DeepStack rows")
            }
            encoder = e
            self.observeLayer = observeLayer
            let chunkCount = (n + c - 1) / c
            tokens = n; chunk = c; count = chunkCount
            self.valid = mask
            let keys = m.keyBuckets.first { $0 >= chunkCount * c }!
            let block = min(keys, m.geometry.keyBlock)
            blocks = keys / block
            summaryName = "summary_b\(block)"
            totalSteps = count * (1 + m.layers * (blocks + 1 + (blocks > 1 ? 1 : 0)))
            var paddedMask = [Bool](repeating: false, count: keys)
            paddedMask.replaceSubrange(0..<n, with: mask)
            banks = try (0..<2).map { _ in try StreamedKVBank(keys: keys, block: block, valid: paddedMask) }
            let hShape = [1, 2048, 1, c], qShape = [1, 8, 128, 2 * c]
            hidden = try (0..<count).map { _ in try HalfSurface(hShape) }
            queries = try (0..<count).map { _ in try HalfSurface(qShape) }
            hiddenScratch = try HalfSurface(hShape)
            keyScratch = try HalfSurface([1, 8, 128, c]); valueScratch = try HalfSurface([1, 8, 128, c])
            normalized = try HalfSurface(hShape)
            attention = try HalfSurface(qShape)
            summaries = try (0..<blocks).map { _ in
                ["maximum": try HalfSurface([8, 1, 1, 2 * c]), "mass": try HalfSurface([8, 1, 1, 2 * c]),
                 "context": try HalfSurface(qShape)]
            }
            var feeds: [String: MLMultiArray] = [:]
            for (b, summary) in summaries.enumerated() {
                for (name, surface) in summary { feeds["\(name)_\(b)"] = surface.array }
            }
            mergeFeeds = feeds
            let zero = try HalfSurface(hShape, zeroed: true)
            deepstack = try (0..<m.deepstackLayers).map { index in
                try (0..<chunkCount).map { _ in
                    index < sequence.deepstack.count ? try HalfSurface(hShape, zeroed: true) : zero
                }
            }
            rope = try (0..<count).map { _ in (try HalfSurface([1, 1, 128, c]), try HalfSurface([1, 1, 128, c])) }
            try e.table.withUnsafeBytes { bytes in
                let table = bytes.bindMemory(to: Float16.self)
                for i in 0..<count {
                    try hidden[i].withMutable { dst, s in
                        for t in 0..<c {
                            let position = i * c + t, id = position < n ? sequence.ids[position] : m.padTokenID
                            for d in 0..<2048 { dst[d * s[1] + t] = table[Int(id) * 2048 + d] }
                        }
                    }
                }
            }
            for replacement in sequence.replacements {
                for row in 0..<(replacement.rows.count / 2048) {
                    let position = replacement.start + row
                    try hidden[position / c].withMutable { dst, s in
                        for d in 0..<2048 { dst[d * s[1] + position % c] = replacement.rows[row * 2048 + d] }
                    }
                }
            }
            for (index, rows) in sequence.deepstack.enumerated() {
                for (row, position) in sequence.deepstackPositions.enumerated() {
                    try deepstack[index][position / c].withMutable { dst, s in
                        for d in 0..<2048 { dst[d * s[1] + position % c] = rows[row * 2048 + d] }
                    }
                }
            }
            var axes = [Int](repeating: 0, count: 64)
            for (axis, offset) in [(1, 1), (2, 2)] {
                for i in stride(from: offset, to: min(64, m.mropeSection[axis] * 3), by: 3) { axes[i] = axis }
            }
            try e.cosine.withUnsafeBytes { cb in
                try e.sine.withUnsafeBytes { sb in
                    let cosine = cb.bindMemory(to: Float16.self), sine = sb.bindMemory(to: Float16.self)
                    for i in 0..<count {
                        try rope[i].0.withMutable { cp, cs in
                            try rope[i].1.withMutable { sp, ss in
                                for d in 0..<128 {
                                    for t in 0..<c {
                                        let position = i * c + t
                                        if sequence.positions != nil, position < n {
                                            let p = sequence.position(position), coords = [p.0, p.1, p.2]
                                            let angle = Float(coords[axes[d % 64]]) * e.inverseFrequency[d % 64]
                                            cp[d * cs[2] + t] = Float16(Foundation.cos(angle))
                                            sp[d * ss[2] + t] = Float16(Foundation.sin(angle))
                                        } else {
                                            cp[d * cs[2] + t] = cosine[d * 32768 + position]
                                            sp[d * ss[2] + t] = sine[d * 32768 + position]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Exactly one Core ML prediction per step; a completed bank is only swapped at the
        /// whole-layer barrier. The single-block case directly consumes normalized context.
        func step() throws -> Bool {
            if embedding != nil { return true }
            try Task.checkCancellation()
            defer { completedSteps += 1 }
            let e = encoder
            if headed < count {
                try e.predict("head", ["hidden": hidden[headed].array, "cos": rope[headed].0.array, "sin": rope[headed].1.array],
                              ["query": queries[headed], "key": keyScratch, "value": valueScratch])
                try banks[0].store(start: headed * chunk, key: keyScratch, value: valueScratch)
                headed += 1
                return false
            }
            if blockIndex < blocks {
                try e.predict(summaryName, banks[layer % 2].feeds(block: blockIndex, query: queries[queryIndex]), summaries[blockIndex])
                blockIndex += 1
                return false
            }
            if blocks > 1, !merged {
                try e.predict("merge_n\(blocks)", mergeFeeds, ["attention": attention])
                merged = true
                return false
            }
            let context = blocks == 1 ? summaries[0]["context"]! : attention
            var feeds = ["attention": context.array, "hidden": hidden[queryIndex].array]
            var outputs = ["hidden_out": hiddenScratch]
            if layer < 27 {
                feeds["cos"] = rope[queryIndex].0.array; feeds["sin"] = rope[queryIndex].1.array
                outputs.merge(["query": queries[queryIndex], "key": keyScratch, "value": valueScratch]) { _, new in new }
            } else { outputs["normalized"] = normalized }
            if layer < e.manifest.deepstackLayers { feeds["deepstack"] = deepstack[layer][queryIndex].array }
            try e.predict("boundary_\(layer)", feeds, outputs)
            let old = hidden[queryIndex]
            hidden[queryIndex] = hiddenScratch
            hiddenScratch = old
            if layer < 27 {
                try banks[(layer + 1) % 2].store(start: queryIndex * chunk, key: keyScratch, value: valueScratch)
            } else {
                try normalized.withBuffer { src, s in
                    for t in 0..<min(chunk, tokens - queryIndex * chunk) where valid[queryIndex * chunk + t] {
                        for d in 0..<2048 { pooled[d] += Double(src[d * s[1] + t]) }
                    }
                }
            }
            queryIndex += 1; blockIndex = 0; merged = false
            if queryIndex == count {
                try observeLayer?(layer, hidden)
                queryIndex = 0; layer += 1
                if layer == 28 {
                    let norm = pooled.reduce(0) { $0 + $1 * $1 }.squareRoot()
                    guard norm.isFinite, norm > 0 else { throw HalfArrays.Failure.nonFinite("streamed pooled embedding") }
                    embedding = pooled.map { Float($0 / norm) }
                }
            }
            return embedding != nil
        }

        func result() throws -> [Float] {
            guard let embedding else { throw BidirLMBundle.Failure.invalid("streamed run has not completed all layers") }
            return embedding
        }
    }
}
