import Foundation

/// Packing decisions, kept pure for tests.
enum PackPlanner {
    static let tokenBudget = 512
    static let rowBudget = 64
    static let fusedBudget = 64

    /// A row that fits a packed execution (otherwise it is a long input).
    static func isShort(_ tokens: Int) -> Bool { tokens <= tokenBudget }

    /// Greedy in-order selection: take candidates while the pack stays within 512 tokens and
    /// 64 sequences. Candidates that do not fit are skipped (a later, smaller row may fit).
    static func select(_ lengths: [Int]) -> [Int] {
        var chosen = [Int]()
        var total = 0
        for (index, length) in lengths.enumerated() where length <= tokenBudget {
            guard chosen.count < rowBudget else { break }
            if total + length <= tokenBudget {
                chosen.append(index)
                total += length
            }
        }
        return chosen
    }

    static func kind(totalTokens: Int) -> String { totalTokens <= fusedBudget ? "fused64" : "pack512" }
}

/// One input to the language model: a ready sequence, or a multimodal sequence whose media
/// towers must run first.
enum ScheduledInput: Sendable {
    case sequence(LMSequence)
    case media(PendingMediaSequence)

    var tokens: Int {
        switch self {
        case let .sequence(s): s.count
        case let .media(m): m.tokenCount
        }
    }
}

/// Cross-request accelerator scheduler.
///
/// Short sequences (at most 512 templated tokens, text or multimodal) from any number of
/// requests are packed into one Core ML execution: up to 64 sequences sharing a 64- or
/// 512-token chunk with block-diagonal attention, so batching wastes no padding. Long inputs and
/// media towers run as resumable executions that advance one chunk operation per accelerator
/// turn; waiting short work gets the next turn, so a multi-minute 32K document or a long audio
/// clip cannot starve interactive queries.
actor TextScheduler {
    private struct Row {
        let job: UUID
        let index: Int
        var input: ScheduledInput
        let enqueued: UInt64
    }

    private struct Job {
        var outputs: [[Float]?]
        var remaining: Int
        let continuation: CheckedContinuation<[[Float]], any Error>
    }

    private final class MediaState: @unchecked Sendable {
        let pending: PendingMediaSequence
        var features: [BidirLMMediaEncoder.Features] = []
        var run: (any BidirLMMediaEncoder.Run)?
        init(_ pending: PendingMediaSequence) { self.pending = pending }

        var progress: Double {
            let total = Double(pending.spans.count)
            guard total > 0 else { return 1 }
            let current = run.map { Double($0.completedSteps) / Double(max(1, $0.totalSteps)) } ?? 0
            return (Double(features.count) + current) / total
        }
    }

    private enum Active {
        case long(Row, any TextModelRun)
        case media(Row, MediaState)

        var row: Row {
            switch self {
            case let .long(row, _), let .media(row, _): row
            }
        }
    }

    private let backend: BidirLMBackend
    private let lane: AcceleratorLane
    private let metrics: ServerMetrics
    private let windowNanoseconds: UInt64
    private let deepstackLayers: Int

    private var jobs: [UUID: Job] = [:]
    private var shortQueue: [Row] = []
    private var longQueue: [Row] = []
    private var active: Active?
    private var running = false
    private var preferLongNext = false

    init(backend: BidirLMBackend, lane: AcceleratorLane, metrics: ServerMetrics, windowMilliseconds: Double) {
        self.backend = backend
        self.lane = lane
        self.metrics = metrics
        deepstackLayers = backend.bundle.manifest.imageDeepstackSets
        windowNanoseconds = UInt64(max(0, windowMilliseconds) * 1_000_000)
    }

    /// Embed templated token rows; results are returned in input order.
    func submit(_ rows: [[Int32]]) async throws -> [[Float]] {
        try await submit(inputs: rows.map { .sequence(LMSequence(ids: $0)) })
    }

    /// Embed prepared inputs; results are returned in input order.
    func submit(inputs: [ScheduledInput]) async throws -> [[Float]] {
        guard !inputs.isEmpty else { return [] }
        try Task.checkCancellation()
        let id = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[[Float]], any Error>) in
                let now = DispatchTime.now().uptimeNanoseconds
                jobs[id] = Job(outputs: Array(repeating: nil, count: inputs.count), remaining: inputs.count,
                               continuation: continuation)
                for (index, input) in inputs.enumerated() {
                    let row = Row(job: id, index: index, input: input, enqueued: now)
                    if case let .sequence(s) = input, PackPlanner.isShort(s.count) {
                        shortQueue.append(row)
                    } else {
                        longQueue.append(row)
                    }
                }
                if !running {
                    running = true
                    Task { await self.drive() }
                }
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }

    struct Snapshot: Sendable {
        let requests: Int
        let shortRows: Int
        let longRows: Int
        let activeLong: (tokens: Int, progress: Double)?
        let activeMedia: (kind: String, progress: Double)?
    }

    func snapshot() -> Snapshot {
        var long: (Int, Double)?
        var media: (String, Double)?
        switch active {
        case let .long(_, run)?:
            long = (run.tokens, Double(run.completedSteps) / Double(max(1, run.totalSteps)))
        case let .media(_, state)?:
            media = (state.pending.kind, state.progress)
        case nil:
            break
        }
        return Snapshot(requests: jobs.count, shortRows: shortQueue.count,
                        longRows: longQueue.count + (active == nil ? 0 : 1),
                        activeLong: long, activeMedia: media)
    }

    private func cancel(_ id: UUID) {
        guard let job = jobs.removeValue(forKey: id) else { return }
        drop(id)
        job.continuation.resume(throwing: CancellationError())
    }

    private func fail(_ id: UUID, _ error: any Error) {
        guard let job = jobs.removeValue(forKey: id) else { return }
        drop(id)
        job.continuation.resume(throwing: error)
    }

    private func drop(_ id: UUID) {
        shortQueue.removeAll { $0.job == id }
        longQueue.removeAll { $0.job == id }
        if active?.row.job == id { active = nil }
    }

    private func deliver(_ row: Row, _ vector: [Float]) {
        guard var job = jobs[row.job] else { return }
        job.outputs[row.index] = vector
        job.remaining -= 1
        if job.remaining == 0 {
            jobs.removeValue(forKey: row.job)
            job.continuation.resume(returning: job.outputs.map { $0! })
        } else {
            jobs[row.job] = job
        }
    }

    // MARK: - loop

    private func drive() async {
        while !shortQueue.isEmpty || !longQueue.isEmpty || active != nil {
            // Let a partially filled pack collect more rows for the batching window, unless
            // stepped work is waiting (its steps are the natural cadence) or the pack is full.
            if backend.supportsPackedText, active == nil, longQueue.isEmpty, let oldest = shortQueue.first?.enqueued, windowNanoseconds > 0 {
                let totalTokens = shortQueue.reduce(0) { $0 + $1.input.tokens }
                let age = DispatchTime.now().uptimeNanoseconds &- oldest
                if totalTokens < PackPlanner.tokenBudget, shortQueue.count < PackPlanner.rowBudget, age < windowNanoseconds {
                    try? await Task.sleep(nanoseconds: windowNanoseconds - age)
                    continue
                }
            }
            do {
                try await lane.acquire()
            } catch {
                break
            }
            let runLong = (active != nil || !longQueue.isEmpty) && (shortQueue.isEmpty || preferLongNext)
            if runLong {
                await longTurn()
                preferLongNext = false
            } else {
                await packTurn()
                preferLongNext = true
            }
            await lane.release()
        }
        running = false
    }

    private func packTurn() async {
        let indices = backend.supportsPackedText ? PackPlanner.select(shortQueue.map(\.input.tokens)) : Array(shortQueue.indices.prefix(1))
        guard !indices.isEmpty else { return }
        let rows = indices.map { shortQueue[$0] }
        let taken = Set(indices)
        shortQueue = shortQueue.enumerated().filter { !taken.contains($0.offset) }.map(\.element)
        let sequences = rows.compactMap { row -> LMSequence? in
            if case let .sequence(s) = row.input { return s }
            return nil
        }
        let tokens = sequences.reduce(0) { $0 + $1.count }
        do {
            let vectors = try await backend.embed(sequences: sequences)
            let kind = backend.supportsPackedText ? PackPlanner.kind(totalTokens: tokens) : "streamed512"
            await metrics.recordWave(kind: kind, rows: rows.count, tokens: tokens,
                                     requests: Set(rows.map(\.job)).count)
            for (row, vector) in zip(rows, vectors) { deliver(row, vector) }
        } catch {
            for id in Set(rows.map(\.job)) { fail(id, error) }
        }
    }

    private func longTurn() async {
        do {
            if active == nil {
                guard !longQueue.isEmpty else { return }
                let row = longQueue.removeFirst()
                guard jobs[row.job] != nil else { return }
                switch row.input {
                case let .sequence(s): active = .long(row, try await backend.startLong(s))
                case let .media(p): active = .media(row, MediaState(p))
                }
            }
            switch active {
            case let .long(row, run)?:
                if let vector = try await backend.step(run) {
                    active = nil
                    await metrics.recordLong(tokens: run.tokens, steps: run.totalSteps)
                    deliver(row, vector)
                }
            case let .media(row, state)?:
                try await mediaStep(row, state)
            case nil:
                break
            }
        } catch {
            if let current = active {
                active = nil
                fail(current.row.job, error)
            }
        }
    }

    /// One tower step; when every span has features, the finished sequence joins the short
    /// queue (front) or becomes the active long run.
    private func mediaStep(_ row: Row, _ state: MediaState) async throws {
        if state.features.count < state.pending.spans.count {
            if state.run == nil {
                state.run = try await backend.startTower(state.pending.spans[state.features.count].input)
            }
            if let features = try await backend.step(state.run!) {
                state.features.append(features)
                state.run = nil
            }
            guard state.features.count == state.pending.spans.count else { return }
        }
        let sequence = try state.pending.sequence(features: state.features, deepstackLayers: deepstackLayers)
        await metrics.recordMedia(kind: state.pending.kind)
        var ready = row
        ready.input = .sequence(sequence)
        if PackPlanner.isShort(sequence.count) {
            active = nil
            shortQueue.insert(ready, at: 0)
        } else {
            active = .long(ready, try await backend.startLong(sequence))
        }
    }
}
