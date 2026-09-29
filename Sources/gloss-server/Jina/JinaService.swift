import Darwin
import Foundation
import Tokenizers

// HTTP service for jina-embeddings-v5-omni-small bundles (manifest `formatVersion` 2). It shares
// the server shell (HTTPServer, admission, accelerator lane, runtime state, metrics) with the
// BidirLM service; `ModelFamily` picks one service per process from the bundle manifest.
//
// Adapted from the pre-BidirLM daemon (branch `archive/jina-omnismall-wip`, 73ad24b), with:
// retrieval roles on the request (`task`/`role`/`input_type`, default document), the model's full
// 32768-token text context, token-ID vocabulary checks, per-modality metrics, and a /health that
// reports the Jina contract.

struct JinaServiceConfig: Sendable {
    var port: UInt16
    var defaultDimensions: OmniSmall.Dimensions
    var modelName: String?
    var maxBatch: Int
    var maxBodyBytes: Int
    var maxTotalBodyBytes: Int
    var maxQueueRequests: Int
    var maxQueueItems: Int
    var maxRequestTokens: Int
    var batchWindowMilliseconds: Double
    var keepWarmSeconds: Int
    var maxConnections: Int
    var ioTimeoutSeconds: Int
    var shutdownGraceSeconds: Int
}

enum JinaLimits {
    static let maximumItems = 2_048
    /// The native text contract: conditioning prompt plus text, never truncated.
    static let maximumTokensPerInput = OmniSmallInputLimits.maximumTextTokens
    /// Retrieval roles the bundle's merged LoRA supports; other Jina tasks are not converted.
    static let roles = ["query", "document"]
    static let defaultRole = RetrievalRole.document
    /// Adaptive placement inside the Jina runtime (not operator-selectable).
    static let computeDescription = "auto"
}

enum JinaTokenizerError: Error, CustomStringConvertible {
    case tokenizerMissing
    case tokenizerLoadFailed(String)
    var description: String {
        switch self {
        case .tokenizerMissing: "bundle does not declare a text tokenizer and retrieval prompts"
        case let .tokenizerLoadFailed(reason): "failed to load bundle tokenizer: \(reason)"
        }
    }
}

/// A small CPU tokenizer pool: exact conditioned token IDs are produced once for preflight,
/// batching, usage accounting, and model execution.
final class JinaTokenizerPool: @unchecked Sendable {
    private let tokenizers: [any Tokenizer]
    private let queryPrompt: String
    private let documentPrompt: String
    /// Text-model vocabulary size (from the bundled `config.json`), for token-ID inputs.
    let vocabulary: Int?
    private let condition = NSCondition()
    private var available: [Int]

    private final class Box: @unchecked Sendable {
        var result: Result<any Tokenizer, any Error>?
    }

    init(bundle: GlossModelBundle) throws {
        guard let text = bundle.manifest.text, let prompts = bundle.manifest.prompts else {
            throw JinaTokenizerError.tokenizerMissing
        }
        queryPrompt = prompts.query
        documentPrompt = prompts.document
        let folder = bundle.resolve(text.tokenizer)
        vocabulary = Self.vocabularySize(folder.appendingPathComponent("config.json"))
        let count = max(1, min(4, ProcessInfo.processInfo.activeProcessorCount))
        var loaded: [any Tokenizer] = []
        for _ in 0..<count {
            let box = Box()
            let semaphore = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                do { box.result = .success(try await AutoTokenizer.from(modelFolder: folder)) }
                catch { box.result = .failure(error) }
                semaphore.signal()
            }
            semaphore.wait()
            switch box.result {
            case let .success(tokenizer)?: loaded.append(tokenizer)
            case let .failure(error)?: throw JinaTokenizerError.tokenizerLoadFailed(String(describing: error))
            case nil: throw JinaTokenizerError.tokenizerLoadFailed("loader completed without a result")
            }
        }
        tokenizers = loaded
        available = Array(loaded.indices)
    }

    func tokenIDs(_ text: String, role: RetrievalRole) -> [Int32] {
        let index = checkout()
        defer { checkin(index) }
        let prefix = role == .query ? queryPrompt : documentPrompt
        return tokenizers[index].encode(text: prefix + text, addSpecialTokens: true).map(Int32.init)
    }

    private static func vocabularySize(_ url: URL) -> Int? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let text = object["text_config"] as? [String: Any], let size = text["vocab_size"] as? Int { return size }
        return object["vocab_size"] as? Int
    }

    private func checkout() -> Int {
        condition.lock()
        while available.isEmpty { condition.wait() }
        let index = available.removeLast()
        condition.unlock()
        return index
    }

    private func checkin(_ index: Int) {
        condition.lock(); available.append(index); condition.signal(); condition.unlock()
    }
}

/// The validated, resident model. One instance serves every Matryoshka output size.
actor JinaModelStore {
    private let model: OmniSmall

    init(model: OmniSmall) { self.model = model }

    func get() -> OmniSmall { model }
    nonisolated func space(for dimensions: OmniSmall.Dimensions) -> String { model.space(for: dimensions) }
}

func jinaProjectMatryoshka(_ values: [Float], to dimensions: OmniSmall.Dimensions) throws -> [Float] {
    let count = dimensions.rawValue
    guard values.count >= count else {
        throw OmniSmallError.invalidEmbedding("cannot project a \(values.count)-wide embedding to \(count) dimensions")
    }
    if values.count == count { return values }
    var projected = Array(values.prefix(count))
    var sum = 0.0
    for value in projected { sum += Double(value) * Double(value) }
    let norm = sum.squareRoot()
    guard norm.isFinite, norm > 0 else {
        throw OmniSmallError.invalidEmbedding("Matryoshka projection produced a zero or non-finite norm")
    }
    let inverse = Float(1.0 / norm)
    for index in projected.indices { projected[index] *= inverse }
    return projected
}

// MARK: - text batching

struct JinaBatchedResult: Sendable {
    var values: [[Float]]
    var space: String
}

enum JinaBatchPlanner {
    static let maximumWaveRows = 64

    /// Native batch capacity of the text function family for a row of `tokenCount` tokens.
    static func nativeCapacity(for tokenCount: Int) -> Int {
        switch tokenCount {
        case ...32: 64
        case ...64: 32
        case ...128: 16
        case ...256: 8
        case ...512: 4
        default: 1
        }
    }

    static func nativeBucket(for capacity: Int) -> Int {
        switch capacity {
        case 64: 32
        case 32: 64
        case 16: 128
        case 8: 256
        case 4: 512
        default: 0
        }
    }

    static func rotatedOrder<T: Equatable>(_ ids: [T], after previous: T?) -> [T] {
        guard ids.count > 1, let previous, let index = ids.firstIndex(of: previous) else { return ids }
        let start = ids.index(after: index)
        guard start != ids.endIndex else { return ids }
        return Array(ids[start...]) + Array(ids[..<start])
    }
}

/// Cross-request text scheduler. Each hardware turn targets one exact native text batch bucket.
/// Query and document rows may share a turn: conditioning is already encoded in the token IDs.
actor JinaTextBatcher {
    private struct Job {
        let id: UUID
        let dimensions: OmniSmall.Dimensions
        let tokenIDRows: [[Int32]]
        let tokenCounts: [Int]
        let enqueuedAtNanos: UInt64
        var pendingIndices: [Int]
        var outputs: [[Float]?]
        let continuation: CheckedContinuation<JinaBatchedResult, any Error>
    }

    private struct Selection: Sendable {
        let jobID: UUID
        let rowIndex: Int
        let tokenIDs: [Int32]
        let dimensions: OmniSmall.Dimensions
    }

    private struct BucketSummary {
        var count = 0
        var oldestEnqueueNanos = UInt64.max
        var earliestJobRank = Int.max
        var earliestRowIndex = Int.max
    }

    private let store: JinaModelStore
    private let lane: AcceleratorLane
    private let metrics: ServerMetrics
    private let windowNanoseconds: UInt64

    private var jobs: [UUID: Job] = [:]
    private var order: [UUID] = []
    private var flushTask: Task<Void, Never>?
    private var executing = false
    private var dispatchSequence: UInt64 = 0
    private var lastBucketDispatch: [Int: UInt64] = [:]
    private var lastJobDispatch: [Int: UUID] = [:]

    init(store: JinaModelStore, lane: AcceleratorLane, metrics: ServerMetrics, windowMilliseconds: Double) {
        self.store = store
        self.lane = lane
        self.metrics = metrics
        self.windowNanoseconds = UInt64(max(0, windowMilliseconds) * 1_000_000)
    }

    func submit(tokenIDRows: [[Int32]], tokenCounts: [Int], dimensions: OmniSmall.Dimensions) async throws -> JinaBatchedResult {
        guard tokenIDRows.count == tokenCounts.count, !tokenIDRows.isEmpty else {
            throw OmniSmallError.invalidInput("dynamic batch submission is empty or malformed")
        }
        try Task.checkCancellation()
        let id = UUID()
        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<JinaBatchedResult, any Error>) in
                guard !Task.isCancelled else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                jobs[id] = Job(
                    id: id, dimensions: dimensions, tokenIDRows: tokenIDRows, tokenCounts: tokenCounts,
                    enqueuedAtNanos: DispatchTime.now().uptimeNanoseconds,
                    pendingIndices: Array(tokenIDRows.indices),
                    outputs: [[Float]?](repeating: nil, count: tokenIDRows.count),
                    continuation: continuation)
                order.append(id)
                scheduleIfNeeded()
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
        try Task.checkCancellation()
        return result
    }

    func snapshot() -> (requests: Int, rows: Int, executing: Bool) {
        let rows = jobs.values.reduce(into: 0) { $0 += $1.pendingIndices.count }
        return (jobs.count, rows, executing)
    }

    private func cancel(_ id: UUID) {
        guard let job = jobs.removeValue(forKey: id) else { return }
        order.removeAll { $0 == id }
        job.continuation.resume(throwing: CancellationError())
        scheduleIfNeeded()
    }

    private func scheduleIfNeeded() {
        guard !executing, !order.isEmpty else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        if readyCapacity(now: now) != nil {
            flushTask?.cancel()
            flushTask = nil
            Task { await self.flushOneWave() }
            return
        }
        guard flushTask == nil else { return }
        let delay = remainingCollectionDelay(now: now)
        flushTask = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(nanoseconds: delay) }
            guard !Task.isCancelled else { return }
            await self?.flushTimerFired()
        }
    }

    private func flushTimerFired() {
        flushTask = nil
        guard !executing, !order.isEmpty else { return }
        Task { await self.flushOneWave() }
    }

    private func bucketSummaries() -> [Int: BucketSummary] {
        var summaries: [Int: BucketSummary] = [:]
        for (rank, id) in order.enumerated() {
            guard let job = jobs[id] else { continue }
            for rowIndex in job.pendingIndices {
                let capacity = JinaBatchPlanner.nativeCapacity(for: job.tokenCounts[rowIndex])
                var summary = summaries[capacity] ?? BucketSummary()
                summary.count += 1
                summary.oldestEnqueueNanos = min(summary.oldestEnqueueNanos, job.enqueuedAtNanos)
                if rank < summary.earliestJobRank
                    || (rank == summary.earliestJobRank && rowIndex < summary.earliestRowIndex) {
                    summary.earliestJobRank = rank
                    summary.earliestRowIndex = rowIndex
                }
                summaries[capacity] = summary
            }
        }
        return summaries
    }

    private func readyCapacity(now: UInt64) -> Int? {
        let ready = bucketSummaries().compactMap {
            capacity, summary -> (capacity: Int, summary: BucketSummary, full: Bool, last: UInt64)? in
            let full = summary.count >= capacity
            let age = now >= summary.oldestEnqueueNanos ? now - summary.oldestEnqueueNanos : 0
            let matured = windowNanoseconds == 0 || age >= windowNanoseconds
            guard capacity == 1 || full || matured else { return nil }
            return (capacity, summary, full, lastBucketDispatch[capacity] ?? 0)
        }
        return ready.sorted { lhs, rhs in
            if lhs.last != rhs.last { return lhs.last < rhs.last }
            if lhs.summary.oldestEnqueueNanos != rhs.summary.oldestEnqueueNanos {
                return lhs.summary.oldestEnqueueNanos < rhs.summary.oldestEnqueueNanos
            }
            if lhs.full != rhs.full { return lhs.full && !rhs.full }
            if lhs.summary.earliestJobRank != rhs.summary.earliestJobRank {
                return lhs.summary.earliestJobRank < rhs.summary.earliestJobRank
            }
            if lhs.summary.earliestRowIndex != rhs.summary.earliestRowIndex {
                return lhs.summary.earliestRowIndex < rhs.summary.earliestRowIndex
            }
            return lhs.capacity > rhs.capacity
        }.first?.capacity
    }

    private func remainingCollectionDelay(now: UInt64) -> UInt64 {
        guard windowNanoseconds > 0 else { return 0 }
        var remaining = windowNanoseconds
        var found = false
        for (_, summary) in bucketSummaries() {
            found = true
            let age = now >= summary.oldestEnqueueNanos ? now - summary.oldestEnqueueNanos : 0
            if age >= windowNanoseconds { return 0 }
            remaining = min(remaining, windowNanoseconds - age)
        }
        return found ? remaining : 0
    }

    private func takeWave(capacity: Int, limit: Int) -> [Selection] {
        var selections: [Selection] = []
        selections.reserveCapacity(limit)
        let ids = JinaBatchPlanner.rotatedOrder(order, after: lastJobDispatch[capacity])
        while selections.count < limit {
            var madeProgress = false
            for id in ids {
                guard selections.count < limit,
                      var job = jobs[id],
                      let pendingPosition = job.pendingIndices.firstIndex(where: {
                          JinaBatchPlanner.nativeCapacity(for: job.tokenCounts[$0]) == capacity
                      }) else { continue }
                let rowIndex = job.pendingIndices.remove(at: pendingPosition)
                selections.append(.init(jobID: id, rowIndex: rowIndex, tokenIDs: job.tokenIDRows[rowIndex],
                                        dimensions: job.dimensions))
                jobs[id] = job
                madeProgress = true
            }
            if !madeProgress { break }
        }
        return selections
    }

    /// What one wave produced, row by row, in `active` order (rows whose jobs vanished while
    /// the wave ran are omitted).
    private struct WaveExecution {
        var outcomes: [(selection: Selection, result: Result<[Float], any Error>)]
        /// The batch failed and every row was re-run alone.
        var isolated: Bool
    }

    private func flushOneWave() async {
        guard !executing, !order.isEmpty else { return }
        flushTask?.cancel()
        flushTask = nil
        executing = true
        var selected: [Selection] = []
        var holdingLane = false

        do {
            // Waiting for media or warmup is free batching time. Freeze the bucket only once the
            // accelerator permit is ours, so arrivals during the wait can join.
            try await lane.acquire()
            holdingLane = true
            let now = DispatchTime.now().uptimeNanoseconds
            guard let capacity = readyCapacity(now: now) else {
                await lane.release()
                executing = false
                scheduleIfNeeded()
                return
            }
            // Sparse native batches are slower than the same rows on single-row functions:
            // dispatch one row per turn until enough same-bucket rows are waiting.
            let pendingRows = bucketSummaries()[capacity]?.count ?? 0
            let threshold = NativeTextBatchPolicy.minimumRows(
                for: capacity, bucket: JinaBatchPlanner.nativeBucket(for: capacity))
            let limit = pendingRows >= threshold ? min(capacity, JinaBatchPlanner.maximumWaveRows) : 1
            selected = takeWave(capacity: capacity, limit: limit)
            guard !selected.isEmpty else {
                await lane.release()
                executing = false
                cleanFinishedOrMissingJobs()
                scheduleIfNeeded()
                return
            }
            dispatchSequence &+= 1
            lastBucketDispatch[capacity] = dispatchSequence
            lastJobDispatch[capacity] = selected.last?.jobID

            let active = selected.filter { jobs[$0.jobID] != nil }
            guard !active.isEmpty else {
                await lane.release()
                executing = false
                cleanFinishedOrMissingJobs()
                scheduleIfNeeded()
                return
            }
            let executionDimensions = OmniSmall.Dimensions(
                rawValue: active.map { $0.dimensions.rawValue }.max() ?? 1024) ?? .d1024
            let model = await store.get()
            let execution = try await executeWave(active, dimensions: executionDimensions, model: model)
            await lane.release()
            holdingLane = false

            // Bookkeeping happens off the accelerator. Every lookup re-checks `jobs`: a client may
            // have cancelled (or an earlier row failed its job) while the lane was busy.
            let failedRows = execution.outcomes.filter { if case .failure = $0.result { true } else { false } }.count
            if execution.isolated {
                await metrics.recordIsolation(failedRows: failedRows)
            } else if failedRows == 0 {
                // A dense wave counts once under its native capacity; otherwise the backend ran
                // each row on the single-row functions.
                let tokens = active.reduce(0) { $0 + $1.tokenIDs.count }
                let kind = active.count >= threshold && capacity > 1 ? "native_b\(capacity)" : "single"
                await metrics.recordWave(kind: kind, rows: active.count, tokens: tokens,
                                         requests: Set(active.map(\.jobID)).count)
            }
            for outcome in execution.outcomes {
                switch outcome.result {
                case let .success(values):
                    if execution.isolated {
                        await metrics.recordWave(kind: "single", rows: 1, tokens: outcome.selection.tokenIDs.count,
                                                 requests: 1)
                    }
                    storeOutput(values, for: outcome.selection)
                case let .failure(error):
                    failJobs([outcome.selection.jobID],
                             error: Self.rowError(error, rowIndex: outcome.selection.rowIndex))
                }
            }
            completeReadyJobs()
        } catch {
            // Only cancellation of the scheduling task itself lands here (row and wave failures
            // are outcomes, not throws).
            if holdingLane { await lane.release() }
            failJobs(Set(selected.map(\.jobID)), error: error)
        }
        executing = false
        cleanFinishedOrMissingJobs()
        scheduleIfNeeded()
    }

    /// Run one wave with the lane held. A failed multi-row batch does not fail its rows: they come
    /// from unrelated clients, so each is re-run alone on the single-row functions and only the
    /// jobs whose own rows fail are failed. Throws only `CancellationError`.
    private func executeWave(
        _ active: [Selection], dimensions: OmniSmall.Dimensions, model: OmniSmall
    ) async throws -> WaveExecution {
        do {
            let raw = try await model.embedConditionedTokenRows(active.map(\.tokenIDs), dimensions: dimensions)
            guard raw.count == active.count else {
                throw OmniSmallError.inferenceFailed("batched execution returned \(raw.count) rows for \(active.count) inputs")
            }
            return WaveExecution(outcomes: zip(active, raw).map { ($0, .success($1)) }, isolated: false)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            guard active.count > 1 else {
                return WaveExecution(outcomes: [(active[0], .failure(error))], isolated: false)
            }
            timestamped("text wave of \(active.count) rows failed (\(error)); re-running each row on the single-row functions", error: true)
            var outcomes: [(selection: Selection, result: Result<[Float], any Error>)] = []
            var failedJobs = Set<UUID>()
            for selection in active {
                // Skip rows nobody is waiting for: a cancelled job, or one an earlier row failed.
                guard jobs[selection.jobID] != nil, !failedJobs.contains(selection.jobID) else { continue }
                do {
                    let rows = try await model.embedConditionedTokenRows([selection.tokenIDs], dimensions: selection.dimensions)
                    guard rows.count == 1 else {
                        throw OmniSmallError.inferenceFailed("single-row execution returned \(rows.count) rows for 1 input")
                    }
                    outcomes.append((selection, .success(rows[0])))
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    failedJobs.insert(selection.jobID)
                    outcomes.append((selection, .failure(error)))
                }
            }
            return WaveExecution(outcomes: outcomes, isolated: true)
        }
    }

    /// Record a finished row on its job, truncating to the job's own size. A row that cannot be
    /// projected fails only its job.
    private func storeOutput(_ values: [Float], for selection: Selection) {
        guard var job = jobs[selection.jobID] else { return }
        do {
            job.outputs[selection.rowIndex] = try jinaProjectMatryoshka(values, to: selection.dimensions)
            jobs[selection.jobID] = job
        } catch {
            failJobs([selection.jobID], error: error)
        }
    }

    /// A row that ran alone is index 0 to the model; the job knows it by its position in its own
    /// request, which is what callers map back to an input index.
    private static func rowError(_ error: any Error, rowIndex: Int) -> any Error {
        if case let OmniSmallError.invalidBatchInput(_, reason) = error {
            return OmniSmallError.invalidBatchInput(index: rowIndex, reason: reason)
        }
        return error
    }

    private func completeReadyJobs() {
        for id in order {
            guard let job = jobs[id], job.pendingIndices.isEmpty, job.outputs.allSatisfy({ $0 != nil }) else { continue }
            let values = job.outputs.compactMap { $0 }
            jobs.removeValue(forKey: id)
            order.removeAll { $0 == id }
            job.continuation.resume(returning: .init(values: values, space: store.space(for: job.dimensions)))
        }
    }

    private func failJobs(_ ids: Set<UUID>, error: any Error) {
        for id in ids {
            guard let job = jobs.removeValue(forKey: id) else { continue }
            order.removeAll { $0 == id }
            job.continuation.resume(throwing: error)
        }
    }

    private func cleanFinishedOrMissingJobs() { order.removeAll { jobs[$0] == nil } }
}

/// Release the shared accelerator between media items; only consecutive text rows benefit from a
/// native text batch. A single mixed request must not monopolize hardware for its whole body.
enum JinaMediaRequestPlanner {
    static func nextChunkEnd(_ inputs: [OmniSmall.Input], from start: Int) -> Int {
        guard start < inputs.count else { return start }
        guard case .text = inputs[start] else { return start + 1 }
        var end = start + 1
        while end < inputs.count, end - start < 64 {
            guard case .text = inputs[end] else { break }
            end += 1
        }
        return end
    }
}

// MARK: - startup verification and hardware identity

/// How much the process proves before `/ready` turns green.
enum JinaStartupVerification: String, Sendable, CaseIterable {
    /// Load and run every Core ML function once, cross-checked for self-consistency.
    case full
    /// One text embedding: bundle validation plus a single warm inference.
    case basic
}

/// The compact verification result reported by `/health`. The per-function list stays in the
/// startup log: 41 entries are not a health payload.
struct JinaVerificationSummary: Encodable, Sendable, Equatable {
    var mode: String
    var functions: Int?
    var passed: Int?
    var failed: Int?
    /// `model.function: reason` for the first failing function.
    var first_failure: String?
    var min_consistency_cosine: Double?
    var wall_ms: Double?
    var footprint_delta_bytes: Int64?

    init(mode: JinaStartupVerification) { self.mode = mode.rawValue }

    init(report: OmniSmallVerificationReport, firstFailure: String? = nil) {
        mode = JinaStartupVerification.full.rawValue
        functions = report.functions.count
        passed = report.passedCount
        failed = report.failedCount
        first_failure = firstFailure
        min_consistency_cosine = report.minimumCosine
        wall_ms = report.wallMilliseconds
        footprint_delta_bytes = report.footprintDeltaBytes
    }
}

/// Where startup verification leaves its result for `/health`. It starts as just the mode and
/// is replaced once the run finishes, whether it passed or not.
actor JinaStartupStatus {
    private(set) var summary: JinaVerificationSummary

    init(mode: JinaStartupVerification) { summary = JinaVerificationSummary(mode: mode) }

    func record(_ summary: JinaVerificationSummary) { self.summary = summary }
}

/// Formats startup verification for the log. `Sendable` and lock-guarded because the runtime
/// reports each function from whichever thread ran it.
final class JinaVerificationLog: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    /// One line per function, in the server's `timestamped` style; failures go to stderr.
    func report(_ check: OmniSmallFunctionCheck) {
        let index = lock.withLock { count += 1; return count }
        timestamped(Self.line(check, index: index), error: !check.passed)
    }

    static func line(_ check: OmniSmallFunctionCheck, index: Int) -> String {
        guard check.passed else {
            return "verify #\(index) \(check.name) FAILED: \(check.failure ?? "unknown failure")"
        }
        var line = "verify #\(index) \(check.name) ok load=\(fixed(check.loadMilliseconds))ms run=\(fixed(check.runMilliseconds))ms"
        if let cosine = check.minimumCosine, let reference = check.referenceFunction {
            line += " cosine=\(String(format: "%.6f", cosine)) vs \(reference)"
        }
        return line
    }

    /// The readiness line for a passing run.
    static func readyLine(_ report: OmniSmallVerificationReport, modalities: [String]) -> String {
        let cosine = report.minimumCosine.map { String(format: "%.6f", $0) } ?? "n/a"
        let mebibytes = Double(report.footprintDeltaBytes) / 1_048_576
        return "ready: verified \(report.functions.count) functions in \(fixed(report.wallMilliseconds / 1000))s "
            + "(min consistency cosine \(cosine), footprint \(String(format: "%+.0f", mebibytes)) MiB); "
            + "modalities=\(modalities.joined(separator: ","))"
    }

    private static func fixed(_ value: Double) -> String { String(format: "%.1f", value) }
}

/// What runs between "the bundle validated" and "ready".
enum JinaStartup {
    /// One text embedding through the served path, with the accelerator lane held. Startup's
    /// `basic` mode is this alone, and the keep-warm loop repeats it.
    static func warm(model: OmniSmall, lane: AcceleratorLane, metrics: ServerMetrics) async throws {
        try await lane.acquire()
        do {
            _ = try await model.embedDocument(.text("warm"), dimensions: .d1024)
            await lane.release()
            await metrics.recordInference()
        } catch {
            await lane.release()
            throw error
        }
    }

    /// Every Core ML function once, with the lane held, each logged as it finishes.
    static func verifyAll(model: OmniSmall, lane: AcceleratorLane, metrics: ServerMetrics) async throws -> OmniSmallVerificationReport {
        try await lane.acquire()
        let log = JinaVerificationLog()
        do {
            let report = try await model.verifyAllFunctions(progress: { log.report($0) })
            await lane.release()
            await metrics.recordInference()
            return report
        } catch {
            await lane.release()
            throw error
        }
    }

    /// Runs the selected startup work and returns the readiness log line. Any throw keeps the server
    /// not ready, as a failed BidirLM placement audit does: a failed verification records its
    /// summary for `/health` (with the first failing function) and rethrows
    /// `OmniSmallVerificationError`, whose description names that function and why.
    static func run(
        mode: JinaStartupVerification, model: OmniSmall, lane: AcceleratorLane, metrics: ServerMetrics,
        status: JinaStartupStatus, modalities: [String]
    ) async throws -> String {
        let started = ContinuousClock.now
        switch mode {
        case .basic:
            try await warm(model: model, lane: lane, metrics: metrics)
            let seconds = elapsedMilliseconds(ContinuousClock.now - started) / 1000
            return "ready: validated bundle, text inference warmed in \(String(format: "%.1f", seconds))s (startup verification basic); modalities=\(modalities.joined(separator: ","))"
        case .full:
            timestamped("startup verification: loading and running every Core ML function (readiness waits for it)")
            let report: OmniSmallVerificationReport
            do {
                report = try await verifyAll(model: model, lane: lane, metrics: metrics)
            } catch let failure as OmniSmallVerificationError {
                await status.record(JinaVerificationSummary(
                    report: failure.report, firstFailure: "\(failure.function): \(failure.reason)"))
                throw failure
            }
            await status.record(JinaVerificationSummary(report: report))
            // The public path end to end, as `basic` does: verification runs the native functions
            // directly, this runs the served text call.
            try await warm(model: model, lane: lane, metrics: metrics)
            return JinaVerificationLog.readyLine(report, modalities: modalities)
        }
    }
}

/// The machine the numbers were calibrated for: batch thresholds (`NativeTextBatchPolicy`) and a W8
/// numeric fix were measured on an Apple M4 Max. Reported so a deployment on other silicon shows it.
enum HardwareIdentity {
    /// `machdep.cpu.brand_string`, e.g. `Apple M4 Max`.
    static let chip: String = sysctlString("machdep.cpu.brand_string") ?? sysctlString("hw.model") ?? "unknown"
    /// Marketing version and build, e.g. `15.4.1 (24E263)`.
    static let macOS: String = {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let base = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        return sysctlString("kern.osversion").map { "\(base) (\($0))" } ?? base
    }()

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        let text = String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
        return text.isEmpty ? nil : text
    }
}

// MARK: - HTTP service

struct JinaHealthResponse: Encodable, Sendable {
    var status: String
    var ready: Bool
    var version: String
    var model: String
    var family: String
    /// The default output dimension, also reported as `default_dimensions` for older clients.
    var dimensions: Int
    var default_dimensions: Int
    var supported_dimensions: [Int]
    var space: String
    var spaces: [String: String]
    var compute: String
    var retrieval_roles: [String]
    var default_role: String
    var max_tokens: Int
    var modalities: [String]
    var video_recipe: String?
    /// What startup proved before readiness (see `--startup-verification`).
    var startup_verification: JinaVerificationSummary
    /// The chip and macOS the process runs on; batch thresholds were calibrated on an Apple M4 Max.
    var chip: String
    var macos: String
    var port: Int
    var uptime_seconds: Int
    var ready_at: Int?
    var queue_requests: Int
    var queue_items: Int
    var batching_requests: Int
    var batching_rows: Int
    var batching_executing: Bool
    var keep_warm_seconds: Int
    var last_keep_warm_at: Int?
    var last_keep_warm_error: String?
    var keep_warm_failures: Int
    var fixture: Bool
    var error: String?
}

struct JinaEmbeddingsService: Sendable {
    let config: JinaServiceConfig
    let bundle: GlossModelBundle
    let store: JinaModelStore
    let tokenizers: JinaTokenizerPool
    let state: RuntimeState
    let admission: AdmissionGate
    let batcher: JinaTextBatcher
    let lane: AcceleratorLane
    let metrics: ServerMetrics
    let startup: JinaStartupStatus
    let docsContext: DocsPage.Context
    let startedAt: Int
    let isFixture: Bool

    var servedModelName: String { config.modelName ?? bundle.manifest.modelID }
    private var matryoshka: [Int] { bundle.capabilities.matryoshkaDimensions.sorted() }

    /// Modalities this bundle serves, in request-type order.
    static func modalities(_ bundle: GlossModelBundle) -> [String] {
        var names = ["text"]
        if bundle.capabilities.supportsImage { names.append("image") }
        if bundle.capabilities.supportsAudio { names.append("audio") }
        if bundle.capabilities.supportsVideo { names.append("video") }
        return names
    }

    func route(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/docs"), ("GET", "/docs/"): return DocsPage.response(docsContext)
        case ("GET", "/"): return .init(status: 302, contentType: "text/plain; charset=utf-8", extraHeaders: [("Location", "/docs")], body: Data())
        case ("GET", "/live"): return live()
        case ("GET", "/health"), ("GET", "/healthz"), ("GET", "/ready"): return await health()
        case ("GET", "/metrics"): return await prometheusMetrics()
        case ("GET", "/v1/models"): return ModelListResponse(data: [modelObject()]).asResponse()
        case ("GET", _) where request.path.hasPrefix("/v1/models/"):
            return model(pathComponent: percentDecode(String(request.path.dropFirst("/v1/models/".count))))
        case ("POST", "/v1/embeddings"): return await embeddings(request)
        case (_, "/docs"), (_, "/docs/"), (_, "/"), (_, "/live"), (_, "/health"), (_, "/healthz"), (_, "/ready"), (_, "/metrics"), (_, "/v1/models"):
            return .error(405, .invalidRequest("method not allowed; use GET"), extraHeaders: [("Allow", "GET")])
        case (_, _) where request.path.hasPrefix("/v1/models/"):
            return .error(405, .invalidRequest("method not allowed; use GET"), extraHeaders: [("Allow", "GET")])
        case (_, "/v1/embeddings"):
            return .error(405, .invalidRequest("method not allowed; use POST"), extraHeaders: [("Allow", "POST")])
        default: return .error(404, .invalidRequest("unknown route: \(request.method) \(request.path)", code: "not_found"))
        }
    }

    private func live() -> HTTPResponse {
        .json(200, LiveResponse(status: "ok", version: BuildInfo.version, model: servedModelName, port: Int(config.port)))
    }

    private func health() async -> HTTPResponse {
        let runtime = await state.snapshot()
        let queue = await admission.snapshot()
        let batching = await batcher.snapshot()
        let verification = await startup.summary
        var spaces = [String: String]()
        for d in matryoshka {
            if let dims = OmniSmall.Dimensions(rawValue: d) { spaces[String(d)] = store.space(for: dims) }
        }
        let payload = JinaHealthResponse(
            status: runtime.phase,
            ready: runtime.ready,
            version: BuildInfo.version,
            model: servedModelName,
            family: ModelFamily.jinaOmniSmall.rawValue,
            dimensions: config.defaultDimensions.rawValue,
            default_dimensions: config.defaultDimensions.rawValue,
            supported_dimensions: matryoshka,
            space: store.space(for: config.defaultDimensions),
            spaces: spaces,
            compute: JinaLimits.computeDescription,
            retrieval_roles: JinaLimits.roles,
            default_role: "document",
            max_tokens: JinaLimits.maximumTokensPerInput,
            modalities: Self.modalities(bundle),
            video_recipe: bundle.capabilities.supportsVideo ? OmniSmall.videoRecipe : nil,
            startup_verification: verification,
            chip: HardwareIdentity.chip,
            macos: HardwareIdentity.macOS,
            port: Int(config.port),
            uptime_seconds: max(0, Int(Date().timeIntervalSince(runtime.startedAt))),
            ready_at: runtime.readyAt.map { Int($0.timeIntervalSince1970) },
            queue_requests: queue.requests,
            queue_items: queue.items,
            batching_requests: batching.requests,
            batching_rows: batching.rows,
            batching_executing: batching.executing,
            keep_warm_seconds: config.keepWarmSeconds,
            last_keep_warm_at: runtime.lastKeepWarmAt.map { Int($0.timeIntervalSince1970) },
            last_keep_warm_error: runtime.lastKeepWarmError,
            keep_warm_failures: runtime.keepWarmFailures,
            fixture: isFixture,
            error: runtime.startupError)
        return .json(runtime.ready ? 200 : 503, payload)
    }

    private func prometheusMetrics() async -> HTTPResponse {
        let runtime = await state.snapshot()
        let counters = await metrics.snapshot()
        let queue = await admission.snapshot()
        let batching = await batcher.snapshot()
        var lines: [String] = []
        func gauge(_ help: String, _ name: String, _ value: String) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) gauge", "\(name) \(value)"]
        }
        func counter(_ help: String, _ name: String, _ value: UInt64) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) counter", "\(name) \(value)"]
        }
        func labeled(_ help: String, _ name: String, _ label: String, _ values: [String: UInt64], _ keys: [String]) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) counter"]
            for key in keys { lines.append("\(name){\(label)=\"\(key)\"} \(values[key, default: 0])") }
        }
        gauge("Whether the service is ready.", "gloss_ready", runtime.ready ? "1" : "0")
        gauge("Process uptime seconds.", "gloss_uptime_seconds", String(format: "%.3f", Date().timeIntervalSince(runtime.startedAt)))
        gauge("Admitted logical requests.", "gloss_queue_requests", String(queue.requests))
        gauge("Admitted input items.", "gloss_queue_items", String(queue.items))
        gauge("Text rows waiting in dynamic batching.", "gloss_batching_rows", String(batching.rows))
        gauge("Whether text execution is active.", "gloss_batching_executing", batching.executing ? "1" : "0")
        counter("HTTP requests handled.", "gloss_http_requests_total", counters.httpRequests)
        counter("Embedding requests accepted.", "gloss_embedding_requests_total", counters.embeddingRequests)
        counter("Embedding input items accepted.", "gloss_embedding_items_total", counters.embeddingItems)
        counter("Embedding requests rejected by admission.", "gloss_rejected_requests_total", counters.rejectedRequests)
        counter("Embedding requests failed after admission.", "gloss_failed_requests_total", counters.failedRequests)
        let kinds = ["native_b64", "native_b32", "native_b16", "native_b8", "native_b4", "single"]
        labeled("Text executions by native batch kind.", "gloss_text_waves_total", "kind", counters.wavesByKind, kinds)
        labeled("Text rows executed by native batch kind.", "gloss_text_rows_total", "kind", counters.rowsByKind, kinds)
        labeled("Conditioned text tokens executed by native batch kind.", "gloss_text_tokens_total", "kind", counters.tokensByKind, kinds)
        counter("Text waves that combined several requests.", "gloss_coalesced_waves_total", counters.coalescedWaves)
        counter("Failed multi-row text waves re-run one row at a time.", "gloss_text_isolation_retries_total", counters.isolationRetries)
        counter("Rows that still failed when re-run alone.", "gloss_text_isolation_failed_rows_total", counters.isolationFailedRows)
        labeled("Media items embedded.", "gloss_media_items_total", "kind", counters.mediaItems, ["image", "audio", "video"])
        gauge("Configured admission request limit.", "gloss_admission_max_requests", String(queue.maxRequests))
        gauge("Configured admission item limit.", "gloss_admission_max_items", String(queue.maxItems))
        gauge("Dynamic batching collection window milliseconds.", "gloss_batch_window_milliseconds", String(format: "%.3f", config.batchWindowMilliseconds))
        counter("Keep-warm attempts.", "gloss_keep_warm_attempts_total", counters.keepWarmAttempts)
        counter("Keep-warm failures.", "gloss_keep_warm_failures_total", counters.keepWarmFailures)
        if let last = counters.lastInferenceUnix {
            gauge("Unix timestamp of last inference.", "gloss_last_inference_unixtime", String(format: "%.3f", last))
        }
        lines.append("")
        return .init(status: 200, contentType: "text/plain; version=0.0.4; charset=utf-8",
                     extraHeaders: [("Cache-Control", "no-store")], body: Data(lines.joined(separator: "\n").utf8))
    }

    private func model(pathComponent: String) -> HTTPResponse {
        pathComponent == servedModelName ? modelObject().asResponse() : .error(404, .modelNotFound(pathComponent))
    }

    private func modelObject() -> ModelObject { .init(id: servedModelName, created: startedAt, owned_by: "glossematics") }

    // MARK: embeddings

    private func embeddings(_ request: HTTPRequest) async -> HTTPResponse {
        guard request.contentTypeIsJSON else {
            return .error(415, .invalidRequest("Content-Type must be application/json", code: "unsupported_media_type"))
        }
        let body: EmbeddingsRequestBody
        do {
            body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: request.body)
        } catch {
            return .error(400, .invalidRequest(decodingMessage(error), param: "input"))
        }
        guard let requested = body.model else {
            return .error(400, .invalidRequest("you must provide a model parameter", param: "model"))
        }
        guard requested == servedModelName else { return .error(404, .modelNotFound(requested)) }
        guard !body.items.isEmpty else {
            return .error(400, .invalidRequest("input must not be an empty array", param: "input"))
        }
        let maximumItems = min(config.maxBatch, JinaLimits.maximumItems)
        guard body.items.count <= maximumItems else {
            return .error(400, .invalidRequest("input exceeds the maximum of \(maximumItems) items per request", param: "input"))
        }
        let base64: Bool
        switch body.encodingFormat?.lowercased() {
        case nil, "float": base64 = false
        case "base64": base64 = true
        default:
            return .error(400, .invalidRequest("encoding_format must be \"float\" or \"base64\"", param: "encoding_format"))
        }
        let dimensions: OmniSmall.Dimensions
        if let requestedDimensions = body.dimensions {
            guard let value = OmniSmall.Dimensions(rawValue: requestedDimensions),
                  matryoshka.contains(requestedDimensions) else {
                return .error(400, .invalidRequest(
                    "dimensions must be one of \(matryoshka) for this model", param: "dimensions"))
            }
            dimensions = value
        } else {
            dimensions = config.defaultDimensions
        }
        guard await state.isReady() else {
            return .error(503, .serviceUnavailable("model is not ready"), extraHeaders: [("Retry-After", "5")])
        }
        let itemCount = body.items.count
        guard await admission.tryAcquire(items: itemCount) else {
            await metrics.recordRejected()
            return .error(503, .serviceUnavailable("embedding queue is full; retry shortly"), extraHeaders: [("Retry-After", "1")])
        }
        var response = await embeddingsAdmitted(
            body: body, dimensions: dimensions, role: body.retrievalRole ?? JinaLimits.defaultRole, base64: base64)
        response.onComplete = { await admission.release(items: itemCount) }
        return response
    }

    private func embeddingsAdmitted(
        body: EmbeddingsRequestBody, dimensions: OmniSmall.Dimensions, role: RetrievalRole, base64: Bool
    ) async -> HTTPResponse {
        await metrics.recordEmbeddingRequest(items: body.items.count)
        var inputs: [OmniSmall.Input] = []
        var tokenRows: [[Int32]] = []
        var tokenCounts: [Int] = []
        var promptTokens = 0
        var hasMedia = false
        var hasVideo = false
        var hasTokenIDs = false
        let capabilities = bundle.capabilities

        do {
            for (index, item) in body.items.enumerated() {
                try Task.checkCancellation()
                switch item {
                case let .text(text):
                    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        throw APIError.invalidRequest("input \(index) must not be an empty string", param: "input")
                    }
                    let ids = tokenizers.tokenIDs(text, role: role)
                    guard ids.count <= JinaLimits.maximumTokensPerInput else {
                        throw APIError.invalidRequest(
                            "input \(index) has \(ids.count) tokens including the \(role == .query ? "query" : "document") prompt; maximum is \(JinaLimits.maximumTokensPerInput)",
                            param: "input")
                    }
                    tokenRows.append(ids)
                    tokenCounts.append(ids.count)
                    promptTokens += ids.count
                    inputs.append(.text(text))
                case let .tokenIDs(ids):
                    guard !ids.isEmpty else {
                        throw APIError.invalidRequest("input \(index): token-ID inputs must not be empty", param: "input")
                    }
                    guard ids.count <= JinaLimits.maximumTokensPerInput else {
                        throw APIError.invalidRequest(
                            "input \(index): token-ID input has \(ids.count) IDs; maximum is \(JinaLimits.maximumTokensPerInput)",
                            param: "input")
                    }
                    if let vocabulary = tokenizers.vocabulary, !ids.allSatisfy({ Int($0) < vocabulary }) {
                        throw APIError.invalidRequest(
                            "input \(index): token IDs must be below the vocabulary size \(vocabulary)", param: "input")
                    }
                    hasTokenIDs = true
                    tokenRows.append(ids)
                    tokenCounts.append(ids.count)
                    promptTokens += ids.count
                    inputs.append(.text(""))   // placeholder keeps positions aligned; never executed
                case let .image(data):
                    guard capabilities.supportsImage else { throw unsupported("image", index) }
                    hasMedia = true
                    inputs.append(.imageData(data))
                case let .audio(data):
                    guard capabilities.supportsAudio else { throw unsupported("audio", index) }
                    hasMedia = true
                    inputs.append(.audioData(data))
                case let .video(data):
                    guard capabilities.supportsVideo else { throw unsupported("video", index) }
                    hasMedia = true
                    hasVideo = true
                    inputs.append(.videoData(data))
                case .message:
                    throw APIError.invalidRequest(
                        "input \(index): interleaved message inputs are not supported by this model; send text, image, audio, and video as separate inputs",
                        param: "input", code: "unsupported_modality")
                }
                guard promptTokens <= config.maxRequestTokens else {
                    throw APIError.invalidRequest(
                        "total input tokens exceed the per-request maximum of \(config.maxRequestTokens)", param: "input")
                }
            }
            guard !(hasMedia && hasTokenIDs) else {
                throw APIError.invalidRequest("token-ID inputs cannot be combined with media inputs", param: "input")
            }
        } catch is CancellationError {
            await metrics.recordFailure()
            return .error(503, .serviceUnavailable("request was cancelled"))
        } catch let error as APIError {
            await metrics.recordFailure()
            return .error(400, error)
        } catch {
            await metrics.recordFailure()
            return .error(400, .invalidRequest(String(describing: error), param: "input"))
        }

        do {
            let values: [[Float]]
            let space: String
            if !hasMedia {
                let result = try await batcher.submit(tokenIDRows: tokenRows, tokenCounts: tokenCounts, dimensions: dimensions)
                values = result.values
                space = result.space
            } else {
                let model = await store.get()
                var ordered = [[Float]]()
                ordered.reserveCapacity(inputs.count)
                var start = 0
                var textOffset = 0
                while start < inputs.count {
                    try Task.checkCancellation()
                    let end = JinaMediaRequestPlanner.nextChunkEnd(inputs, from: start)
                    if case .text = inputs[start] {
                        let count = end - start
                        guard textOffset + count <= tokenRows.count else {
                            throw OmniSmallError.inferenceFailed("mixed request text rows are incomplete")
                        }
                        do {
                            let result = try await batcher.submit(
                                tokenIDRows: Array(tokenRows[textOffset..<(textOffset + count)]),
                                tokenCounts: Array(tokenCounts[textOffset..<(textOffset + count)]),
                                dimensions: dimensions)
                            ordered.append(contentsOf: result.values)
                        } catch OmniSmallError.invalidBatchInput(let localIndex, let reason) {
                            throw OmniSmallError.invalidBatchInput(index: start + localIndex, reason: reason)
                        }
                        textOffset += count
                    } else {
                        // CPU preparation first, with no accelerator permit; Core ML second, with it.
                        let input = inputs[start]
                        ordered.append(try await Self.embedMediaItem(
                            input, at: start, role: role, dimensions: dimensions, model: model, lane: lane))
                        await metrics.recordMedia(kind: Self.mediaKind(input))
                    }
                    start = end
                }
                guard textOffset == tokenRows.count, ordered.count == inputs.count else {
                    throw OmniSmallError.inferenceFailed("mixed request result count is inconsistent")
                }
                values = ordered
                await metrics.recordInference()
                space = store.space(for: dimensions)
            }
            let data = values.enumerated().map {
                EmbeddingObject(index: $0.offset,
                                embedding: base64 ? .base64(EmbeddingsService.base64LittleEndian($0.element)) : .floats($0.element))
            }
            var headers = [
                ("X-Glossematics-Space", space),
                ("X-Glossematics-Dimensions", String(dimensions.rawValue)),
                ("X-Glossematics-Role", role == .query ? "query" : "document"),
                ("X-Glossematics-Usage-Scope", hasMedia ? "text-only" : "all-inputs"),
            ]
            // Video vectors from different decode/sampling recipes share a space ID by design, so a
            // client indexing video needs the recipe that produced them.
            if hasVideo { headers.append(("X-Glossematics-Video-Recipe", OmniSmall.videoRecipe)) }
            return .json(
                200,
                EmbeddingsResponse(data: data, model: servedModelName,
                                   usage: .init(prompt_tokens: promptTokens, total_tokens: promptTokens)),
                extraHeaders: headers)
        } catch let error as OmniSmallError {
            await metrics.recordFailure()
            switch error {
            case .invalidInput, .invalidBatchInput:
                return .error(400, .invalidRequest(error.description, param: "input", code: "invalid_media"))
            default:
                return .error(500, .apiError(error.description))
            }
        } catch is CancellationError {
            await metrics.recordFailure()
            return .error(503, .serviceUnavailable("request was cancelled"))
        } catch {
            await metrics.recordFailure()
            return .error(500, .apiError(String(describing: error)))
        }
    }

    /// Embed one image, audio clip, or video. The CPU half (`prepareMedia`: decode, resize, mel, prompt
    /// ids) runs with NO accelerator permit, so a multi-second video decode blocks nothing; only the
    /// Core ML half holds the lane, and the lane is released on every path, cancellation included.
    /// `index` is the input's position in the request: a bad input is reported against it.
    static func embedMediaItem(
        _ input: OmniSmall.Input,
        at index: Int,
        role: RetrievalRole,
        dimensions: OmniSmall.Dimensions,
        model: OmniSmall,
        lane: AcceleratorLane
    ) async throws -> [Float] {
        let prepared: OmniSmallPreparedMedia
        do {
            prepared = try await model.prepareMedia(input, role: role == .query ? .query : .document)
        } catch OmniSmallError.invalidInput(let reason) {
            throw OmniSmallError.invalidBatchInput(index: index, reason: reason)
        } catch OmniSmallError.invalidBatchInput(let localIndex, let reason) {
            throw OmniSmallError.invalidBatchInput(index: index + localIndex, reason: reason)
        }
        try await lane.acquire()
        do {
            let vector = try await model.embedPreparedMedia(prepared, dimensions: dimensions)
            await lane.release()
            return vector
        } catch {
            await lane.release()
            throw error
        }
    }

    private func unsupported(_ kind: String, _ index: Int) -> APIError {
        .invalidRequest("input \(index): this bundle has no \(kind) tower", param: "input", code: "unsupported_modality")
    }

    private static func mediaKind(_ input: OmniSmall.Input) -> String {
        switch input {
        case .image, .imageData: "image"
        case .audio, .audioData: "audio"
        case .video, .videoData: "video"
        case .text: "text"
        }
    }

    private func percentDecode(_ s: String) -> String { s.removingPercentEncoding ?? s }

    private func decodingMessage(_ error: any Error) -> String {
        if let e = error as? DecodingError {
            switch e {
            case let .valueNotFound(_, c): return c.debugDescription
            case let .dataCorrupted(c): return c.debugDescription
            case let .keyNotFound(k, _): return "missing required parameter: '\(k.stringValue)'"
            case let .typeMismatch(_, c): return c.debugDescription
            @unknown default: return "malformed JSON body"
            }
        }
        return "malformed JSON body: \(String(describing: error))"
    }
}
