import Foundation
import Testing
@_spi(Server) @testable import gloss_server

// The Jina HTTP service's scheduling and accelerator discipline, against fake backends (no Core ML):
//
// * text waves: a failed multi-row batch is re-run row by row, so one bad row fails only its own
//   job; results stay in order; a job cancelled mid-wave is dropped without disturbing the rest;
// * media: CPU preparation holds no accelerator permit, Core ML does, and the permit is released
//   on every path;
// * startup verification summary, log lines, and the flag.

private let testSpace = "glossematics:omni-small:sha256:" + String(repeating: "a", count: 64)
private let testArtifact = "sha256:" + String(repeating: "c", count: 64)

/// First-token markers that make a row misbehave. Healthy rows use their own index (0..<32) as the
/// token, which the fake turns into a one-hot vector at that position.
private let serverFault: Int32 = 999   // any call containing it fails like a Core ML fault
private let badInput: Int32 = 998      // any call containing it rejects that row as invalid input

// MARK: - Text waves

private actor WaveBackend: OmniSmallBackend {
    /// The first token of every row, per `embedTexts` call, in call order.
    private(set) var calls: [[Int32]] = []
    private var gateArmed = false
    private var blocked = false
    private var blockedWaiters: [CheckedContinuation<Void, Never>] = []
    private var gate: CheckedContinuation<Void, Never>?

    func prepareTexts(_ texts: [String], role: OmniSmallRole) async throws -> [ValidatedText] {
        throw OmniSmallBackendError.failure("unused")
    }

    func embedTexts(_ rows: [ValidatedText], dimensions: OmniSmall.Dimensions) async throws -> [[Float]] {
        calls.append(rows.map { $0.tokenIDs[0] })
        if gateArmed, rows.count > 1 {
            // Hold the first multi-row call open so the test can act while a wave is executing.
            gateArmed = false
            blocked = true
            for waiter in blockedWaiters { waiter.resume() }
            blockedWaiters = []
            await withCheckedContinuation { gate = $0 }
        }
        for (position, row) in rows.enumerated() {
            switch row.tokenIDs[0] {
            case serverFault: throw OmniSmallBackendError.failure("poisoned row")
            case badInput: throw OmniSmallBackendError.item(index: position, reason: "bad row")
            default: break
            }
        }
        return rows.map { row in
            var vector = [Float](repeating: 0, count: dimensions.rawValue)
            vector[Int(row.tokenIDs[0]) % dimensions.rawValue] = 1
            return vector
        }
    }

    func embedMedia(_ input: OmniSmall.Input, role: OmniSmallRole, dimensions: OmniSmall.Dimensions) async throws -> [Float] {
        throw OmniSmallBackendError.failure("unused")
    }

    func armGate() { gateArmed = true }

    func waitUntilBlocked() async {
        if blocked { return }
        await withCheckedContinuation { blockedWaiters.append($0) }
    }

    func openGate() {
        blocked = false
        gate?.resume()
        gate = nil
    }
}

private struct WaveRig {
    let backend = WaveBackend()
    let lane = AcceleratorLane()
    let metrics = ServerMetrics()
    let batcher: JinaTextBatcher

    init() {
        let model = OmniSmall(backend: backend, dimensions: .d1024, space: testSpace, artifactFingerprint: testArtifact)
        batcher = JinaTextBatcher(store: JinaModelStore(model: model), lane: lane, metrics: metrics,
                                  windowMilliseconds: 5)
    }

    /// One row per token; every token is below 32, so its one-hot survives every Matryoshka size.
    func submit(_ tokens: [Int32], dimensions: OmniSmall.Dimensions = .d32) -> Task<JinaBatchedResult, any Error> {
        let batcher = batcher
        return Task {
            try await batcher.submit(
                tokenIDRows: tokens.map { [$0] }, tokenCounts: tokens.map { _ in 1 }, dimensions: dimensions)
        }
    }

    /// Hold the accelerator lane while the jobs are submitted, one at a time so they queue in the
    /// given order, then let them go as one wave.
    func collect(_ specs: [(tokens: [Int32], dimensions: OmniSmall.Dimensions)]) async -> [Task<JinaBatchedResult, any Error>] {
        try? await lane.acquire()
        var tasks: [Task<JinaBatchedResult, any Error>] = []
        for spec in specs {
            tasks.append(submit(spec.tokens, dimensions: spec.dimensions))
            await waitUntil { await batcher.snapshot().requests == tasks.count }
        }
        await lane.release()
        return tasks
    }
}

/// Whether the accelerator lane is free right now. The probe takes the permit and gives it back.
private func laneIsFree(_ lane: AcceleratorLane) async -> Bool {
    guard await lane.tryAcquire() else { return false }
    await lane.release()
    return true
}

private func waitUntil(_ condition: () async -> Bool) async {
    for _ in 0..<2_500 {
        if await condition() { return }
        try? await Task.sleep(for: .milliseconds(2))
    }
}

/// The token each output row must encode, checked position by position.
private func expectRows(_ result: JinaBatchedResult, tokens: [Int32], dimensions: Int,
                        sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(result.values.count == tokens.count, sourceLocation: sourceLocation)
    for (row, token) in zip(result.values, tokens) {
        #expect(row.count == dimensions, sourceLocation: sourceLocation)
        #expect(row.firstIndex(of: 1) == Int(token), sourceLocation: sourceLocation)
    }
}

/// The single-row calls a failed wave should make: rows enter a wave round-robin, one per job per
/// pass, and a job stops being re-run at its first poisoned row. Cancelled jobs are never re-run.
private func expectedSingleCalls(_ jobs: [[Int32]], cancelled: Set<Int> = []) -> [Int32] {
    var calls: [Int32] = []
    var failed = cancelled
    for pass in 0..<(jobs.map(\.count).max() ?? 0) {
        for (index, tokens) in jobs.enumerated() where pass < tokens.count && !failed.contains(index) {
            calls.append(tokens[pass])
            if tokens[pass] == serverFault { failed.insert(index) }
        }
    }
    return calls
}

private let tokensA = (0..<10).map(Int32.init)
private let tokensB = (10..<20).map(Int32.init)
private let tokensC = (20..<30).map(Int32.init)

@Test func healthyDenseWaveIsOneNativeBatchAndKeepsOrder() async throws {
    let rig = WaveRig()
    let jobs = await rig.collect([(tokensA, .d32), (tokensB, .d32), (tokensC, .d64)])
    expectRows(try await jobs[0].value, tokens: tokensA, dimensions: 32)
    expectRows(try await jobs[1].value, tokens: tokensB, dimensions: 32)
    expectRows(try await jobs[2].value, tokens: tokensC, dimensions: 64)

    #expect(await rig.backend.calls.count == 1, "30 rows in one bucket are one native batch")
    let counters = await rig.metrics.snapshot()
    #expect(counters.wavesByKind["native_b64"] == 1)
    #expect(counters.isolationRetries == 0)
    #expect(await laneIsFree(rig.lane), "the lane is free after the wave")
}

@Test func poisonedRowFailsOnlyItsOwnJobAndHealthyJobsSucceedInOrder() async throws {
    let rig = WaveRig()
    var poisoned = tokensB
    poisoned[4] = serverFault
    // Mixed sizes too: the wave executes at 64 and A/B are cut back to 32 afterwards.
    let jobs = await rig.collect([(tokensA, .d32), (poisoned, .d32), (tokensC, .d64)])
    let a = await jobs[0].result
    let b = await jobs[1].result
    let c = await jobs[2].result

    expectRows(try a.get(), tokens: tokensA, dimensions: 32)
    expectRows(try c.get(), tokens: tokensC, dimensions: 64)
    guard case let .failure(error) = b, case let OmniSmallError.inferenceFailed(reason) = error else {
        Issue.record("the poisoned job must fail with the row's error, got \(b)")
        return
    }
    #expect(reason.contains("poisoned row"))

    // One batch of 30, then each row alone; B's rows after the poisoned one are skipped, because
    // its job has already failed.
    let calls = await rig.backend.calls
    #expect(calls.first?.count == 30)
    #expect(calls.dropFirst().allSatisfy { $0.count == 1 })
    #expect(calls.dropFirst().flatMap { $0 } == expectedSingleCalls([tokensA, poisoned, tokensC]))
    let counters = await rig.metrics.snapshot()
    #expect(counters.isolationRetries == 1)
    #expect(counters.isolationFailedRows == 1)
    #expect(counters.wavesByKind["single"] == 24, "A's 10, B's 4 healthy rows, C's 10 ran alone")
    #expect(counters.wavesByKind["native_b64"] == nil)
    #expect(await laneIsFree(rig.lane), "the lane is free after isolation")
}

@Test func invalidRowInABatchIsReportedAtItsPositionInItsOwnJob() async throws {
    let rig = WaveRig()
    var invalid = tokensB
    invalid[4] = badInput
    let jobs = await rig.collect([(tokensA, .d32), (invalid, .d32), (tokensC, .d32)])
    expectRows(try await jobs[0].value, tokens: tokensA, dimensions: 32)
    expectRows(try await jobs[2].value, tokens: tokensC, dimensions: 32)
    do {
        _ = try await jobs[1].value
        Issue.record("the invalid row must fail its job")
    } catch {
        // Alone the row is index 0 to the model; the job knows it as row 4 of its request.
        #expect(error as? OmniSmallError == .invalidBatchInput(index: 4, reason: "bad row"))
    }
}

@Test func everyJobWithABadRowFailsAndTheRestSucceed() async throws {
    let rig = WaveRig()
    var firstBad = tokensA
    firstBad[0] = serverFault
    firstBad[9] = serverFault
    var secondBad = tokensC
    secondBad[7] = serverFault
    let jobs = await rig.collect([(firstBad, .d32), (tokensB, .d32), (secondBad, .d32)])
    let results = [await jobs[0].result, await jobs[1].result, await jobs[2].result]
    if case .success = results[0] { Issue.record("job 0 has bad rows") }
    if case .success = results[2] { Issue.record("job 2 has a bad row") }
    expectRows(try results[1].get(), tokens: tokensB, dimensions: 32)
    let counters = await rig.metrics.snapshot()
    #expect(counters.isolationFailedRows == 2, "job 0 stops at its first bad row")
}

@Test func lowVolumeRowFailureIsNotRetriedAndKeepsItsRowIndex() async throws {
    // Four rows are far below the native-batch threshold: one row per turn, so there is nothing to
    // isolate. The job still fails with the row's own position, not the position in the turn.
    let rig = WaveRig()
    let job = rig.submit([1, badInput, 3, 4])
    do {
        _ = try await job.value
        Issue.record("the invalid row must fail the job")
    } catch {
        #expect(error as? OmniSmallError == .invalidBatchInput(index: 1, reason: "bad row"))
    }
    #expect(await rig.metrics.snapshot().isolationRetries == 0)
    #expect(await laneIsFree(rig.lane))
}

@Test func jobCancelledMidWaveIsDroppedAndTheOthersComplete() async throws {
    let rig = WaveRig()
    await rig.backend.armGate()
    let jobs = await rig.collect([(tokensA, .d32), (tokensB, .d32), (tokensC, .d32)])
    await rig.backend.waitUntilBlocked()
    jobs[0].cancel()
    let cancelled = await jobs[0].result
    if case .failure(let error) = cancelled { #expect(error is CancellationError) } else {
        Issue.record("a cancelled job must not return a result")
    }
    await rig.backend.openGate()

    expectRows(try await jobs[1].value, tokens: tokensB, dimensions: 32)
    expectRows(try await jobs[2].value, tokens: tokensC, dimensions: 32)
    #expect(await rig.metrics.snapshot().isolationRetries == 0)
    #expect(await laneIsFree(rig.lane))

    // The scheduler keeps serving: its fairness and admission state survived the cancellation.
    let later = rig.submit([5, 6])
    expectRows(try await later.value, tokens: [5, 6], dimensions: 32)
}

@Test func cancelledJobsRowsAreSkippedWhenAFailedWaveIsIsolated() async throws {
    let rig = WaveRig()
    var poisoned = tokensB
    poisoned[2] = serverFault
    await rig.backend.armGate()
    let jobs = await rig.collect([(tokensA, .d32), (poisoned, .d32), (tokensC, .d32)])
    await rig.backend.waitUntilBlocked()
    jobs[0].cancel()
    _ = await jobs[0].result
    await rig.backend.openGate()

    let b = await jobs[1].result
    if case .success = b { Issue.record("the poisoned job must fail") }
    expectRows(try await jobs[2].value, tokens: tokensC, dimensions: 32)

    // Nobody waits for A any more: none of its rows is re-run.
    let singles = await rig.backend.calls.dropFirst().flatMap { $0 }
    #expect(singles.filter { tokensA.contains($0) }.isEmpty)
    #expect(singles == expectedSingleCalls([tokensA, poisoned, tokensC], cancelled: [0]))
    #expect(await laneIsFree(rig.lane))
}

// MARK: - Media lane discipline

private actor MediaBackend: OmniSmallBackend {
    enum Behavior { case ok, invalid, embedFails, embedHangs }

    private let lane: AcceleratorLane
    private let behavior: Behavior
    private(set) var events: [String] = []

    init(lane: AcceleratorLane, behavior: Behavior) {
        self.lane = lane
        self.behavior = behavior
    }

    func prepareTexts(_ texts: [String], role: OmniSmallRole) async throws -> [ValidatedText] {
        throw OmniSmallBackendError.failure("unused")
    }

    func embedTexts(_ rows: [ValidatedText], dimensions: OmniSmall.Dimensions) async throws -> [[Float]] {
        throw OmniSmallBackendError.failure("unused")
    }

    func embedMedia(_ input: OmniSmall.Input, role: OmniSmallRole, dimensions: OmniSmall.Dimensions) async throws -> [Float] {
        throw OmniSmallBackendError.failure("unused")
    }

    func prepareMedia(_ input: OmniSmall.Input, role: OmniSmallRole) async throws -> OmniSmallPreparedMedia {
        events.append("prepare role=\(role == .query ? "query" : "document") lane_free=\(await laneIsFree(lane))")
        if behavior == .invalid { throw OmniSmallBackendError.invalidInput("undecodable image") }
        return OmniSmallPreparedMedia(
            kind: .image, role: role, prepareMilliseconds: 0, queuedMilliseconds: 0, payload: .passthrough(input))
    }

    func embedPreparedMedia(_ prepared: OmniSmallPreparedMedia, dimensions: OmniSmall.Dimensions) async throws -> [Float] {
        events.append("embed lane_free=\(await laneIsFree(lane))")
        switch behavior {
        case .embedFails: throw OmniSmallBackendError.failure("prediction failed")
        case .embedHangs: try await Task.sleep(for: .seconds(60))
        case .ok, .invalid: break
        }
        var vector = [Float](repeating: 0, count: dimensions.rawValue)
        vector[3] = 1
        return vector
    }
}

private func mediaRig(_ behavior: MediaBackend.Behavior) -> (backend: MediaBackend, model: OmniSmall, lane: AcceleratorLane) {
    let lane = AcceleratorLane()
    let backend = MediaBackend(lane: lane, behavior: behavior)
    let model = OmniSmall(backend: backend, dimensions: .d1024, space: testSpace, artifactFingerprint: testArtifact)
    return (backend, model, lane)
}

private let image = OmniSmall.Input.imageData(Data([1, 2, 3]))

@Test func mediaIsPreparedWithoutTheLaneAndEmbeddedWithIt() async throws {
    let (backend, model, lane) = mediaRig(.ok)
    let vector = try await JinaEmbeddingsService.embedMediaItem(
        image, at: 0, role: .query, dimensions: .d32, model: model, lane: lane)
    #expect(vector.count == 32 && vector[3] == 1)
    #expect(await backend.events == ["prepare role=query lane_free=true", "embed lane_free=false"])
    #expect(await laneIsFree(lane), "the lane is released after the item")

    // The role follows the request.
    _ = try await JinaEmbeddingsService.embedMediaItem(
        image, at: 0, role: .document, dimensions: .d32, model: model, lane: lane)
    #expect(await backend.events.last(where: { $0.hasPrefix("prepare") }) == "prepare role=document lane_free=true")
}

@Test func badMediaIsReportedAgainstItsInputIndexAndNeverTakesTheLane() async throws {
    let (backend, model, lane) = mediaRig(.invalid)
    do {
        _ = try await JinaEmbeddingsService.embedMediaItem(
            image, at: 7, role: .document, dimensions: .d32, model: model, lane: lane)
        Issue.record("an undecodable image must fail")
    } catch {
        #expect(error as? OmniSmallError == .invalidBatchInput(index: 7, reason: "undecodable image"))
    }
    #expect(await backend.events.count == 1, "no Core ML step ran")
    #expect(await laneIsFree(lane))
}

@Test func laneIsReleasedWhenTheAcceleratorHalfFails() async throws {
    let (_, model, lane) = mediaRig(.embedFails)
    do {
        _ = try await JinaEmbeddingsService.embedMediaItem(
            image, at: 0, role: .document, dimensions: .d32, model: model, lane: lane)
        Issue.record("a failed prediction must fail the item")
    } catch {
        guard case OmniSmallError.inferenceFailed = error else {
            Issue.record("a prediction failure is a server error, got \(error)")
            return
        }
    }
    #expect(await laneIsFree(lane))
}

@Test func laneIsReleasedWhenTheRequestIsCancelledInsideCoreML() async throws {
    let (backend, model, lane) = mediaRig(.embedHangs)
    let task = Task {
        try await JinaEmbeddingsService.embedMediaItem(
            image, at: 0, role: .document, dimensions: .d32, model: model, lane: lane)
    }
    await waitUntil { await backend.events.count == 2 }
    task.cancel()
    do {
        _ = try await task.value
        Issue.record("a cancelled item must not return a vector")
    } catch {
        #expect(error is CancellationError)
    }
    #expect(await laneIsFree(lane), "cancellation must not leak the accelerator permit")
}

@Test func laneIsFreeForOthersWhileAnotherRequestPrepares() async throws {
    // A request waiting for the lane (its own Core ML half) does not stop another request from
    // preparing: preparation never touches the lane.
    let (backend, model, lane) = mediaRig(.ok)
    try await lane.acquire()
    let waiting = Task {
        try await JinaEmbeddingsService.embedMediaItem(
            image, at: 0, role: .document, dimensions: .d32, model: model, lane: lane)
    }
    await waitUntil { await backend.events.count == 1 }
    #expect(await backend.events == ["prepare role=document lane_free=false"],
            "prepare ran while another holder had the lane")
    await lane.release()
    let vector = try await waiting.value
    #expect(vector.count == 32)
}

// MARK: - Startup verification and hardware

private func check(_ name: String, cosine: Double? = nil, reference: String? = nil,
                   failure: String? = nil) -> OmniSmallFunctionCheck {
    let parts = name.split(separator: ".").map(String.init)
    return OmniSmallFunctionCheck(
        model: parts[0], function: parts[1], computeUnits: "cpu+ane", loadMilliseconds: 12.34,
        runMilliseconds: 1.5, referenceFunction: reference, minimumCosine: cosine,
        cosineThreshold: cosine == nil ? nil : 0.99, passed: failure == nil, failure: failure)
}

private actor StartupBackend: OmniSmallBackend {
    private let failing: Bool
    private(set) var verifyCalls = 0
    private(set) var warmCalls = 0

    init(failing: Bool) { self.failing = failing }

    func prepareTexts(_ texts: [String], role: OmniSmallRole) async throws -> [ValidatedText] {
        try texts.map { _ in try ValidatedText(tokenIDs: [1]) }
    }

    func embedTexts(_ rows: [ValidatedText], dimensions: OmniSmall.Dimensions) async throws -> [[Float]] {
        warmCalls += 1
        return rows.map { _ in
            var vector = [Float](repeating: 0, count: dimensions.rawValue)
            vector[0] = 1
            return vector
        }
    }

    func embedMedia(_ input: OmniSmall.Input, role: OmniSmallRole, dimensions: OmniSmall.Dimensions) async throws -> [Float] {
        throw OmniSmallBackendError.failure("unused")
    }

    func verifyAllFunctions(
        validateEmbedding: @escaping @Sendable ([Float]) throws -> Void,
        progress: (@Sendable (OmniSmallFunctionCheck) -> Void)?
    ) async throws -> OmniSmallVerificationReport {
        verifyCalls += 1
        let functions = [
            check("text.bucket_32"),
            check("text.bucket_64", cosine: 0.9999, reference: "bucket_32"),
            failing
                ? check("image.f1024", cosine: 0.4, reference: "f512", failure: "disagrees with f512")
                : check("image.f1024", cosine: 0.9998, reference: "f512"),
        ]
        for function in functions { progress?(function) }
        let report = OmniSmallVerificationReport(functions: functions, wallMilliseconds: 10)
        if failing {
            throw OmniSmallVerificationError(
                function: "image.f1024", reason: .inconsistent(cosine: 0.4, threshold: 0.99, reference: "f512"),
                report: report)
        }
        return report
    }
}

private func startupRig(failing: Bool = false, mode: JinaStartupVerification = .full)
    -> (backend: StartupBackend, model: OmniSmall, lane: AcceleratorLane, metrics: ServerMetrics,
        status: JinaStartupStatus) {
    let backend = StartupBackend(failing: failing)
    let model = OmniSmall(backend: backend, dimensions: .d1024, space: testSpace, artifactFingerprint: testArtifact)
    return (backend, model, AcceleratorLane(), ServerMetrics(), JinaStartupStatus(mode: mode))
}

@Test func fullStartupVerifiesEveryFunctionThenRunsTheServedWarm() async throws {
    let rig = startupRig()
    let line = try await JinaStartup.run(
        mode: .full, model: rig.model, lane: rig.lane, metrics: rig.metrics, status: rig.status,
        modalities: ["text", "image"])
    #expect(line.hasPrefix("ready: verified 3 functions in "))
    #expect(await rig.backend.verifyCalls == 1)
    #expect(await rig.backend.warmCalls == 1, "the served text path runs once after verification")
    let summary = await rig.status.summary
    #expect(summary.mode == "full" && summary.functions == 3 && summary.passed == 3 && summary.failed == 0)
    #expect(summary.first_failure == nil)
    #expect(await laneIsFree(rig.lane))
    #expect(await rig.metrics.snapshot().lastInferenceUnix != nil)
}

@Test func failedVerificationKeepsTheServerNotReadyAndNamesTheFunction() async throws {
    let rig = startupRig(failing: true)
    do {
        _ = try await JinaStartup.run(
            mode: .full, model: rig.model, lane: rig.lane, metrics: rig.metrics, status: rig.status,
            modalities: ["text"])
        Issue.record("a failed verification must throw, so the server never becomes ready")
    } catch let failure as OmniSmallVerificationError {
        #expect(failure.function == "image.f1024")
        #expect(failure.description.contains("image.f1024") && failure.description.contains("disagrees with f512"))
    }
    #expect(await rig.backend.warmCalls == 0, "nothing else runs after a failed verification")
    let summary = await rig.status.summary
    #expect(summary.functions == 3 && summary.passed == 2 && summary.failed == 1)
    #expect(summary.first_failure?.hasPrefix("image.f1024: ") == true)
    #expect(summary.first_failure?.contains("disagrees with f512") == true)
    #expect(await laneIsFree(rig.lane), "the lane is released after a failed verification")
}

@Test func basicStartupRunsOnlyTheSingleWarm() async throws {
    let rig = startupRig(mode: .basic)
    let line = try await JinaStartup.run(
        mode: .basic, model: rig.model, lane: rig.lane, metrics: rig.metrics, status: rig.status,
        modalities: ["text"])
    #expect(line.contains("text inference warmed") && line.contains("startup verification basic"))
    #expect(await rig.backend.verifyCalls == 0)
    #expect(await rig.backend.warmCalls == 1)
    #expect(await rig.status.summary == JinaVerificationSummary(mode: .basic), "basic reports only its mode")
    #expect(await laneIsFree(rig.lane))
}

@Test func verificationSummaryIsCompactAndNamesTheFirstFailure() throws {
    let report = OmniSmallVerificationReport(
        functions: [
            check("text.bucket_32"),
            check("text.bucket_64", cosine: 0.9999, reference: "bucket_32"),
            check("image.f1024", cosine: 0.42, reference: "f512", failure: "disagrees with f512"),
        ],
        wallMilliseconds: 2_500, footprintBytesBefore: 100, footprintBytesAfter: 1_100)
    let summary = JinaVerificationSummary(report: report, firstFailure: "image.f1024: disagrees with f512")
    #expect(summary.mode == "full")
    #expect(summary.functions == 3 && summary.passed == 2 && summary.failed == 1)
    #expect(summary.min_consistency_cosine == 0.42)
    #expect(summary.wall_ms == 2_500 && summary.footprint_delta_bytes == 1_000)

    let json = try #require(JSONSerialization.jsonObject(
        with: JSONEncoder().encode(summary)) as? [String: Any])
    #expect(Set(json.keys) == ["mode", "functions", "passed", "failed", "first_failure",
                               "min_consistency_cosine", "wall_ms", "footprint_delta_bytes"])

    let basic = try #require(JSONSerialization.jsonObject(
        with: JSONEncoder().encode(JinaVerificationSummary(mode: .basic))) as? [String: Any])
    #expect(basic.keys.sorted() == ["mode"] && basic["mode"] as? String == "basic")
}

@Test func verificationLogLinesNameTheFunctionAndReason() {
    let ok = check("text.bucket_64", cosine: 0.9999, reference: "bucket_32")
    #expect(JinaVerificationLog.line(ok, index: 2)
        == "verify #2 text.bucket_64 ok load=12.3ms run=1.5ms cosine=0.999900 vs bucket_32")
    let bad = check("video.f512", failure: "prediction failed: boom")
    #expect(JinaVerificationLog.line(bad, index: 24) == "verify #24 video.f512 FAILED: prediction failed: boom")
    let report = OmniSmallVerificationReport(
        functions: [ok], wallMilliseconds: 4_321, footprintBytesBefore: 0, footprintBytesAfter: 314_572_800)
    #expect(JinaVerificationLog.readyLine(report, modalities: ["text", "image"])
        == "ready: verified 1 functions in 4.3s (min consistency cosine 0.999900, footprint +300 MiB); modalities=text,image")
}

@Test func startupVerificationFlagDefaultsToFullAndRejectsOtherValues() throws {
    #expect(try Arguments.parse(["gloss-server"]).startupVerification == .full)
    #expect(try Arguments.parse(["gloss-server", "--startup-verification", "basic"]).startupVerification == .basic)
    #expect(try Arguments.parse(["gloss-server", "--startup-verification", "full"]).startupVerification == .full)
    #expect(throws: ArgumentError.self) {
        _ = try Arguments.parse(["gloss-server", "--startup-verification", "sometimes"])
    }
    #expect(throws: ArgumentError.self) {
        _ = try Arguments.parse(["gloss-server", "--startup-verification"])
    }
    #expect(Arguments.helpText.contains("--startup-verification full|basic"))
}

@Test func hardwareIdentityReportsChipAndOSVersion() {
    #expect(!HardwareIdentity.chip.isEmpty)
    #expect(HardwareIdentity.macOS.split(separator: ".").count >= 3, "major.minor.patch, then the build")
}

@Test func jinaDocsDescribeStartupMediaAndIsolationBehavior() throws {
    let context = DocsPage.Context(
        baseURL: "http://127.0.0.1:11435", modelID: "jinaai/jina-embeddings-v5-omni-small", dimensions: 1024,
        maxTokens: 32_768, compute: "auto", modalities: ["text", "image", "audio", "video"],
        maxBatch: 2_048, maxRequestTokens: 131_072, maxBodyMB: 64, maxTotalBodyMB: 256,
        maxQueueRequests: 128, maxQueueItems: 8_192, batchWindowMS: 2, keepWarmSeconds: 60,
        maxConnections: 256, ioTimeoutSeconds: 30, shutdownGraceSeconds: 15, accessLogMode: "errors",
        spaceID: testSpace, family: .jinaOmniSmall, matryoshka: [32, 64, 128, 256, 512, 1_024])
    let html = try #require(DocsPage.render(context))
    for phrase in [
        "--startup-verification full|basic", "startup_verification", "off the accelerator",
        "X-Glossematics-Video-Recipe", "480 samples", "4096 pixels", "gloss-media-*",
        "gloss_text_isolation_retries_total", "make bench",
    ] {
        #expect(html.contains(phrase), "docs-jina.html should mention \(phrase)")
    }
}
