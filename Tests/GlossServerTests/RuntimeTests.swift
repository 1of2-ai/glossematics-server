import XCTest
@testable import gloss_server

final class RuntimeTests: XCTestCase {
    func testAdmissionIsWholeRequestAndBounded() async {
        let gate = AdmissionGate(maxRequests: 2, maxItems: 5)
        let first = await gate.tryAcquire(items: 3)
        XCTAssertTrue(first)
        let rejectedItems = await gate.tryAcquire(items: 3)
        XCTAssertFalse(rejectedItems)
        let second = await gate.tryAcquire(items: 2)
        XCTAssertTrue(second)
        let rejectedRequests = await gate.tryAcquire(items: 1)
        XCTAssertFalse(rejectedRequests)
        await gate.release(items: 3)
        let afterRelease = await gate.tryAcquire(items: 1)
        XCTAssertTrue(afterRelease)
    }

    func testReadinessFailureIsSticky() async {
        let state = RuntimeState()
        await state.failStartup("broken")
        await state.markReady()
        let snapshot = await state.snapshot()
        XCTAssertFalse(snapshot.ready)
        XCTAssertEqual(snapshot.phase, "failed")
        XCTAssertEqual(snapshot.startupError, "broken")
    }

    func testKeepWarmFailureDoesNotChangeReadyState() async {
        let state = RuntimeState()
        await state.markReady()
        await state.recordKeepWarmFailure("transient")
        let snapshot = await state.snapshot()
        XCTAssertTrue(snapshot.ready)
        XCTAssertEqual(snapshot.lastKeepWarmError, "transient")
        XCTAssertEqual(snapshot.keepWarmFailures, 1)
    }

    func testWaveMetricsAreKeyedByExecutionKind() async {
        let metrics = ServerMetrics()
        await metrics.recordWave(kind: "fused64", rows: 3, tokens: 40, requests: 2)
        await metrics.recordWave(kind: "pack512", rows: 10, tokens: 480, requests: 1)
        await metrics.recordLong(tokens: 4000, steps: 448)
        let snapshot = await metrics.snapshot()
        XCTAssertEqual(snapshot.wavesByKind["fused64"], 1)
        XCTAssertEqual(snapshot.rowsByKind["pack512"], 10)
        XCTAssertEqual(snapshot.tokensByKind["fused64"], 40)
        XCTAssertEqual(snapshot.coalescedWaves, 1)
        XCTAssertEqual(snapshot.longDocuments, 1)
        XCTAssertEqual(snapshot.longSteps, 448)
    }
}
