import XCTest
@testable import gloss_server

final class JinaBatchPlannerTests: XCTestCase {
    func testNativeCapacityMatchesCompiledLadder() {
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 1), 64)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 32), 64)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 33), 32)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 64), 32)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 65), 16)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 129), 8)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 257), 4)
        XCTAssertEqual(JinaBatchPlanner.nativeCapacity(for: 513), 1)
    }



    func testRoundRobinOrderAdvancesPastPreviousWinner() {
        XCTAssertEqual(JinaBatchPlanner.rotatedOrder([1, 2, 3, 4], after: 2), [3, 4, 1, 2])
        XCTAssertEqual(JinaBatchPlanner.rotatedOrder([1, 2, 3, 4], after: 4), [1, 2, 3, 4])
    }

    func testSparseBatchThresholdsMatchMeasuredNativeCrossovers() {
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 64, bucket: 32), 24)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 32, bucket: 64), 24)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 16, bucket: 128), 16)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 8, bucket: 256), 6)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 4, bucket: 512), 3)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 4, bucket: 128), 4)
        XCTAssertEqual(NativeTextBatchPolicy.minimumRows(for: 7, bucket: 64), 7)
    }
}
