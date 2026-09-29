import Testing
@testable import gloss_server

/// Unit tests for the index-grouped batch plan — the pure grouping logic behind bulk text
/// embedding. The plan must cover every row exactly once, put batch calls only on rows that fit
/// their bucket, and leave order restoration to the caller's index assembly.
private let ladder = [
    (size: 64, bucket: 32), (size: 32, bucket: 64), (size: 16, bucket: 128),
    (size: 8, bucket: 256), (size: 4, bucket: 512),
]
private let nativeBuckets = [32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768]

@Test func planKeepsSparseInterleavedLengthsOnSingleRowFunctions() {
    // A sparse native batch would execute mostly padding, even if rows can be grouped across
    // interleaved positions. Use the smallest fitting single-row function for each real row.
    let counts = [30, 400, 28, 380, 31]
    let plan = GlossTextEmbedder.embeddingPlan(
        tokenCounts: counts, buckets: nativeBuckets, batchPairs: ladder)
    #expect(plan == [
        .single(index: 0, bucket: 32),
        .single(index: 2, bucket: 32),
        .single(index: 4, bucket: 32),
        .single(index: 1, bucket: 512),
        .single(index: 3, bucket: 512),
    ])
}

@Test func planChunksLargeGroupsAtTheLadderSize() {
    let counts = [Int](repeating: 30, count: 70)
    let plan = GlossTextEmbedder.embeddingPlan(
        tokenCounts: counts, buckets: nativeBuckets, batchPairs: ladder)
    #expect(plan.count == 7)
    #expect(plan[0] == .batch(rows: Array(0..<64), size: 64, bucket: 32))
    #expect(Array(plan.dropFirst()) == (64..<70).map { .single(index: $0, bucket: 32) })
}

@Test func planSendsRowsOverEveryBatchBucketToSingleBuckets() {
    let counts = [512, 513, 600, 5_000]
    let plan = GlossTextEmbedder.embeddingPlan(
        tokenCounts: counts, buckets: nativeBuckets, batchPairs: ladder)
    // A lone 512-token row fits the batch function but is faster on its single-row bucket.
    #expect(plan.contains(.single(index: 0, bucket: 512)))
    #expect(plan.contains(.single(index: 1, bucket: 1_024)))
    #expect(plan.contains(.single(index: 2, bucket: 1_024)))
    #expect(plan.contains(.single(index: 3, bucket: 8_192)))
}

@Test func planFallsBackToAllSinglesWithoutBatchFunctions() {
    let counts = [30, 300, 3_000]
    let plan = GlossTextEmbedder.embeddingPlan(
        tokenCounts: counts, buckets: nativeBuckets, batchPairs: [])
    #expect(plan == [
        .single(index: 0, bucket: 32),
        .single(index: 1, bucket: 512),
        .single(index: 2, bucket: 4_096),
    ])
}

@Test func planCoversEveryRowExactlyOnceAcrossShapes() {
    // Deterministic LCG sweep: whatever the shape, the plan partitions the rows.
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next(_ bound: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((state >> 33) % UInt64(bound))
    }
    for trial in 0..<200 {
        let count = 2 + next(60)
        let counts = (0..<count).map { _ in 1 + next(700) }
        let plan = GlossTextEmbedder.embeddingPlan(
            tokenCounts: counts, buckets: nativeBuckets, batchPairs: ladder)
        var seen = Set<Int>()
        for step in plan {
            switch step {
            case let .single(index, bucket):
                #expect(index >= 0 && index < count, "trial \(trial): bad index")
                #expect(bucket >= counts[index], "trial \(trial): single bucket too small")
                #expect(seen.insert(index).inserted, "trial \(trial): row \(index) planned twice")
            case let .batch(rows, size, bucket):
                #expect(!rows.isEmpty, "trial \(trial): empty batch step")
                #expect(rows == rows.sorted(), "trial \(trial): batch rows must ascend")
                #expect(rows.count <= size, "trial \(trial): batch step exceeds function size")
                for row in rows {
                    #expect(counts[row] <= bucket, "trial \(trial): row \(row) overflows bucket")
                    #expect(seen.insert(row).inserted, "trial \(trial): row \(row) planned twice")
                }
            }
        }
        #expect(seen.count == count, "trial \(trial): plan missed rows")
    }
}
