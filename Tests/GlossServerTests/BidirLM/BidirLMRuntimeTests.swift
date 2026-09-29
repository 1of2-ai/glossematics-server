import Foundation
import Testing
@testable import gloss_server

private func fixtureURL() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/BidirLMOmni.dummy.bundle", isDirectory: true)
}

/// The fixture's 622 MB zero token table is written by `make fixture` and is not checked in.
private func fixtureReady() -> Bool {
    FileManager.default.fileExists(atPath: fixtureURL().appendingPathComponent("token_embeddings.f16").path)
}

// MARK: - planning

@Test func textPlanPacksShortSequencesAndChunksLongOnes() throws {
    #expect(try BidirLMTextPlan.make(lengths: [5, 10, 20]) == .init(chunk: 64, chunks: 1, keys: 64, packed: true))
    #expect(try BidirLMTextPlan.make(lengths: [64]) == .init(chunk: 64, chunks: 1, keys: 64, packed: true))
    #expect(try BidirLMTextPlan.make(lengths: [65]) == .init(chunk: 512, chunks: 1, keys: 512, packed: true))
    #expect(try BidirLMTextPlan.make(lengths: [100, 300, 112]) == .init(chunk: 512, chunks: 1, keys: 512, packed: true))
    #expect(try BidirLMTextPlan.make(lengths: [513]) == .init(chunk: 512, chunks: 2, keys: 1024, packed: false))
    #expect(try BidirLMTextPlan.make(lengths: [4097]) == .init(chunk: 512, chunks: 9, keys: 6144, packed: false))
    #expect(try BidirLMTextPlan.make(lengths: [32768]) == .init(chunk: 512, chunks: 64, keys: 32768, packed: false))
    #expect(throws: BidirLMTextPlan.Failure.tooLong(32769, 32768)) { try BidirLMTextPlan.make(lengths: [32769]) }
    #expect(throws: BidirLMTextPlan.Failure.longMustBeAlone) { try BidirLMTextPlan.make(lengths: [400, 200]) }
    #expect(throws: BidirLMTextPlan.Failure.empty) { try BidirLMTextPlan.make(lengths: [3, 0]) }
    #expect(throws: BidirLMTextPlan.Failure.tooMany(65, 64)) {
        try BidirLMTextPlan.make(lengths: Array(repeating: 1, count: 65))
    }
}

@Test func packPlannerFillsTokenAndRowBudgetsInOrder() {
    #expect(PackPlanner.select([10, 20, 30]) == [0, 1, 2])
    #expect(PackPlanner.select([300, 300, 200]) == [0, 2], "the second row does not fit; a later one does")
    #expect(PackPlanner.select(Array(repeating: 1, count: 70)).count == 64)
    #expect(PackPlanner.select([512, 1]) == [0])
    #expect(PackPlanner.select([600]).isEmpty, "long documents are never packed")
    #expect(PackPlanner.kind(totalTokens: 64) == "fused64")
    #expect(PackPlanner.kind(totalTokens: 65) == "pack512")
    #expect(PackPlanner.isShort(512) && !PackPlanner.isShort(513))
}

@Test func contractGeometryMatchesFunctionLists() {
    #expect(BidirLMContract.longKeys.first == 1024 && BidirLMContract.longKeys.last == 32_768)
    #expect(BidirLMContract.longKeys.allSatisfy { $0 % 512 == 0 })
    #expect(BidirLMContract.attentionFunctions.count == BidirLMContract.longKeys.count)
    #expect(BidirLMContract.stacks == 4)
    #expect(BidirLMContract.groupFunctions(0) == ["stack_c64", "stack_c512", "head_c512"] + (0...6).map { "mid_c512_l\($0)" })
    #expect(BidirLMContract.groupFunctions(3) == ["stack_c64", "stack_c512"] + (21...26).map { "mid_c512_l\($0)" } + ["tail_c512"])
    let long = (0..<4).flatMap { BidirLMContract.groupFunctions($0) }.filter { !$0.hasPrefix("stack_") }
    #expect(long.count == 29, "head + 27 mids + tail")
    #expect(BidirLMContract.towerFunctions(layers: 24).count == 25)
    #expect(BidirLMContract.spaceID.hasSuffix(":2048:w8a16:ane-chunked-mean-v1"))
}

// MARK: - media host preprocessing

@Test func smartResizeMatchesTheProcessor() throws {
    // (source h, w) -> processor output, from the pinned Qwen2-VL fast image processor.
    let cases: [((Int, Int), (Int, Int))] = [
        ((300, 420), (288, 416)), ((40, 30), (320, 224)), ((200, 1400), (192, 1408)),
        ((1500, 1100), (1184, 864)), ((256, 384), (256, 384)), ((48, 80), (224, 352)),
        ((80, 1000), (96, 928)),  // Python rounds half to even in the first step
    ]
    for ((h, w), expected) in cases {
        let got = try BidirLMMediaInputs.smartResize(height: h, width: w, factor: 32, minPixels: 65_536, maxPixels: 1_048_576)
        #expect(got == expected, "\(h)x\(w)")
    }
    #expect(throws: BidirLMMediaInputs.Failure.self) {
        try BidirLMMediaInputs.smartResize(height: 10, width: 4000, factor: 32, minPixels: 65_536, maxPixels: 1_048_576)
    }
}

@Test func bicubicResamplerKeepsIdentityAndIsSeparable() {
    let rgb = (0..<(8 * 8 * 3)).map { UInt8($0 % 256) }
    #expect(BicubicResampler.resize(rgb, height: 8, width: 8, toHeight: 8, toWidth: 8) == rgb)
    let flat = [UInt8](repeating: 77, count: 10 * 6 * 3)
    #expect(BicubicResampler.resize(flat, height: 10, width: 6, toHeight: 32, toWidth: 3).allSatisfy { $0 == 77 },
            "normalized weights preserve a constant image in both directions")
}

@Test func audioTokenCountMatchesTheConvolutionStack() {
    #expect(AudioGeometry.tokens(frames: 300) == 38)
    #expect(AudioGeometry.tokens(frames: 200) == 25)
    #expect(AudioGeometry.tokens(frames: 201) == 26)
    #expect(AudioGeometry.tokens(frames: 4500) == 563)
}

@Test func mropePositionsFollowGetRopeIndex() {
    // 3 text tokens, a 2x3 merged image, then 2 text tokens.
    let positions = MediaSequenceBuilder.mropePositions(count: 11, images: [(start: 3, h: 2, w: 3)])
    #expect(positions.count == 11)
    #expect(positions[0] == (0, 0, 0) && positions[2] == (2, 2, 2))
    #expect(positions[3] == (3, 3, 3) && positions[5] == (3, 3, 5) && positions[6] == (3, 4, 3) && positions[8] == (3, 4, 5))
    #expect(positions[9] == (6, 6, 6), "text resumes at image start + max(h, w)")
    #expect(positions[10] == (7, 7, 7))
}

@Test(.enabled(if: fixtureReady(), "run `make fixture` first"))
func messageTemplateExpandsPlaceholdersAndRejectsControlTokens() throws {
    let bundle = try BidirLMBundle.load(from: fixtureURL(), allowFixture: true)
    let tokenizer = try BidirLMTokenizer(folder: bundle.root.appendingPathComponent("tokenizer"),
                                         manifest: bundle.manifest, instances: 1)
    let tokens = BidirLMContract.mediaTokens
    let image = PreparedImage(pixels: [Float](repeating: 0, count: 16 * 16 * 1536), gridH: 16, gridW: 16)
    let audio = PreparedAudio(mel: [Float](repeating: 0, count: 128 * 300), frames: 300)
    let pending = try MediaSequenceBuilder.build(
        [.text("A photo: "), .media(.image(image)), .text(" and "), .media(.audio(audio))],
        tokenizer: tokenizer, tokens: tokens, kind: "message")
    let ids = pending.ids
    #expect(Array(ids.prefix(3)) == BidirLMContract.userPrefixIDs)
    #expect(Array(ids.suffix(2)) == BidirLMContract.userSuffixIDs)
    #expect(pending.spans.count == 2)
    #expect(ids.filter { $0 == tokens.imagePad }.count == 64 && ids.filter { $0 == tokens.audioPad }.count == 38)
    #expect(ids[pending.spans[0].start - 1] == tokens.visionStart && ids[pending.spans[0].start + 64] == tokens.visionEnd)
    #expect(ids[pending.spans[1].start - 1] == tokens.audioStart && ids[pending.spans[1].start + 38] == tokens.audioEnd)
    #expect(pending.positions?.count == ids.count, "an image switches the sequence to 3-D positions")
    #expect(throws: MediaSequenceBuilder.Failure.self) {
        try MediaSequenceBuilder.build([.text("<|image_pad|>"), .media(.image(image))], tokenizer: tokenizer,
                                       tokens: tokens, kind: "message")
    }
}

// MARK: - fixture bundle

@Test(.enabled(if: fixtureReady(), "run `make fixture` first"))
func fixtureValidatesOnlyWithAllowDummy() throws {
    #expect(throws: BidirLMBundle.Failure.self) { try BidirLMBundle.load(from: fixtureURL()) }
    let bundle = try BidirLMBundle.load(from: fixtureURL(), allowFixture: true)
    #expect(bundle.manifest.isFixture)
    #expect(bundle.manifest.vision != nil && bundle.manifest.audio != nil)
    #expect(bundle.artifactFingerprint.hasPrefix("sha256:"))
}

@Test(.enabled(if: fixtureReady(), "run `make fixture` first"))
func tamperedFixtureFailsIntegrity() throws {
    let copy = FileManager.default.temporaryDirectory.appendingPathComponent("bidirlm-tamper-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: copy) }
    try FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
    for item in try FileManager.default.contentsOfDirectory(atPath: fixtureURL().path) where item != "token_embeddings.f16" {
        try FileManager.default.copyItem(at: fixtureURL().appendingPathComponent(item), to: copy.appendingPathComponent(item))
    }
    // A table of the right size but wrong bytes must fail the checksum, not load.
    FileManager.default.createFile(atPath: copy.appendingPathComponent("token_embeddings.f16").path,
                                   contents: Data(repeating: 1, count: 151_936 * 2048 * 2))
    #expect(throws: BidirLMBundle.Failure.self) { try BidirLMBundle.load(from: copy, allowFixture: true) }
}

@Test(.enabled(if: fixtureReady(), "run `make fixture` first"))
func fixtureRunsTextAndMediaThroughCoreML() async throws {
    let bundle = try BidirLMBundle.load(from: fixtureURL(), allowFixture: true)
    let tokenizer = try BidirLMTokenizer(folder: bundle.root.appendingPathComponent("tokenizer"),
                                         manifest: bundle.manifest, instances: 1)
    #expect(tokenizer.templated("cat") == [151_644, 872, 198, 4616, 151_645, 198])
    for mode in [ComputeMode.cpu, .gpu] {
        let backend = BidirLMBackend(bundle: bundle, mode: mode, tokenizer: tokenizer)
        try await backend.prepare(verifyPlacement: true)
        let short = try await backend.embed(tokenRows: [tokenizer.templated("cat"), tokenizer.templated("dog")])
        #expect(short.count == 2 && short.allSatisfy { $0.count == 2048 && abs($0[0] - 1) < 1e-3 })
        let medium = try await backend.embed(tokenRows: [Array(repeating: 11, count: 300)])
        #expect(medium.count == 1)
        let run = try await backend.startLong(tokens: Array(repeating: 11, count: 1500))
        var vector: [Float]?
        var steps = 0
        while vector == nil {
            vector = try await backend.step(run)
            steps += 1
        }
        #expect(steps == 3 * (1 + 28), "one step per chunk operation: 3 heads, then 28 layers x 3 chunks")
        #expect(vector?.count == 2048)

        guard mode != .cpu else {
            #expect(!backend.hasVision && !backend.hasAudio, "media towers are not served in cpu mode")
            continue
        }
        let image = PreparedImage(pixels: [Float](repeating: 0.1, count: 40 * 32 * 1536), gridH: 40, gridW: 32)
        let vision = try await backend.startTower(.image(image))
        var features: BidirLMMediaEncoder.Features?
        while features == nil { features = try await backend.step(vision) }
        #expect(features?.count == 320 && features?.deepstack.count == 2)

        let clip = PreparedAudio(mel: [Float](repeating: 0.2, count: 128 * 4500), frames: 4500)
        let audio = try await backend.startTower(.audio(clip))
        features = nil
        while features == nil { features = try await backend.step(audio) }
        #expect(features?.count == 563 && features?.deepstack.isEmpty == true)
    }
}

@Test(.enabled(if: fixtureReady(), "run `make fixture` first"))
func residencyEvictsLeastRecentlyUsedSetsUnderTheBudget() throws {
    let bundle = try BidirLMBundle.load(from: fixtureURL(), allowFixture: true)
    let m = bundle.manifest
    let attention = { (keys: Int) -> [ProgramResidency.Function] in
        [.init(path: m.text.attentionModels[BidirLMContract.attentionName(chunk: 512, keys: keys, packed: false)]!, name: nil)]
    }
    let residency = ProgramResidency(root: bundle.root, mode: .ane, budget: 3, verifyNeuralEngine: false)
    _ = try residency.acquire("a", attention(1024), pinned: true)
    _ = try residency.acquire("b", attention(2048))
    _ = try residency.acquire("c", attention(4096))
    #expect(residency.residentPrograms == 3)
    _ = try residency.acquire("b", attention(2048))        // refresh b; c is now least recent
    _ = try residency.acquire("d", attention(6144))
    #expect(residency.residentSets == ["a", "b", "d"])
    #expect(residency.evictions == 1)
    #expect(throws: ProgramResidency.Failure.self) {
        _ = try residency.acquire("big", attention(8192) + attention(10240) + attention(12288))
    }
}
