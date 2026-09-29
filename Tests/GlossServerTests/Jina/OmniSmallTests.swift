import CryptoKit
import Foundation
import Testing
@_spi(Server) @testable import gloss_server

private let testSpace = "glossematics:omni-small:sha256:" + String(repeating: "a", count: 64)
private let otherTestSpace = "glossematics:omni-small:sha256:" + String(repeating: "b", count: 64)
private let testArtifact = "sha256:" + String(repeating: "c", count: 64)
private let otherTestArtifact = "sha256:" + String(repeating: "d", count: 64)

private actor FakeOmniSmallBackend: OmniSmallBackend {
    private var textBatchSizes = [Int]()
    private var roles = [String]()
    private var prepareCalls = 0
    private var textEncodeCalls = 0
    private var mediaCalls = 0
    /// Trigger word found by the most recent prepare, consumed by the next encode.
    private var pendingEncodeTrigger: String?

    func prepareTexts(
        _ texts: [String],
        role: OmniSmallRole
    ) async throws -> [ValidatedText] {
        prepareCalls += 1
        roles.append(contentsOf: texts.map { _ in role == .query ? "query" : "document" })
        if let index = texts.firstIndex(of: "over-limit") {
            throw OmniSmallBackendError.item(
                index: index,
                reason: "text has 32769 total tokens including conditioning; maximum is 32768")
        }
        pendingEncodeTrigger = ["generic-error", "slow", "ignores-cancel"]
            .first { texts.contains($0) }
        return try texts.map { text in
            try ValidatedText(tokenIDs: [Int32(Int(text) ?? 0)])
        }
    }

    func embedTexts(
        _ rows: [ValidatedText],
        dimensions: OmniSmall.Dimensions
    ) async throws -> [[Float]] {
        textEncodeCalls += 1
        textBatchSizes.append(rows.count)
        if pendingEncodeTrigger == "generic-error" {
            pendingEncodeTrigger = nil
            throw OmniSmallBackendError.failure("native prediction failed")
        }
        if pendingEncodeTrigger == "slow" {
            try await Task.sleep(for: .seconds(30))
        }
        if pendingEncodeTrigger == "ignores-cancel" {
            try? await Task.sleep(for: .milliseconds(100))
        }
        pendingEncodeTrigger = nil
        return rows.map {
            $0.tokenIDs[0] == 255
                ? [Float](repeating: 0, count: dimensions.rawValue)
                : vector(key: Int($0.tokenIDs[0]), dimensions: dimensions.rawValue)
        }
    }

    func embedMedia(
        _ input: OmniSmall.Input,
        role: OmniSmallRole,
        dimensions: OmniSmall.Dimensions
    ) async throws -> [Float] {
        mediaCalls += 1
        roles.append(role == .query ? "query" : "document")
        switch input {
        case .text:
            throw OmniSmallBackendError.failure("text reached media backend")
        case let .image(url), let .audio(url), let .video(url):
            return vector(
                key: Int(url.deletingPathExtension().lastPathComponent) ?? 0,
                dimensions: dimensions.rawValue)
        case let .imageData(data), let .audioData(data), let .videoData(data):
            if data.first == 255 {
                return [Float](repeating: 0, count: dimensions.rawValue)
            }
            return vector(
                key: Int(data.first ?? 0),
                dimensions: dimensions.rawValue)
        }
    }

    func snapshot() -> (
        batchSizes: [Int], roles: [String],
        prepareCalls: Int, textEncodeCalls: Int, mediaCalls: Int
    ) {
        (textBatchSizes, roles, prepareCalls, textEncodeCalls, mediaCalls)
    }

    private func vector(key: Int, dimensions: Int) -> [Float] {
        var result = [Float](repeating: 0, count: dimensions)
        result[key % dimensions] = 1
        return result
    }
}

@Test func omniSmallMalformedBackendVectorsAreInferenceFailures() async throws {
    let model = OmniSmall(
        backend: FakeOmniSmallBackend(),
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    for input in [OmniSmall.Input.text("255"), .imageData(Data([255]))] {
        do {
            _ = try await model.embedDocuments([input])
            Issue.record("an invalid backend vector must not be blamed on input")
        } catch let error as OmniSmallError {
            guard case let .inferenceFailed(reason) = error else {
                Issue.record("unexpected error: \(error)")
                continue
            }
            #expect(reason.contains("output at index 0"))
        }
    }
    do {
        _ = try await model.embedConditionedTokenRows([[255]], dimensions: .d32)
        Issue.record("an invalid conditioned text vector must be a server failure")
    } catch let error as OmniSmallError {
        guard case let .inferenceFailed(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("text output at index 0"))
    }
}

@Test func omniSmallTypedEmbeddingsAndSemanticCompatibility() async throws {
    let backend = FakeOmniSmallBackend()
    let model = OmniSmall(
        backend: backend,
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    let query = try await model.embedQuery(.text("3"))
    let document = try await model.embedDocument(.text("3"))

    #expect(query.values.count == 32)
    #expect(query.dimensions == .d32)
    #expect(query.space == testSpace)
    #expect(try query.similarity(to: document) == 1)

    let recompiledModel = OmniSmall(
        backend: FakeOmniSmallBackend(),
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: otherTestArtifact)
    let recompiledDocument = try await recompiledModel.embedDocument(.text("3"))
    #expect(try query.similarity(to: recompiledDocument) == 1,
            "artifact provenance must not replace semantic compatibility")

    let incompatibleModel = OmniSmall(
        backend: FakeOmniSmallBackend(),
        dimensions: .d32,
        space: otherTestSpace,
        artifactFingerprint: testArtifact)
    let incompatibleDocument = try await incompatibleModel.embedDocument(.text("3"))
    #expect(throws: OmniSmallError.self) {
        _ = try query.similarity(to: incompatibleDocument)
    }

    let snapshot = await backend.snapshot()
    #expect(snapshot.roles == ["query", "document"])
}

@Test func omniSmallOneBackendServesMultipleMatryoshkaDimensions() async throws {
    let backend = FakeOmniSmallBackend()
    let model = OmniSmall(
        backend: backend,
        dimensions: .d32,
        spaces: [.d32: testSpace, .d64: otherTestSpace],
        artifactFingerprint: testArtifact)

    let small = try await model.embedDocument(.text("3"), dimensions: .d32)
    let larger = try await model.embedDocument(.text("3"), dimensions: .d64)

    #expect(small.values.count == 32)
    #expect(larger.values.count == 64)
    #expect(small.space == testSpace)
    #expect(larger.space == otherTestSpace)
    #expect(model.space(for: .d64) == otherTestSpace)

    let snapshot = await backend.snapshot()
    #expect(snapshot.textEncodeCalls == 2)
}

@Test func documentEmbeddingCodableValidatesShapeNormAndIdentity() async throws {
    let model = OmniSmall(
        backend: FakeOmniSmallBackend(),
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    let document = try await model.embedDocument(.text("5"))
    let encoded = try JSONEncoder().encode(document)
    let decoded = try JSONDecoder().decode(DocumentEmbedding.self, from: encoded)
    #expect(decoded.values == document.values)
    #expect(decoded.dimensions == .d32)
    #expect(decoded.space == testSpace)

    var object = try #require(
        JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    object["values"] = [Double](repeating: 0, count: 32)
    let zeroVector = try JSONSerialization.data(withJSONObject: object)
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(DocumentEmbedding.self, from: zeroVector)
    }

    object["values"] = [Double](repeating: 1, count: 31)
    let wrongShape = try JSONSerialization.data(withJSONObject: object)
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(DocumentEmbedding.self, from: wrongShape)
    }

    object["values"] = document.values
    object["space"] = "unversioned"
    let badIdentity = try JSONSerialization.data(withJSONObject: object)
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(DocumentEmbedding.self, from: badIdentity)
    }

    object["space"] = "glossematics:omni-small:sha256:"
        + String(repeating: "١", count: 64)
    let nonASCIIIdentity = try JSONSerialization.data(withJSONObject: object)
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(DocumentEmbedding.self, from: nonASCIIIdentity)
    }

    #expect(throws: OmniSmallError.self) {
        _ = try DocumentEmbedding(
            values: [Float.nan] + [Float](repeating: 0, count: 31),
            dimensions: .d32,
            space: testSpace,
            artifactFingerprint: testArtifact)
    }
}

@Test func omniSmallBatchPreservesOrderBoundsWorkAndReportsOriginalIndex() async throws {
    let backend = FakeOmniSmallBackend()
    let model = OmniSmall(
        backend: backend,
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    let inputs = (0..<130).map { OmniSmall.Input.text(String($0)) }
    let documents = try await model.embedDocuments(inputs)
    #expect(documents.count == inputs.count)
    for index in documents.indices {
        #expect(documents[index].values[index % 32] == 1)
    }
    let snapshot = await backend.snapshot()
    #expect(snapshot.batchSizes == [64, 64, 2])

    let mediaDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("omni-small-media-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: mediaDirectory, withIntermediateDirectories: true)
    let image = mediaDirectory.appendingPathComponent("5.png")
    let audio = mediaDirectory.appendingPathComponent("6.wav")
    try Data().write(to: image)
    try Data().write(to: audio)
    defer {
        try? FileManager.default.removeItem(at: mediaDirectory)
    }
    let mixed = try await model.embedDocuments([
        .text("0"),
        .image(image),
        .text("1"),
        .audio(audio),
        .text("2"),
    ])
    #expect(mixed.count == 5)
    #expect(mixed[0].values[0] == 1)
    #expect(mixed[1].values[5] == 1)
    #expect(mixed[2].values[1] == 1)
    #expect(mixed[3].values[6] == 1)
    #expect(mixed[4].values[2] == 1)

    do {
        _ = try await model.embedDocuments([
            .text("0"),
            .text("1"),
            .image(URL(fileURLWithPath: "/definitely/missing.png")),
        ])
        Issue.record("missing image should fail")
    } catch let error as OmniSmallError {
        guard case let .invalidBatchInput(index, _) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(index == 2)
    }

    do {
        _ = try await model.embedDocuments([
            .text("0"),
            .text("over-limit"),
            .text("2"),
        ])
        Issue.record("over-limit text should fail")
    } catch let error as OmniSmallError {
        guard case let .invalidBatchInput(index, reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(index == 1)
        #expect(reason.contains("32769"))
    }

    do {
        _ = try await model.embedDocuments([
            .text("0"),
            .text("generic-error"),
            .text("2"),
        ])
        Issue.record("native backend failure should not blame an input")
    } catch let error as OmniSmallError {
        guard case let .inferenceFailed(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("chunk starting at 0"))
    }
}

@Test func omniSmallBatchFailsBeforeAnyInferenceWhenAnyInputIsInvalid() async throws {
    let backend = FakeOmniSmallBackend()
    let model = OmniSmall(
        backend: backend,
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    // The over-limit row sits past the first 64-row chunk: preparation must reject it before any
    // chunk encodes, so a 100-document batch never wastes work on a doomed submission.
    var inputs = (0..<80).map { OmniSmall.Input.text(String($0)) }
    inputs[70] = .text("over-limit")
    do {
        _ = try await model.embedDocuments(inputs)
        Issue.record("over-limit text past the first chunk should fail without inference")
    } catch let error as OmniSmallError {
        guard case let .invalidBatchInput(index, reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(index == 70)
        #expect(reason.contains("32769"))
    }
    let snapshot = await backend.snapshot()
    #expect(snapshot.textEncodeCalls == 0, "no encode call may precede full validation")
    #expect(snapshot.mediaCalls == 0, "no media call may precede full validation")
}

@Test func omniSmallEmptyBatchReturnsEmptyWithoutBackendWork() async throws {
    let backend = FakeOmniSmallBackend()
    let model = OmniSmall(
        backend: backend,
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    let out = try await model.embedDocuments([])
    #expect(out.isEmpty)
    let snapshot = await backend.snapshot()
    #expect(snapshot.prepareCalls == 0)
    #expect(snapshot.textEncodeCalls == 0)
    #expect(snapshot.mediaCalls == 0)
}

@Test func omniSmallRejectsEmptyTextAndPropagatesCancellation() async throws {
    let model = OmniSmall(
        backend: FakeOmniSmallBackend(),
        dimensions: .d32,
        space: testSpace,
        artifactFingerprint: testArtifact)
    do {
        _ = try await model.embedQuery(.text(" \n\t "))
        Issue.record("empty text should fail")
    } catch is OmniSmallError {
        // Expected.
    } catch {
        Issue.record("expected OmniSmallError, got \(error)")
    }

    let task = Task {
        try await model.embedDocument(.text("ignores-cancel"))
    }
    try await Task.sleep(for: .milliseconds(10))
    task.cancel()
    do {
        _ = try await task.value
        Issue.record("cancelled inference should throw")
    } catch is CancellationError {
        // Expected.
    } catch {
        Issue.record("expected CancellationError, got \(error)")
    }

    let emptyTask = Task {
        while !Task.isCancelled { await Task.yield() }
        return try await model.embedDocuments([])
    }
    emptyTask.cancel()
    do {
        _ = try await emptyTask.value
        Issue.record("cancelled empty batch should throw")
    } catch is CancellationError {
        // Expected.
    } catch {
        Issue.record("expected CancellationError, got \(error)")
    }
}

@Test func omniSmallAudioLimitsRejectBeforeUnboundedDecode() throws {
    try OmniSmallInputLimits.validateEstimatedAudio(
        frameCount: 1_440_000,
        sampleRate: 48_000,
        channelCount: 2)
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: 1_440_001,
            sampleRate: 48_000,
            channelCount: 2)
    }
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: 0,
            sampleRate: 48_000,
            channelCount: 2)
    }
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: Int64(UInt32.max) + 1,
            sampleRate: 48_000,
            channelCount: 2)
    }
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: 1,
            sampleRate: .leastNonzeroMagnitude,
            channelCount: 2)
    }
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: 5_760_000,
            sampleRate: 192_000,
            channelCount: 8)
    }
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: 1_440_000,
            sampleRate: 48_000,
            channelCount: 0)
    }
    try OmniSmallInputLimits.validateDecodedAudio(sampleCount: 480_000)
    #expect(throws: OmniSmallBackendError.self) {
        try OmniSmallInputLimits.validateDecodedAudio(sampleCount: 480_001)
    }
}

/// One or two mel frames pool to ZERO audio tokens in the reference, so the floor is three frames
/// (480 samples, 30 ms). 160 samples (one frame) used to be accepted.
@Test func omniSmallAudioMinimumIsThreeMelFrames() throws {
    #expect(OmniSmallInputLimits.minimumAudioSamples == 480)
    for samples in [0, 1, 159, 160, 320, 479] {
        do {
            try OmniSmallInputLimits.validateDecodedAudio(sampleCount: samples)
            Issue.record("\(samples) samples must be rejected")
        } catch let error as OmniSmallBackendError {
            guard case let .invalidInput(reason) = error else {
                Issue.record("expected invalidInput, got \(error)")
                continue
            }
            #expect(reason.contains("480") && reason.contains("30 ms"), Comment(rawValue: reason))
        }
    }
    for samples in [480, 481, 639, 640, 16_000, 480_000] {
        try OmniSmallInputLimits.validateDecodedAudio(sampleCount: samples)
    }
    // Every accepted length yields at least one pooled token.
    for samples in [480, 481, 639, 640] {
        #expect(AudioMasks.pooledTokenCount(frames: samples / 160) >= 1)
    }
}

@Test func omniSmallLoadValidatesNativeCapabilitiesAndChecksums() async throws {
    do {
        _ = try await OmniSmall.load(
            from: try #require(URL(string: "https://example.com/model.bundle")))
        Issue.record("remote bundle URL should fail before any read")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("local file URL"))
    }

    let fixture = try OmniSmallFixture()
    defer { fixture.remove() }

    let model = try await OmniSmall.load(from: fixture.root, dimensions: .d256)
    #expect(model.dimensions == .d256)
    // Every space ID is pinned here (an independent recomputation of the semantic string over
    // revision 87f7a45d... gives these values). The formula matches the SDK daemon's, but the
    // source revision moved from 41a20a1e to 87f7a45d, so none of these equal the IDs of earlier
    // builds. A change here is a vector-space migration. The video file recipe is versioned
    // separately (`OmniSmall.videoRecipe`).
    #expect(model.space == "glossematics:omni-small:sha256:5b91ad2cc1420b56f69b7af8abf5197e8a5d4268bb897d325caf9d5e42d9b226")
    let pinnedSpaces: [OmniSmall.Dimensions: String] = [
        .d32: "glossematics:omni-small:sha256:036037bf0bd73c6b278d8268c8c34daebeeed1d7f8b024effddd7b0d5ee5999a",
        .d64: "glossematics:omni-small:sha256:2987e04de5354fe38307ff4c9d43dcbaa1d11b3a5ebb38b23c7b42fbc490ba74",
        .d128: "glossematics:omni-small:sha256:b7ae7daeee531a23ccc091d83020a5031580c420fce36e7db6978a7a7a0cd2df",
        .d256: "glossematics:omni-small:sha256:5b91ad2cc1420b56f69b7af8abf5197e8a5d4268bb897d325caf9d5e42d9b226",
        .d512: "glossematics:omni-small:sha256:ce0f5aa8164f1b9579e7fd95ad52cdfbd4f48c8e86fd5c4d0734d82c88a7b4fd",
        .d1024: "glossematics:omni-small:sha256:40ef09cf246a8c05eca644978333c7bb63aeec83cb7dd9a01d85be334c0f00f9",
    ]
    for (dimensions, space) in pinnedSpaces {
        #expect(model.space(for: dimensions) == space)
    }

    let alias = FileManager.default.temporaryDirectory
        .appendingPathComponent("omni-small-alias-\(UUID().uuidString).bundle")
    try FileManager.default.createSymbolicLink(
        at: alias,
        withDestinationURL: fixture.root)
    defer { try? FileManager.default.removeItem(at: alias) }
    let aliasedModel = try await OmniSmall.load(from: alias, dimensions: .d64)
    #expect(aliasedModel.dimensions == .d64)

    let hiddenAlias = try OmniSmallFixture()
    defer { hiddenAlias.remove() }
    try FileManager.default.createSymbolicLink(
        at: hiddenAlias.root
            .appendingPathComponent("text.mlmodelc")
            .appendingPathComponent(".DS_Store"),
        withDestinationURL: hiddenAlias.root.appendingPathComponent("manifest.json"))
    do {
        _ = try await OmniSmall.load(from: hiddenAlias.root)
        Issue.record("a hidden symlink in an artifact must not be ignored")
    } catch let error as OmniSmallError {
        guard case let .artifactIntegrity(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("symbolic links"))
    }

    let overlapping = try OmniSmallFixture()
    defer { overlapping.remove() }
    for name in ["meta.json", "pos_embed_table.f32", "rope_inv_freq.f32"] {
        try FileManager.default.copyItem(
            at: overlapping.root.appendingPathComponent("vision").appendingPathComponent(name),
            to: overlapping.root.appendingPathComponent("tokenizer").appendingPathComponent(name))
    }
    var overlappingManifest = try GlossModelBundle(url: overlapping.root).manifest
    overlappingManifest.image?.resources = "tokenizer"
    try JSONEncoder().encode(overlappingManifest).write(
        to: overlapping.root.appendingPathComponent("manifest.json"))
    do {
        _ = try await OmniSmall.load(from: overlapping.root)
        Issue.record("overlapping artifact roots must not hide duplicate coverage")
    } catch let error as OmniSmallError {
        guard case let .artifactIntegrity(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("multiple roots"))
    }

    let nestedAlias = try OmniSmallFixture()
    defer { nestedAlias.remove() }
    try FileManager.default.createSymbolicLink(
        at: nestedAlias.root.appendingPathComponent("linked"),
        withDestinationURL: nestedAlias.root)
    var linkedManifest = try GlossModelBundle(url: nestedAlias.root).manifest
    linkedManifest.text?.tokenizer = "linked/tokenizer"
    try JSONEncoder().encode(linkedManifest).write(
        to: nestedAlias.root.appendingPathComponent("manifest.json"))
    do {
        _ = try await OmniSmall.load(from: nestedAlias.root)
        Issue.record("a symlink in an artifact parent path must not be followed")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("symlink"))
    }

    let missingAsset = try OmniSmallFixture()
    defer { missingAsset.remove() }
    try missingAsset.removeTokenizerConfiguration()
    do {
        _ = try await OmniSmall.load(from: missingAsset.root)
        Issue.record("missing tokenizer configuration should fail before checksums")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("tokenizer_config.json"))
    }

    let badSignature = try OmniSmallFixture()
    defer { badSignature.remove() }
    try badSignature.corruptTextInputShape()
    do {
        _ = try await OmniSmall.load(from: badSignature.root)
        Issue.record("mismatched compiled input signature should fail")
    } catch let error as OmniSmallError {
        guard case let .unsupportedCapability(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("bucket_32.input_ids"))
    }

    let extraInput = try OmniSmallFixture()
    defer { extraInput.remove() }
    try extraInput.addUnexpectedTextInput()
    do {
        _ = try await OmniSmall.load(from: extraInput.root)
        Issue.record("a compiled function with an extra required input must fail validation")
    } catch let error as OmniSmallError {
        guard case let .unsupportedCapability(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("exactly 3 inputs"))
    }

    try Data("tampered".utf8).write(
        to: fixture.root
            .appendingPathComponent("tokenizer")
            .appendingPathComponent("tokenizer.json"))
    do {
        _ = try await OmniSmall.load(from: fixture.root)
        Issue.record("tampered artifact should fail checksum validation")
    } catch let error as OmniSmallError {
        guard case let .artifactIntegrity(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("checksum mismatch"))
    }
}

@Test func omniSmallRefusesBundlesPinnedToAnyOtherSourceRevision() async throws {
    let fixture = try OmniSmallFixture()
    defer { fixture.remove() }
    let manifestURL = fixture.root.appendingPathComponent("manifest.json")
    var manifest = try GlossModelBundle(url: fixture.root).manifest
    #expect(manifest.source?.revision == "87f7a45d1ae0265843f8569c47fb53847cb193c3")

    // The pre-re-pin revision no longer resolves on the Hub and is a different space: one
    // identity per binary, so it must be refused with an actionable message, not accepted.
    manifest.source?.revision = "41a20a1e1f56dad91e3a55d52ac6dc13007d67a5"
    try JSONEncoder().encode(manifest).write(to: manifestURL)
    do {
        _ = try await OmniSmall.load(from: fixture.root)
        Issue.record("a bundle pinned to the pre-re-pin revision must be refused")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("87f7a45d1ae0265843f8569c47fb53847cb193c3"))
        #expect(reason.contains("41a20a1e1f56dad91e3a55d52ac6dc13007d67a5"))
        #expect(reason.contains("repin_bundle_source.py"))
    }

    manifest.source = nil
    try JSONEncoder().encode(manifest).write(to: manifestURL)
    do {
        _ = try await OmniSmall.load(from: fixture.root)
        Issue.record("a bundle without a source pin must be refused")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("no source"))
    }
}

@Test func omniSmallRejectsBundlesWithoutThePinnedVideoTowerAndTokens() async throws {
    let fixture = try OmniSmallFixture()
    defer { fixture.remove() }
    let manifestURL = fixture.root.appendingPathComponent("manifest.json")
    let original = try GlossModelBundle(url: fixture.root).manifest

    var missingTower = original
    missingTower.video = nil
    try JSONEncoder().encode(missingTower).write(to: manifestURL)
    do {
        _ = try await OmniSmall.load(from: fixture.root)
        Issue.record("a bundle without the video tower must not be ready")
    } catch let error as OmniSmallError {
        guard case let .unsupportedCapability(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("video"))
    }

    var missingTokens = original
    missingTokens.tokens.video = nil
    try JSONEncoder().encode(missingTokens).write(to: manifestURL)
    do {
        _ = try await OmniSmall.load(from: fixture.root)
        Issue.record("a bundle without video conditioning tokens must not be ready")
    } catch let error as OmniSmallError {
        guard case let .invalidBundle(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("video"))
    }
}

@Test(.enabled(if: FileManager.default.fileExists(atPath: repositoryRoot().appendingPathComponent("artifacts/JinaV5OmniSmall.bundle/manifest.json").path), "Requires legacy converted bundle"))
func omniSmallRejectsLegacy2048BundleRatherThanDowngrading() async throws {
    let root = repositoryRoot()
    let legacy = root.appendingPathComponent(
        "artifacts/JinaV5OmniSmall.bundle")
    guard FileManager.default.fileExists(
        atPath: legacy.appendingPathComponent("manifest.json").path) else {
        return
    }
    do {
        _ = try await OmniSmall.load(from: legacy)
        Issue.record("legacy 2048-token bundle should fail closed")
    } catch let error as OmniSmallError {
        guard case let .unsupportedCapability(reason) = error else {
            Issue.record("unexpected error: \(error)")
            return
        }
        #expect(reason.contains("32768"))
    }
}

#if os(macOS)
/// Type-checks the fixture with one swiftc candidate. Returns exit status and stderr.
private func typecheckFixture(
    _ swiftc: String, fixture: URL, module: URL, sdkPath: String
) -> (status: Int32, diagnostics: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: swiftc)
    process.arguments = [
        "-typecheck",
        fixture.path,
        "-I", module.deletingLastPathComponent().path,
        "-sdk", sdkPath,
        "-target", "arm64-apple-macos15.0",
    ]
    let errorPipe = Pipe()
    process.standardError = errorPipe
    try? process.run()
    process.waitUntilExit()
    let diagnostics = String(
        decoding: errorPipe.fileHandleForReading.readDataToEndOfFile(),
        as: UTF8.self)
    return (process.terminationStatus, diagnostics)
}

/// swiftc candidates, best first: an explicit override, xcrun's default, then every installed
/// toolchain (user and system scopes). A snapshot-toolchain build paired with the system swiftc
/// fails at import ("module compiled with Swift X cannot be imported by Swift Y"), so the probe
/// walks the list when the first candidate cannot even import the module.
private func swiftcCandidates() throws -> [String] {
    var candidates = [String]()
    if let override = ProcessInfo.processInfo.environment["GLOSSEMATICS_SWIFTC"] {
        candidates.append(override)
    }
    if let xcrunSwiftc = try? run("/usr/bin/xcrun", ["--find", "swiftc"]) {
        candidates.append(xcrunSwiftc.output.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    let scopes = [
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Developer/Toolchains"),
        URL(fileURLWithPath: "/Library/Developer/Toolchains"),
    ]
    for scope in scopes {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: scope.path)) ?? []
        for name in names.sorted().reversed() {
            candidates.append(
                scope.appendingPathComponent("\(name)/usr/bin/swiftc").path)
        }
    }
    return candidates.filter { FileManager.default.isExecutableFile(atPath: $0) }
}

@Test func queryToQuerySimilarityDoesNotTypeCheck() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let fixture = packageRoot
        .appendingPathComponent("Tests")
        .appendingPathComponent("CompileFail")
        .appendingPathComponent("QueryToQuerySimilarity.swift")
    let moduleCandidates = [
        packageRoot.appendingPathComponent(
            ".build/out/Products/Debug/gloss_server.swiftmodule"),
        packageRoot.appendingPathComponent(
            ".build/arm64-apple-macosx/debug/Modules/gloss_server.swiftmodule"),
    ]
    let module = try #require(moduleCandidates.first {
        FileManager.default.fileExists(atPath: $0.path)
    })

    let sdk = try run("/usr/bin/xcrun", ["--sdk", "macosx", "--show-sdk-path"])
    let sdkPath = sdk.output.trimmingCharacters(in: .whitespacesAndNewlines)

    // Probe candidates until one can actually import the module, so the observed failure is the
    // fixture's type error rather than a toolchain/module version mismatch.
    var status: Int32 = 0
    var diagnostics = ""
    for candidate in try swiftcCandidates() {
        (status, diagnostics) = typecheckFixture(candidate, fixture: fixture, module: module, sdkPath: sdkPath)
        guard diagnostics.contains("cannot be imported by the Swift") else { break }
    }

    #expect(status != 0, "query-to-query similarity must not type-check")
    #expect(diagnostics.contains("DocumentEmbedding"), "diagnostics: \(diagnostics)")
    #expect(diagnostics.contains("QueryEmbedding"), "diagnostics: \(diagnostics)")
}
#endif

private struct OmniSmallFixture {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-small-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)

        func feature(_ name: String, _ dataType: String, _ shape: [Int]) -> [String: Any] {
            [
                "name": name,
                "dataType": dataType,
                "shape": Self.shape(shape),
            ]
        }
        func function(
            _ name: String,
            inputs: [[String: Any]],
            outputs: [[String: Any]]
        ) -> [String: Any] {
            [
                "name": name,
                "inputSchema": inputs,
                "outputSchema": outputs,
            ]
        }
        func writeMetadata(_ path: String, functions: [[String: Any]]) throws {
            let directory = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: [["functions": functions]]).write(
                to: directory.appendingPathComponent("metadata.json"))
        }

        let textBuckets = [32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768]
        let textBatchPairs = [(64, 32), (32, 64), (16, 128), (8, 256), (4, 512)]
        var textFunctions = textBuckets.map { bucket in
            function(
                "bucket_\(bucket)",
                inputs: [
                    feature("input_ids", "Int32", [1, bucket]),
                    feature("position_ids", "Int32", [3, 1, bucket]),
                    feature("selector", "Float32", [1, bucket]),
                ],
                outputs: [feature("embedding", "Float32", [1, 1_024])])
        }
        textFunctions.append(contentsOf: textBatchPairs.map { size, bucket in
            function(
                "bucket_\(bucket)_b\(size)",
                inputs: [
                    feature("input_ids", "Int32", [size, bucket]),
                    feature("position_ids", "Int32", [3, size, bucket]),
                    feature("selector", "Float32", [size, bucket]),
                ],
                outputs: [feature("embedding", "Float32", [size, 1_024])])
        })
        try writeMetadata("text.mlmodelc", functions: textFunctions)

        let imageBuckets = [1_024, 1_600, 2_304, 3_072, 4_032, 5_120]
        try writeMetadata("image.mlmodelc", functions: imageBuckets.map { patches in
            function(
                "f\(patches)",
                inputs: [
                    feature("pixel_values", "Float32", [patches, 1_536]),
                    feature("pos_embeds", "Float32", [patches, 1_024]),
                    feature("rope_cos", "Float32", [patches, 64]),
                    feature("rope_sin", "Float32", [patches, 64]),
                    feature("attn_bias", "Float32", [1, 1, 1, patches]),
                ],
                outputs: [feature("vision_features", "Float32", [patches / 4, 1_024])])
        })

        let audioBuckets = [200, 400, 800, 1_600, 3_200]
        try writeMetadata("audio.mlmodelc", functions: audioBuckets.map { frames in
            let chunks = frames / 200
            let tokens = chunks * 100
            return function(
                "f\(frames)",
                inputs: [
                    feature("packed_mel", "Float32", [128, frames]),
                    feature("conv_mask", "Float32", [chunks, 1, 200]),
                    feature("attn_bias", "Float32", [1, 1, tokens, tokens]),
                ],
                outputs: [feature("audio_features", "Float32", [chunks * 50, 1_024])])
        })

        let decoderBuckets = [128, 256, 512, 1_024, 2_048]
        try writeMetadata("embed.mlmodelc", functions: decoderBuckets.map { sequence in
            function(
                "f\(sequence)",
                inputs: [feature("input_ids", "Int32", [1, sequence])],
                outputs: [feature("out", "Float32", [1, sequence, 1_024])])
        })
        try writeMetadata("decoder.mlmodelc", functions: decoderBuckets.map { sequence in
            function(
                "f\(sequence)",
                inputs: [
                    feature("inputs_embeds", "Float32", [1, sequence, 1_024]),
                    feature("position_ids", "Int32", [3, 1, sequence]),
                    feature("selector", "Float32", [1, sequence]),
                ],
                outputs: [feature("embedding", "Float32", [1, 1_024])])
        })

        let videoBuckets = [256, 512, 1_024, 2_048]
        try writeMetadata("video.mlmodelc", functions: videoBuckets.map { patches in
            function(
                "f\(patches)",
                inputs: [
                    feature("pixel_values", "Float32", [patches, 1_536]),
                    feature("pos_embeds", "Float32", [patches, 1_024]),
                    feature("rope_cos", "Float32", [patches, 64]),
                    feature("rope_sin", "Float32", [patches, 64]),
                    feature("attn_bias", "Float32", [1, 1, patches, patches]),
                ],
                outputs: [feature("vision_features", "Float32", [patches / 4, 1_024])])
        })
        let tokenizer = root.appendingPathComponent("tokenizer")
        let vision = root.appendingPathComponent("vision")
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vision, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer_config.json"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("config.json"))
        try Data("fixture".utf8).write(to: vision.appendingPathComponent("meta.json"))
        try Data("fixture".utf8).write(to: vision.appendingPathComponent("pos_embed_table.f32"))
        try Data("fixture".utf8).write(to: vision.appendingPathComponent("rope_inv_freq.f32"))

        var checksums = [String: String]()
        for relative in try FileManager.default.subpathsOfDirectory(atPath: root.path)
            where relative != "manifest.json" {
            let url = root.appendingPathComponent(relative)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                checksums[relative] = SHA256.hash(
                    data: try Data(contentsOf: url)).map {
                        String(format: "%02x", $0)
                    }.joined()
            }
        }

        let manifest = GlossModelBundle.Manifest(
            modelID: "jinaai/jina-embeddings-v5-omni-small",
            embeddingDimension: 1_024,
            matryoshkaDimensions: [32, 64, 128, 256, 512, 1_024],
            source: .init(
                repo: "jinaai/jina-embeddings-v5-omni-small",
                revision: "87f7a45d1ae0265843f8569c47fb53847cb193c3"),
            prompts: .init(query: "Query: ", document: "Document: "),
            tokens: .init(
                padID: 151_643,
                image: .init(
                    prefixIDs: [151_644, 872, 198, 151_652],
                    suffixIDs: [151_653, 151_645, 198],
                    placeholderID: 151_655,
                    queryPrefixIDs: [151_644, 872, 198, 2_859, 25, 220, 151_652],
                    documentPrefixIDs: [151_644, 872, 198, 7_524, 25, 220, 151_652]),
                video: .init(
                    prefixIDs: [151_644, 872, 198, 151_652],
                    suffixIDs: [151_653, 151_645, 198],
                    placeholderID: 151_656,
                    queryPrefixIDs: [151_644, 872, 198, 2_859, 25, 220, 151_652],
                    documentPrefixIDs: [151_644, 872, 198, 7_524, 25, 220, 151_652]),
                audio: .init(
                    prefixIDs: [151_644, 872, 198, 151_670],
                    suffixIDs: [151_671, 151_645, 198],
                    placeholderID: 151_669,
                    queryPrefixIDs: [151_644, 872, 198, 2_859, 25, 220, 151_670],
                    documentPrefixIDs: [151_644, 872, 198, 7_524, 25, 220, 151_670])),
            text: .init(
                model: "text.mlmodelc",
                tokenizer: "tokenizer",
                buckets: textBuckets,
                maxTokens: 32_768,
                requiresAttentionMask: false,
                batch: .init(
                    sizes: textBatchPairs.map(\.0),
                    buckets: textBatchPairs.map(\.1))),
            image: .init(
                encoder: "image.mlmodelc",
                resources: "vision",
                patchBuckets: imageBuckets,
                preprocess: .init(
                    patch: 16,
                    merge: 2,
                    minPixels: 262_144,
                    maxPixels: 1_310_720)),
            audio: .init(
                encoder: "audio.mlmodelc",
                frameBuckets: audioBuckets,
                sampleRate: 16_000,
                maxSamples: 480_000,
                maxFrames: 3_000),
            video: .init(
                encoder: "video.mlmodelc",
                patchBuckets: videoBuckets),
            decoder: .init(
                embed: "embed.mlmodelc",
                model: "decoder.mlmodelc",
                sequenceBuckets: decoderBuckets),
            compiled: .init(
                format: "mlmodelc",
                tool: "coremlcompiler",
                platform: "macOS"),
            spaceID: "jinaai/jina-embeddings-v5-omni-small:1024:w8a16:native-v1",
            precision: .init(weights: "int8", activations: "float16"),
            artifactChecksums: .init(files: checksums))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: root.appendingPathComponent("manifest.json"))
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    func removeTokenizerConfiguration() throws {
        try FileManager.default.removeItem(
            at: root
                .appendingPathComponent("tokenizer")
                .appendingPathComponent("tokenizer_config.json"))
    }

    func corruptTextInputShape() throws {
        let url = root
            .appendingPathComponent("text.mlmodelc")
            .appendingPathComponent("metadata.json")
        var roots = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [[String: Any]])
        var metadata = try #require(roots.first)
        var functions = try #require(metadata["functions"] as? [[String: Any]])
        let index = try #require(functions.firstIndex {
            $0["name"] as? String == "bucket_32"
        })
        var function = functions[index]
        var inputs = try #require(function["inputSchema"] as? [[String: Any]])
        let inputIndex = try #require(inputs.firstIndex {
            $0["name"] as? String == "input_ids"
        })
        inputs[inputIndex]["shape"] = "[1, 31]"
        function["inputSchema"] = inputs
        functions[index] = function
        metadata["functions"] = functions
        roots[0] = metadata
        try JSONSerialization.data(withJSONObject: roots).write(to: url)
    }

    func addUnexpectedTextInput() throws {
        let url = root
            .appendingPathComponent("text.mlmodelc")
            .appendingPathComponent("metadata.json")
        var roots = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [[String: Any]])
        var metadata = try #require(roots.first)
        var functions = try #require(metadata["functions"] as? [[String: Any]])
        let index = try #require(functions.firstIndex {
            $0["name"] as? String == "bucket_32"
        })
        var function = functions[index]
        var inputs = try #require(function["inputSchema"] as? [[String: Any]])
        inputs.append(["name": "unexpected", "dataType": "Int32", "shape": "[1]"])
        function["inputSchema"] = inputs
        functions[index] = function
        metadata["functions"] = functions
        roots[0] = metadata
        try JSONSerialization.data(withJSONObject: roots).write(to: url)
    }

    private static func shape(_ values: [Int]) -> String {
        "[" + values.map(String.init).joined(separator: ", ") + "]"
    }
}

private func repositoryRoot() -> URL {
    var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    while url.path != "/" {
        if FileManager.default.fileExists(
            atPath: url.appendingPathComponent("artifacts").path),
           FileManager.default.fileExists(
            atPath: url.appendingPathComponent("Package.swift").path) {
            return url
        }
        url.deleteLastPathComponent()
    }
    return URL(fileURLWithPath: #filePath)
}

#if os(macOS)
private struct ProcessResult {
    let output: String
}

private func run(_ executable: String, _ arguments: [String]) throws -> ProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    let outputPipe = Pipe()
    process.standardOutput = outputPipe
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw OmniSmallError.inferenceFailed(
            "\(executable) failed with status \(process.terminationStatus)")
    }
    return ProcessResult(output: String(
        decoding: outputPipe.fileHandleForReading.readDataToEndOfFile(),
        as: UTF8.self))
}
#endif
