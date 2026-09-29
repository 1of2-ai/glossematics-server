import AVFoundation
import CoreML
import Foundation
import Testing
@testable import gloss_server

private let goldenFixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle")

private func expectGolden(_ values: [Float], dimensions: Int) {
    #expect(values.count == dimensions)
    let value = Float(1 / Double(dimensions).squareRoot())
    #expect(values.allSatisfy { abs($0 - value) < 0.0002 })
}

@Test func goldenFixtureRunsRealCoreMLTextAudioAndVideo() async throws {
    let manifest = try GlossModelBundle(url: goldenFixture).manifest
    #expect(manifest.converter?.name == "dummy-noop")
    #expect(manifest.audio != nil && manifest.image != nil && manifest.video != nil)

    let model = try await OmniSmall.load(from: goldenFixture)
    let documents = try await model.embedDocuments(
        [.text("first document"), .text("second document")], dimensions: .d32)
    #expect(documents.count == 2)
    for document in documents { expectGolden(document.values, dimensions: 32) }

    let audioURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-golden-audio-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: audioURL) }
    let count = 3_200
    let format = try #require(AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
        channels: 1, interleaved: false))
    let buffer = try #require(AVAudioPCMBuffer(
        pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
    buffer.frameLength = AVAudioFrameCount(count)
    let samples = try #require(buffer.floatChannelData?[0])
    for index in 0..<count {
        samples[index] = 0.2 * sinf(2 * .pi * 220 * Float(index) / 16_000)
    }
    do {
        let file = try AVAudioFile(forWriting: audioURL, settings: format.settings)
        try file.write(from: buffer)
    }
    let audio = try await model.embedDocument(.audio(audioURL), dimensions: .d64)
    expectGolden(audio.values, dimensions: 64)
    let audioBytes = try Data(contentsOf: audioURL)
    let uploadedAudio = try await model.embedDocument(.audioData(audioBytes), dimensions: .d64)
    expectGolden(uploadedAudio.values, dimensions: 64)

    let videoURL = goldenFixture.deletingLastPathComponent()
        .appendingPathComponent("golden-video.mp4")
    let videoBytes = try Data(contentsOf: videoURL)
    let video = try await model.embedDocument(.video(videoURL), dimensions: .d128)
    expectGolden(video.values, dimensions: 128)
    let uploadedVideo = try await model.embedDocument(.videoData(videoBytes), dimensions: .d128)
    expectGolden(uploadedVideo.values, dimensions: 128)
}

@Test func goldenFixtureLongVideoRunsLargestNativeBucket() async throws {
    let videoURL = goldenFixture.deletingLastPathComponent()
        .appendingPathComponent("golden-long-video.mp4")
    let preprocessor = GlossImagePreprocessor()
    let extracted = try GlossVideoFile.extractFrames(
        videoURL, maxPatches: 2_048, preprocessor: preprocessor)
    let patches = (extracted.frames.count / preprocessor.temporal)
        * (extracted.h / preprocessor.patch) * (extracted.w / preprocessor.patch)
    #expect(extracted.frames.count == 32)
    #expect(patches > 1_024 && patches <= 2_048,
            "long-clip profile should execute the f2048 native video function")

    let model = try await OmniSmall.load(from: goldenFixture)
    let video = try await model.embedDocument(.video(videoURL), dimensions: .d32)
    expectGolden(video.values, dimensions: 32)
}

@Test func smartResizeCannotExceedNativePixelBudgetAtExtremeAspectRatios() {
    let preprocessor = GlossImagePreprocessor(
        minPixels: 262_144, maxPixels: 1_310_720)
    for (height, width) in [(40_000_000, 1), (1, 40_000_000), (10_000, 5_000)] {
        let (resizedHeight, resizedWidth) = preprocessor.smartResize(
            h: height, w: width, maxPixelsOverride: 1_310_720)
        #expect(resizedHeight > 0 && resizedWidth > 0)
        #expect(resizedHeight.isMultiple(of: 32) && resizedWidth.isMultiple(of: 32))
        #expect(resizedHeight * resizedWidth <= 1_310_720)
    }
}

@Test func malformedImageAndVideoGeometryFailsBeforeAllocationOrCoreML() throws {
    let preprocessor = GlossImagePreprocessor()
    #expect(throws: GlossImagePreprocessor.ImageError.self) {
        _ = try preprocessor.pixelValues(rgb: [UInt8](repeating: 0, count: 3), h: 32, w: 32)
    }
    #expect(throws: GlossImagePreprocessor.ImageError.self) {
        _ = try preprocessor.videoPixelValues(
            frames: [[UInt8](repeating: 0, count: 3), [UInt8](repeating: 0, count: 3)],
            h: 32, w: 32)
    }

    let manifest = try GlossModelBundle(url: goldenFixture).manifest
    let image = try #require(manifest.image)
    let video = try #require(manifest.video)
    let decoder = try #require(manifest.decoder)
    let imagePipeline = try GlossImageEmbedderMasked(
        visionModelURL: goldenFixture.appendingPathComponent(image.encoder),
        embedModelURL: goldenFixture.appendingPathComponent(decoder.embed),
        decoderModelURL: goldenFixture.appendingPathComponent(decoder.model),
        resourcesDir: goldenFixture.appendingPathComponent(image.resources),
        tokens: .jinaV5OmniSmallImage)
    let videoPipeline = try GlossVideoEmbedderMasked(
        visionModelURL: goldenFixture.appendingPathComponent(video.encoder),
        embedModelURL: goldenFixture.appendingPathComponent(decoder.embed),
        decoderModelURL: goldenFixture.appendingPathComponent(decoder.model),
        resourcesDir: goldenFixture.appendingPathComponent(image.resources),
        tokens: .jinaV5OmniSmallVideo)
    #expect(throws: VisionCoreMLEncoderMasked.EncoderError.self) {
        _ = try imagePipeline.embed(pixelValues: [], gh: 0, gw: 0)
    }
    #expect(throws: VideoCoreMLEncoderMasked.EncoderError.self) {
        _ = try videoPipeline.embed(pixelValues: [], t: 0, gh: 0, gw: 0)
    }
    #expect(throws: VideoCoreMLEncoderMasked.EncoderError.self) {
        _ = try videoPipeline.embed(
            videoURL: URL(fileURLWithPath: "/nonexistent.mp4"), frameCount: 8, frameSize: 33)
    }
}

@Test func coreMLArrayReadsRequireShapeTypeAndContiguousStorage() throws {
    let floats = try MLMultiArray(shape: [2, 2], dataType: .float32)
    try CoreMLArrayReader.fillFloat32(floats, with: [1, 2, 3, 4], label: "test")
    #expect(try CoreMLArrayReader.float32(floats, shape: [2, 2], label: "test") == [1, 2, 3, 4])
    #expect(throws: CoreMLArrayReader.ArrayError.self) {
        try CoreMLArrayReader.float32(floats, shape: [1, 4], label: "test")
    }
    let integers = try MLMultiArray(shape: [2, 2], dataType: .int32)
    #expect(throws: CoreMLArrayReader.ArrayError.self) {
        try CoreMLArrayReader.float32(integers, shape: [2, 2], label: "test")
    }
}

@Test func mediaDecoderRejectsInvalidScatterBeforeCoreMLPrediction() throws {
    let decoder = try GeneralMediaDecoder(
        embedModelURL: goldenFixture.appendingPathComponent("embed_multifunc.mlmodelc"),
        decoderModelURL: goldenFixture.appendingPathComponent("decoder_embeds_multifunc.mlmodelc"))
    #expect(throws: GeneralMediaDecoder.DecoderError.self) {
        try decoder.decode(tokenIds: [1, 2], features: [Float](repeating: 0, count: 1_023), scatterOffset: 1)
    }
    #expect(throws: GeneralMediaDecoder.DecoderError.self) {
        try decoder.decode(tokenIds: [1, 2], features: [Float](repeating: 0, count: 1_024), scatterOffset: 2)
    }
}

@Test func visionPositionMetadataMustMatchPinnedGeometry() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-vision-meta-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let meta = root.appendingPathComponent("meta.json")
    try Data(#"{"num_grid_per_side":100000000,"hidden":1024,"spatial_merge_size":2,"patch_size":16,"rope_theta":10000,"pos_table_rows":2304,"rope_inv_freq_len":16}"#.utf8)
        .write(to: meta)
    #expect(throws: VisionPositions.PosError.self) {
        try VisionPositions(
            metaURL: meta,
            posTableURL: root.appendingPathComponent("missing.f32"),
            invFreqURL: root.appendingPathComponent("missing-rope.f32"))
    }
}

// MARK: - Video source-size cap

/// A copy of the golden MP4 whose track header declares another frame size (and optionally a 90
/// degree rotation). Only the container metadata changes, so it exercises the pre-decode size
/// checks without needing a 4K or 8K encoder.
private func goldenVideoDeclaring(width: UInt32, height: UInt32, rotated: Bool = false) throws -> Data {
    var bytes = [UInt8](try Data(contentsOf: goldenFixture.deletingLastPathComponent()
        .appendingPathComponent("golden-video.mp4")))
    let tag = Array("tkhd".utf8)
    let index = try #require((0..<(bytes.count - 4)).first { Array(bytes[$0..<($0 + 4)]) == tag })
    try #require(bytes[index + 4] == 0, "expected a version-0 tkhd box")
    func put(_ value: UInt32, at offset: Int) {
        for shift in 0..<4 { bytes[offset + shift] = UInt8((value >> UInt32(24 - 8 * shift)) & 0xff) }
    }
    put(width << 16, at: index + 4 + 76)
    put(height << 16, at: index + 4 + 80)
    if rotated {   // matrix [a b u c d v x y w] for a 90 degree clockwise rotation
        put(0, at: index + 4 + 40)
        put(0x0001_0000, at: index + 4 + 44)
        put(0xFFFF_0000, at: index + 4 + 52)
        put(0, at: index + 4 + 56)
    }
    return Data(bytes)
}

private func withVideoFile<T>(_ data: Data, _ body: (URL) throws -> T) throws -> T {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-cap-test-\(UUID().uuidString).mp4")
    try data.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    return try body(url)
}

@Test func videoSourceSizeCapAdmitsDCI4KInEitherOrientationAndRejects8K() throws {
    // Admitted: DCI 4K and UHD, landscape and portrait, and the square at the cap.
    for (width, height) in [(4_096.0, 2_160.0), (2_160.0, 4_096.0), (3_840.0, 2_160.0),
                            (2_160.0, 3_840.0), (4_096.0, 4_096.0), (64.0, 64.0), (1.0, 1.0)] {
        let size = try VideoFrameDecoder.validateSourceSize(width: width, height: height)
        #expect(size.width == Int(width) && size.height == Int(height))
    }
    // Rejected as too large: 5K, 8K in both orientations, one pixel over the edge, huge values.
    for (width, height) in [(5_120.0, 2_880.0), (7_680.0, 4_320.0), (4_320.0, 7_680.0), (4_097.0, 100.0),
                            (100.0, 4_097.0), (1e30, 1e30), (Double.greatestFiniteMagnitude, 10)] {
        do {
            _ = try VideoFrameDecoder.validateSourceSize(width: width, height: height)
            Issue.record("\(width) x \(height) must exceed the source cap")
        } catch let failure as VideoFrameDecoder.Failure {
            guard case .frameTooLarge = failure else {
                Issue.record("\(width) x \(height): expected frameTooLarge, got \(failure)")
                continue
            }
            #expect("\(failure)".contains("DCI 4K"), "the message must explain the cap")
        }
    }
    // Rejected as invalid: zero, negative, sub-pixel, and non-finite sizes — none may trap.
    for (width, height) in [(0.0, 0.0), (-1.0, 100.0), (100.0, -5.0), (0.4, 100.0), (Double.nan, 100.0),
                            (100.0, .infinity), (-Double.infinity, 100.0), (.nan, .nan)] {
        do {
            _ = try VideoFrameDecoder.validateSourceSize(width: width, height: height)
            Issue.record("\(width) x \(height) must be an invalid frame size")
        } catch let failure as VideoFrameDecoder.Failure {
            guard case .invalidFrameSize = failure else {
                Issue.record("\(width) x \(height): expected invalidFrameSize, got \(failure)")
                continue
            }
        }
    }
    #expect(GlossVideoFile.maximumSourcePixels == 4_096 * 4_096)
}

@Test func oversizedVideoIsRejectedBeforeAnyDecodeWithAClear400() async throws {
    let eightK = try goldenVideoDeclaring(width: 7_680, height: 4_320)
    try withVideoFile(eightK) { url in
        #expect(throws: VideoFrameDecoder.Failure.self) { _ = try VideoFrameDecoder.open(url) }
        #expect(throws: VideoFrameDecoder.Failure.self) {
            _ = try GlossVideoFile.extractFrames(url, maxPatches: 2_048, preprocessor: GlossImagePreprocessor())
        }
    }

    let model = try await OmniSmall.load(from: goldenFixture)
    for input in [OmniSmall.Input.videoData(eightK)] {
        do {
            _ = try await model.embedDocument(input)
            Issue.record("an 8K video must be rejected")
        } catch let error as OmniSmallError {
            guard case let .invalidInput(reason) = error else {
                Issue.record("expected invalidInput (HTTP 400), got \(error)")
                continue
            }
            #expect(reason.contains("7680") && reason.contains("4320") && reason.contains("DCI 4K"), Comment(rawValue: reason))
        }
    }
}

@Test func dci4KVideoDeclarationIsAdmittedInEitherOrientation() throws {
    let landscape = try goldenVideoDeclaring(width: 4_096, height: 2_160)
    try withVideoFile(landscape) { url in
        let source = try VideoFrameDecoder.open(url)
        #expect(source.width == 4_096 && source.height == 2_160 && source.quarterTurns == 0)
    }
    // A portrait file stores its frames sideways and rotates them for display: the displayed size
    // is what the cap applies to, and it is the same 4096 x 2160 either way.
    let rotated = try goldenVideoDeclaring(width: 2_160, height: 4_096, rotated: true)
    try withVideoFile(rotated) { url in
        let source = try VideoFrameDecoder.open(url)
        #expect(source.quarterTurns == 1)
        #expect(source.width == 4_096 && source.height == 2_160)
    }
    let portrait = try goldenVideoDeclaring(width: 2_160, height: 4_096)
    try withVideoFile(portrait) { url in
        let source = try VideoFrameDecoder.open(url)
        #expect(source.width == 2_160 && source.height == 4_096)
    }
}

// MARK: - Short audio and the mel frontend

@Test func wholeClipLogMelThrowsInsteadOfTrappingAndBidirLMMapsItToInvalidInput() throws {
    let mel = try GlossMelFrontend()
    for count in [0, 1, 199, 200] {   // reflect padding needs MORE than nFFT / 2 = 200 samples
        do {
            _ = try mel.wholeClipLogMel([Float](repeating: 0.1, count: count))
            Issue.record("\(count) samples must be too short")
        } catch GlossMelFrontend.MelError.audioTooShort(let reported) {
            #expect(reported == count)
        }
    }
    let (features, frames) = try mel.wholeClipLogMel([Float](repeating: 0.1, count: 201))
    #expect(frames == 1 && features.count == mel.nMels)

    // BidirLM keeps its own 100 ms minimum, and a mel-frontend failure of any kind reaches the
    // same invalid-input path (HTTP 400) rather than a crash or a generic server error.
    #expect(throws: BidirLMMediaInputs.Failure.self) {
        _ = try BidirLMMediaInputs.audio([Float](repeating: 0.1, count: 150), frontend: mel)
    }
    #expect(throws: BidirLMMediaInputs.Failure.self) {
        _ = try BidirLMMediaInputs.audio([], frontend: mel)
    }
    let prepared = try BidirLMMediaInputs.audio([Float](repeating: 0.1, count: 1_600), frontend: mel)
    #expect(prepared.frames == 10)
}
