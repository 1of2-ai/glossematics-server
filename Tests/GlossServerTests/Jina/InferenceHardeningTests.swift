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
