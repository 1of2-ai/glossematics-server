import Foundation
import Testing
@testable import gloss_server

/// Full-model checks for a real jina-embeddings-v5-omni-small bundle. Enabled with
/// `GLOSS_JINA_BUNDLE` (bundle directory) and `GLOSS_JINA_REFERENCE` (the GlossematicsCoreML
/// `reference/` directory with `video_swift/` goldens, exported by
/// `python/parity/export_video_swift_refs.py`). The HTTP-level MP4 references live in this repo
/// (`reference/jina/video_reference.json`).

private func env(_ name: String) -> URL? {
    ProcessInfo.processInfo.environment[name].map { URL(fileURLWithPath: $0, isDirectory: true) }
}

private func jinaEnabled() -> Bool { env("GLOSS_JINA_BUNDLE") != nil && env("GLOSS_JINA_REFERENCE") != nil }

private func floats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
    return dot / (na.squareRoot() * nb.squareRoot())
}

private func videoPipeline(_ bundle: GlossModelBundle) throws -> GlossVideoEmbedderMasked {
    let m = bundle.manifest
    let video = try #require(m.video), image = try #require(m.image), decoder = try #require(m.decoder)
    let tokens = try #require(m.tokens.video)
    return try GlossVideoEmbedderMasked(
        visionModelURL: bundle.resolve(video.encoder), embedModelURL: bundle.resolve(decoder.embed),
        decoderModelURL: bundle.resolve(decoder.model), resourcesDir: bundle.resolve(image.resources),
        tokens: tokens.mediaTokens, featureDim: m.embeddingDimension, patchBuckets: video.patchBuckets,
        padTokenID: m.tokens.padID, encoderUnits: .cpuAndGPU, decoderUnits: nil,
        sequenceBuckets: decoder.sequenceBuckets)
}

@Suite(.serialized) struct JinaFullModel {
    /// Raw frames and processor pixel values from the HF export: isolates patchify, the video
    /// tower, and the decoder from file decoding, sampling, and resizing.
    @Test(.enabled(if: jinaEnabled(), "Set GLOSS_JINA_BUNDLE and GLOSS_JINA_REFERENCE"))
    func videoGoldenFrames() throws {
        let bundle = try GlossModelBundle(url: try #require(env("GLOSS_JINA_BUNDLE")))
        let refs = try #require(env("GLOSS_JINA_REFERENCE")).appendingPathComponent("video_swift")
        struct Case: Decodable { let tag: String; let nframes: Int; let fsize: Int; let t: Int; let gh: Int; let gw: Int }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: refs.appendingPathComponent("manifest.json")))
        let pipeline = try videoPipeline(bundle)
        for c in cases {
            let raw = try Data(contentsOf: refs.appendingPathComponent("v\(c.tag)_frames.u8"))
            let size = c.fsize * c.fsize * 3
            let frames = (0..<c.nframes).map { [UInt8](raw[($0 * size)..<(($0 + 1) * size)]) }
            let expected = try floats(refs.appendingPathComponent("v\(c.tag)_emb.f32"))
            let fromFrames = try pipeline.embed(frames: frames, h: c.fsize, w: c.fsize, prompt: .document)
            let pixels = try floats(refs.appendingPathComponent("v\(c.tag)_pixel_values.f32"))
            let fromPixels = try pipeline.embed(pixelValues: pixels, t: c.t, gh: c.gh, gw: c.gw, prompt: .document)
            let (swiftPixels, _, _, _) = try pipeline.preprocessor.videoPixelValues(frames: frames, h: c.fsize, w: c.fsize)
            let pixelError = zip(swiftPixels, pixels).map { abs($0 - $1) }.max() ?? .infinity
            print("video \(c.tag): frames cos \(cosine(fromFrames, expected)) | pixels cos \(cosine(fromPixels, expected)) | patchify max err \(pixelError)")
            #expect(pixelError < 1e-5, "\(c.tag) patchify")
            #expect(cosine(fromPixels, expected) > 0.995, "\(c.tag) tower + decoder")
            #expect(cosine(fromFrames, expected) > 0.995, "\(c.tag) frames")
        }
    }

    /// The HTTP references: MP4 -> AVFoundation decode and 2 fps sampling -> resize -> model.
    @Test(.enabled(if: jinaEnabled(), "Set GLOSS_JINA_BUNDLE and GLOSS_JINA_REFERENCE"))
    func videoFilesMatchTheSourceRecipe() throws {
        let bundle = try GlossModelBundle(url: try #require(env("GLOSS_JINA_BUNDLE")))
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        struct Ref: Decodable {
            struct Case: Decodable { let name: String; let mp4: String; let embedding: [Float]; let sampled_indices: [Int]; let grid_thw: [Int] }
            let cases: [Case]
        }
        let ref = try JSONDecoder().decode(Ref.self, from: Data(contentsOf: root.appendingPathComponent("reference/jina/video_reference.json")))
        let pipeline = try videoPipeline(bundle)
        let dump = ProcessInfo.processInfo.environment["GLOSS_JINA_DUMP"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        for c in ref.cases {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(c.name)-\(UUID().uuidString).mp4")
            try Data(base64Encoded: c.mp4)!.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let input = try GlossVideoFile.extractFrames(url, maxPatches: pipeline.encoder.patchBuckets.last!,
                                                         preprocessor: pipeline.preprocessor)
            let emb = try pipeline.embed(frames: input.frames, h: input.h, w: input.w, prompt: .document)
            print("video file \(c.name): frames \(input.frames.count) \(input.h)x\(input.w) (reference grid \(c.grid_thw), indices \(c.sampled_indices)) cos \(cosine(emb, c.embedding))")
            if let dump {
                try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
                try Data(input.frames.flatMap { $0 }).write(to: dump.appendingPathComponent("\(c.name)_frames.u8"))
                try Data("\(input.frames.count) \(input.h) \(input.w)\n".utf8).write(to: dump.appendingPathComponent("\(c.name)_shape.txt"))
            }
            #expect(cosine(emb, c.embedding) > 0.995, "\(c.name)")
        }
    }
}
