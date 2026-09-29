import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import gloss_server

/// End-to-end retrieval through the public API — the e2e layer above the parity tests.
///
/// This test checks text parity and that a full bundle retrieves the right topic for paraphrase
/// queries on the adversarial synthetic
/// corpus in `reference/retrieval_corpus.json` (generator archived alongside the conversion tooling,
/// which asserts the corpus is not keyword-solvable). Mirrors the `gloss-retrieval` Validation CLI
/// so `make verify-model` covers the same gates: the embedding ranking must retrieve every query's topic
/// at rank 1, must beat a baited keyword (Jaccard) baseline, and must hold under Matryoshka-256
/// truncation.
///
/// Run only with an explicitly supplied full model. The constant Core ML fixture cannot prove quality.
@Test(.enabled(if: ProcessInfo.processInfo.environment["GLOSS_PRODUCTION_BUNDLE"] != nil, "Set GLOSS_PRODUCTION_BUNDLE to a full native bundle"))
func productionModelRetrievalEndToEnd() async throws {
    guard let bundlePath = ProcessInfo.processInfo.environment["GLOSS_PRODUCTION_BUNDLE"] else { return }
    let bundleURL = URL(fileURLWithPath: bundlePath, isDirectory: true)
    let bundle = try GlossModelBundle(url: bundleURL)
    #expect(bundle.manifest.converter?.name != "dummy-noop")

    struct Bait: Decodable { let topic: String; let text: String }
    struct Topic: Decodable { let name: String; let query: String; let docs: [String]; let baits: [Bait] }
    struct Corpus: Decodable { let seed: Int; let topics: [Topic] }
    struct PoolItem { let topic: String; let text: String }

    func words(_ s: String) -> Set<String> {
        var out = Set<String>()
        var current = ""
        for ch in s.lowercased() {
            if ch.isLetter || ch.isNumber { current.append(ch) }
            else if !current.isEmpty { out.insert(current); current = "" }
        }
        if !current.isEmpty { out.insert(current) }
        return out
    }
    func jaccard(_ a: String, _ b: String) -> Double {
        let wa = words(a), wb = words(b)
        guard !wa.isEmpty, !wb.isEmpty else { return 0 }
        return Double(wa.intersection(wb).count) / Double(wa.union(wb).count)
    }
    func rankBy(_ sims: [Double]) -> [Int] { sims.indices.sorted { sims[$0] > sims[$1] } }

    let corpus = try JSONDecoder().decode(Corpus.self,
        from: Data(contentsOf: path("reference/jina/retrieval_corpus.json")))
    let queries = corpus.topics
    var pool: [PoolItem] = []
    for t in corpus.topics {
        for d in t.docs { pool.append(PoolItem(topic: t.name, text: d)) }
        for b in t.baits { pool.append(PoolItem(topic: b.topic, text: b.text)) }
    }

    // Production API: ordered document batches, typed queries. Loading runs the full native
    // contract validation, so this exercises exactly what a host performs.
    let model = try await OmniSmall.load(
        from: bundleURL, dimensions: .d1024)
    struct TextRef: Decodable {
        let text: String
        let prompt_name: String
        let embedding: [String: [Float]]
    }
    let references = try JSONDecoder().decode(
        [TextRef].self,
        from: Data(contentsOf: path("reference/jina/text_reference.json")))
    for reference in references.prefix(6) {
        let result = try await (reference.prompt_name == "query"
            ? model.embedQuery(.text(reference.text)).values
            : model.embedDocument(.text(reference.text)).values)
        let expected = try #require(reference.embedding["1024"])
        let similarity = cosine(result, expected)
        #expect(similarity > 0.995, "text parity cos=\(similarity) for \(reference.prompt_name)")
    }

    // The golden fixture proves control flow, not numeric model execution. Exercise each
    // production media tower with bounded, generated inputs before retrieval quality gates.
    func checkMediaVector(_ values: [Float], name: String) {
        #expect(values.count == 1_024, "\(name) dimensions")
        #expect(values.allSatisfy { $0.isFinite }, "\(name) contains non-finite values")
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1) < 0.005, "\(name) norm \(norm)")
        #expect((values.max() ?? 0) - (values.min() ?? 0) > 0.01,
                "\(name) output appears constant")
    }
    let imageEmbedding = try await model.embedDocument(.imageData(makeTestPNG())).values
    checkMediaVector(imageEmbedding, name: "image")

    let audioURL = try makeTestWAV()
    defer { try? FileManager.default.removeItem(at: audioURL) }
    let audioEmbedding = try await model.embedDocument(.audio(audioURL)).values
    checkMediaVector(audioEmbedding, name: "audio")

    let videoEmbedding = try await model.embedDocument(
        .video(path("Fixtures/golden-video.mp4"))).values
    checkMediaVector(videoEmbedding, name: "video")
    // HF reference for Fixtures/golden-video.mp4 (same ffmpeg recipe), exported by
    // GlossematicsCoreML/python/parity/export_video_swift_refs.py --file into its reference/.
    if let refs = ProcessInfo.processInfo.environment["GLOSS_JINA_REFERENCE"] {
        let fileVideoReference = try Data(contentsOf: URL(fileURLWithPath: refs)
            .appendingPathComponent("video_swift/golden_video_file_emb.f32"))
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let fileVideoCosine = cosine(videoEmbedding, fileVideoReference)
        print("media parity video file cosine=\(fileVideoCosine)")
        #expect(fileVideoCosine > 0.98, "video MP4 decode/sampling parity cosine \(fileVideoCosine)")
    }
    let longVideoEmbedding = try await model.embedDocument(
        .video(path("Fixtures/golden-long-video.mp4"))).values
    checkMediaVector(longVideoEmbedding, name: "long video")
    // Converter goldens live in GlossematicsCoreML/reference (GLOSS_JINA_REFERENCE).
    if ProcessInfo.processInfo.environment["GLOSS_JINA_REFERENCE"] != nil {
        try checkProductionMediaParity(bundle)
    }

    let poolEmbs = try await model.embedDocuments(pool.map { .text($0.text) }).map(\.values)
    let queryEmbs = try await model.embedQueries(queries.map { .text($0.query) }).map(\.values)
    let dim = model.dimensions.rawValue

    func topicP1(_ qs: [[Float]], _ ds: [[Float]]) -> Int {
        var hits = 0
        for (qi, topic) in queries.enumerated() {
            let sims = ds.map { cosine($0, qs[qi]) }
            if pool[rankBy(sims)[0]].topic == topic.name { hits += 1 }
        }
        return hits
    }
    func topicMRR(_ qs: [[Float]], _ ds: [[Float]]) -> Double {
        var rr = 0.0
        for (qi, topic) in queries.enumerated() {
            let sims = ds.map { cosine($0, qs[qi]) }
            if let rank = rankBy(sims).firstIndex(where: { pool[$0].topic == topic.name }) {
                rr += 1.0 / Double(rank + 1)
            }
        }
        return rr / Double(queries.count)
    }

    let fullP1 = topicP1(queryEmbs, poolEmbs)
    let truncP1 = topicP1(queryEmbs.map { matryoshka($0, dim: 256) }, poolEmbs.map { matryoshka($0, dim: 256) })
    let fullMRR = topicMRR(queryEmbs, poolEmbs)

    // Keyword baseline over the same pool — baits must pull it toward the wrong topics.
    var lexP1 = 0
    for (qi, topic) in queries.enumerated() {
        let sims = pool.map { jaccard(queries[qi].query, $0.text) }
        if pool[rankBy(sims)[0]].topic == topic.name { lexP1 += 1 }
    }

    #expect(fullP1 == queries.count, "retrieval P@1 \(fullP1)/\(queries.count) at dim \(dim)")
    #expect(fullMRR > 0.95, "retrieval MRR \(fullMRR) at dim \(dim)")
    #expect(fullP1 > lexP1, "embedding (\(fullP1)) must beat the baited keyword baseline (\(lexP1))")
    #expect(truncP1 >= fullP1 - 1, "Matryoshka-256 P@1 \(truncP1) within 1 of full-dim \(fullP1)")
}

private func checkProductionMediaParity(_ bundle: GlossModelBundle) throws {
    func f32(_ relative: String) throws -> [Float] {
        try Data(contentsOf: goldenPath(relative)).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }
    let image = try #require(bundle.manifest.image)
    let audio = try #require(bundle.manifest.audio)
    let video = try #require(bundle.manifest.video)
    let decoder = try #require(bundle.manifest.decoder)
    let imageTokens = try #require(bundle.manifest.tokens.image)
    let audioTokens = try #require(bundle.manifest.tokens.audio)
    let videoTokens = try #require(bundle.manifest.tokens.video)

    let imageRGB = [UInt8](try Data(contentsOf: goldenPath("vision_swift/g24x24_rgb.u8")))
    let preprocessor = GlossImagePreprocessor()
    let imagePixels = try preprocessor.pixelValues(rgb: imageRGB, h: 384, w: 384)
    let imagePixelReference = try f32("vision_swift/g24x24_pixel_values.f32")
    #expect(imagePixels.pixels.count == imagePixelReference.count)
    #expect(zip(imagePixels.pixels, imagePixelReference).allSatisfy { abs($0.0 - $0.1) < 0.00001 },
            "image patchification diverged from the source processor")
    let imagePipeline = try GlossImageEmbedderMasked(
        visionModelURL: bundle.resolve(image.encoder),
        embedModelURL: bundle.resolve(decoder.embed),
        decoderModelURL: bundle.resolve(decoder.model),
        resourcesDir: bundle.resolve(image.resources),
        tokens: imageTokens.mediaTokens,
        patchBuckets: image.patchBuckets,
        padTokenID: bundle.manifest.tokens.padID,
        sequenceBuckets: decoder.sequenceBuckets)
    let imageReference = try f32("vision_swift/g24x24_emb.f32")
    let imageCosine = cosine(
        try imagePipeline.embed(rgb: imageRGB, h: 384, w: 384, prompt: .document),
        imageReference)
    print("media parity image cosine=\(imageCosine)")
    #expect(imageCosine > 0.995, "image FP32 parity cosine \(imageCosine)")

    let audioPipeline = try GlossAudioEmbedderMasked(
        audioModelURL: bundle.resolve(audio.encoder),
        embedModelURL: bundle.resolve(decoder.embed),
        decoderModelURL: bundle.resolve(decoder.model),
        tokens: audioTokens.mediaTokens,
        padTokenID: bundle.manifest.tokens.padID,
        sequenceBuckets: decoder.sequenceBuckets)
    for tag in ["3s", "6s", "30s"] {
        let wave = try f32("audio_swift/off_\(tag)_wave.f32")
        let reference = try f32("audio_swift/off_\(tag)_emb.f32")
        let similarity = cosine(try audioPipeline.embed(wave, prompt: .document), reference)
        print("media parity audio \(tag) cosine=\(similarity)")
        #expect(similarity > 0.995, "audio \(tag) FP32 parity cosine \(similarity)")
    }

    let videoPipeline = try GlossVideoEmbedderMasked(
        visionModelURL: bundle.resolve(video.encoder),
        embedModelURL: bundle.resolve(decoder.embed),
        decoderModelURL: bundle.resolve(decoder.model),
        resourcesDir: bundle.resolve(image.resources),
        tokens: videoTokens.mediaTokens,
        patchBuckets: video.patchBuckets,
        padTokenID: bundle.manifest.tokens.padID,
        sequenceBuckets: decoder.sequenceBuckets)
    for size in [256, 384] {
        let tag = "v4f\(size)"
        let rgb = [UInt8](try Data(contentsOf: goldenPath("video_swift/\(tag)_frames.u8")))
        let frameBytes = size * size * 3
        let frames = (0..<4).map { Array(rgb[$0 * frameBytes ..< ($0 + 1) * frameBytes]) }
        if size == 256 {
            let pixels = try preprocessor.videoPixelValues(frames: frames, h: size, w: size)
            let pixelReference = try f32("video_swift/\(tag)_pixel_values.f32")
            #expect(pixels.pixels.count == pixelReference.count)
            #expect(zip(pixels.pixels, pixelReference).allSatisfy { abs($0.0 - $0.1) < 0.00001 },
                    "video patchification diverged from the source processor")
        }
        let reference = try f32("video_swift/\(tag)_emb.f32")
        let similarity = cosine(
            try videoPipeline.embed(frames: frames, h: size, w: size, prompt: .document),
            reference)
        print("media parity video \(tag) cosine=\(similarity)")
        #expect(similarity > 0.995, "video \(tag) FP32 parity cosine \(similarity)")
    }
}

// MARK: - Shared helpers (same root-discovery pattern as EmbedderTests)

private let root: URL = {
    var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    while url.path != "/" {
        if FileManager.default.fileExists(atPath: url.appendingPathComponent("reference").path),
           FileManager.default.fileExists(atPath: url.appendingPathComponent("Package.swift").path) {
            return url
        }
        url.deleteLastPathComponent()
    }
    return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
}()
private func path(_ rel: String) -> URL { root.appendingPathComponent(rel) }
private func exists(_ rel: String) -> Bool { FileManager.default.fileExists(atPath: path(rel).path) }

private enum TestMediaError: Error { case pngEncodingFailed }

private func makeTestPNG() throws -> Data {
    let size = 64
    var pixels = [UInt8](repeating: 255, count: size * size * 4)
    for y in 0..<size {
        for x in 0..<size {
            let index = (y * size + x) * 4
            pixels[index] = UInt8(x * 4)
            pixels[index + 1] = UInt8(y * 4)
            pixels[index + 2] = 128
        }
    }
    let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
    let image = try #require(CGImage(
        width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let output = NSMutableData()
    let destination = try #require(CGImageDestinationCreateWithData(
        output as CFMutableData, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw TestMediaError.pngEncodingFailed }
    return output as Data
}

private func makeTestWAV() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-production-media-\(UUID().uuidString).wav")
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
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
    return url
}

/// Converter golden files (`vision_swift/`, `video_swift/`) under GLOSS_JINA_REFERENCE.
private func goldenPath(_ relative: String) -> URL {
    URL(fileURLWithPath: ProcessInfo.processInfo.environment["GLOSS_JINA_REFERENCE"] ?? "/nonexistent")
        .appendingPathComponent(relative)
}
