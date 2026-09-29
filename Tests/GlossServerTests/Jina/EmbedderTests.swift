import AVFoundation
import CoreML
import Foundation
import Testing
@testable import gloss_server

/// Transparency handling (no model) — CHARACTERIZATION of a documented caveat. The reference loads
/// images via PIL `Image.open(...).convert("RGB")`, which DROPS alpha keeping the raw RGB. The Swift
/// path resizes through a premultiplied CGContext, so a fully-transparent pixel composites to BLACK
/// (0,0,0) rather than keeping its stored RGB. This is a real but narrow divergence (only RGBA inputs
/// with actual transparency; the RGB *under* a transparent pixel is semantically arbitrary), documented
/// in README "Limitations". A faithful fix is hard (CGContext is premultiplied-only; PIL works in
/// non-premultiplied space) and would risk the validated opaque-image parity — so it's documented, not
/// papered over. This test pins the behavior so any future change to it is intentional.
@Test func transparentImageAlphaHandling() throws {
    let w = 32, h = 32
    var px = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<(w * h) { px[i*4] = 200; px[i*4+1] = 100; px[i*4+2] = 50; px[i*4+3] = 0 }  // non-premultiplied RGBA, alpha=0
    let cs = CGColorSpaceCreateDeviceRGB()
    let provider = CGDataProvider(data: Data(px) as CFData)!
    let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                     space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                     provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let rgb = try GlossImagePreprocessor().resizedRGB(cg, w: w, h: h)
    let c = (16 * w + 16) * 3   // a center pixel
    // Documented behavior: premultiplied compositing -> transparent pixel becomes black (NOT PIL's (200,100,50)).
    #expect(rgb[c] == 0 && rgb[c+1] == 0 && rgb[c+2] == 0,
            "transparent-pixel RGB = (\(rgb[c]),\(rgb[c+1]),\(rgb[c+2])); expected premultiplied black (0,0,0)")
    // Opaque pixels are unaffected (the common case) — sanity-check the path is otherwise faithful.
    for i in 0..<(w * h) { px[i*4+3] = 255 }
    let cg2 = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                      space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                      provider: CGDataProvider(data: Data(px) as CFData)!, decode: nil,
                      shouldInterpolate: false, intent: .defaultIntent)!
    let rgb2 = try GlossImagePreprocessor().resizedRGB(cg2, w: w, h: h)
    #expect(rgb2[c] == 200 && rgb2[c+1] == 100 && rgb2[c+2] == 50, "opaque RGBA pixel must keep its RGB")
}

/// AVFoundation audio-file decode round-trip (no model artifacts needed): write a known 16 kHz mono
/// waveform to a .wav, decode it back via GlossAudioFile, and confirm it matches — validates the
/// `embed(audioURL:)` decode path for the exact (no-resample) case.
@Test func audioFileDecodeRoundTrip() throws {
    let sr = 16000, n = sr   // 1 s
    var wave = [Float](repeating: 0, count: n)
    for i in 0..<n { wave[i] = 0.5 * sinf(2 * .pi * 220 * Float(i) / Float(sr)) }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("jina_\(UUID().uuidString).wav")
    let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    do {   // scope the write file so it flushes/closes before we read it back
        let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n))!
        buf.frameLength = AVAudioFrameCount(n)
        for i in 0..<n { buf.floatChannelData![0][i] = wave[i] }
        try file.write(from: buf)
    }
    let decoded = try GlossAudioFile.decode16kMono(url)
    try? FileManager.default.removeItem(at: url)
    // decode reads the file's PCM directly (16kHz mono path = no resampling) -> sample VALUES are exact
    // on the overlap. (The count can differ slightly from the AVAudioFile write's buffering.)
    #expect(decoded.count > n * 9 / 10, "decoded \(decoded.count) of \(n)")
    var maxAbs: Float = 0
    for i in 0..<min(decoded.count, n) { maxAbs = max(maxAbs, abs(decoded[i] - wave[i])) }
    #expect(maxAbs < 1e-4, "16kHz .wav decode value match maxAbs=\(maxAbs)")
}

@Test func stereo48kAudioDecodeIsBoundedAndResampled() throws {
    let sourceRate = 48_000, sourceFrames = sourceRate
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-resample-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    let format = try #require(AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sourceRate),
        channels: 2,
        interleaved: false))
    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(sourceFrames)))
        buffer.frameLength = AVAudioFrameCount(sourceFrames)
        let channels = try #require(buffer.floatChannelData)
        for index in 0..<sourceFrames {
            let sample = 0.5 * sinf(2 * .pi * 220 * Float(index) / Float(sourceRate))
            channels[0][index] = sample
            channels[1][index] = sample
        }
        try file.write(from: buffer)
    }
    let decoded = try GlossAudioFile.decode16kMono(url)
    #expect((15_900...16_100).contains(decoded.count), "resampled frame count=\(decoded.count)")
    #expect(decoded.allSatisfy { $0.isFinite })
    let rms = (decoded.reduce(0.0) { $0 + Double($1 * $1) }
        / Double(decoded.count)).squareRoot()
    #expect(rms > 0.25 && rms < 0.45, "resampled stereo RMS=\(rms)")
}

@Test func thirtySecondStereoAudioResamplesWithinNativeLimit() throws {
    let sourceRate = 44_100
    let sourceFrames = sourceRate * 30
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-resample-limit-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    let format = try #require(AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sourceRate),
        channels: 2,
        interleaved: false))
    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(sourceFrames)))
        buffer.frameLength = AVAudioFrameCount(sourceFrames)
        let channels = try #require(buffer.floatChannelData)
        for index in 0..<sourceFrames {
            let sample = 0.2 * sinf(2 * .pi * 220 * Float(index) / Float(sourceRate))
            channels[0][index] = sample
            channels[1][index] = sample
        }
        try file.write(from: buffer)
    }
    let decoded = try GlossAudioFile.decode16kMono(url)
    #expect((479_900...480_000).contains(decoded.count),
            "30-second resampled frame count=\(decoded.count)")
    #expect(decoded.allSatisfy { $0.isFinite })
}

/// Converter-artifact parity tests are explicitly skipped when their source artifacts are absent.
/// The checked-in fixture tests and production-bundle verification do not depend on these paths.
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
private func hasAll(_ paths: String...) -> Bool { paths.allSatisfy(exists) }

private struct TextRef: Decodable {
    let text: String; let prompt_name: String; let prompted: String
    let token_ids: [Int32]; let embedding: [String: [Float]]
}

private func textRefs() throws -> [TextRef] {
    try JSONDecoder().decode([TextRef].self, from: Data(contentsOf: path("reference/jina/text_reference.json")))
}

@Test func bundledMelResourcesLoad() throws {
    // No model artifacts needed — just confirms the bundled mel constants load + produce output.
    let fe = try GlossMelFrontend()
    let mel = try fe.packedMel([Float](repeating: 0.1, count: 32000))
    #expect(mel.count == fe.nMels * fe.nFrames)
    #expect(mel.allSatisfy { $0.isFinite })
}

@Test func tokenizerByteExact() async throws {
    let folder = path("Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle/jina-v5-omni-small")
    let tok = try await GlossTokenizer(modelFolder: folder)
    for r in try textRefs().prefix(6) {
        #expect(tok.encode(r.prompted) == r.token_ids)
    }
}

private let textSingleBuckets = [32, 64, 128, 256, 512, 1_024, 2_048]
private let textBatchPairs = [
    (size: 64, bucket: 32),
    (size: 32, bucket: 64),
    (size: 16, bucket: 128),
    (size: 8, bucket: 256),
    (size: 4, bucket: 512),
]

@Test func textBatchPlanRoutesEveryLengthBoundaryWithoutTruncation() {
    let sparseBoundaries = [
        (tokens: 31, bucket: 32),
        (tokens: 32, bucket: 32),
        (tokens: 33, bucket: 64),
        (tokens: 63, bucket: 64),
        (tokens: 64, bucket: 64),
        (tokens: 65, bucket: 128),
        (tokens: 127, bucket: 128),
        (tokens: 128, bucket: 128),
        (tokens: 129, bucket: 256),
        (tokens: 255, bucket: 256),
        (tokens: 256, bucket: 256),
        (tokens: 257, bucket: 512),
        (tokens: 511, bucket: 512),
        (tokens: 512, bucket: 512),
    ]
    for boundary in sparseBoundaries {
        let plan = GlossTextEmbedder.embeddingPlan(
            tokenCounts: [boundary.tokens, boundary.tokens],
            buckets: textSingleBuckets,
            batchPairs: textBatchPairs)
        #expect(plan == [
            .single(index: 0, bucket: boundary.bucket),
            .single(index: 1, bucket: boundary.bucket),
        ], "\(boundary.tokens)-token rows planned as \(plan)")
    }

    let singleBoundaries = [
        (tokens: 513, bucket: 1_024),
        (tokens: 1_023, bucket: 1_024),
        (tokens: 1_024, bucket: 1_024),
        (tokens: 1_025, bucket: 2_048),
        (tokens: 2_047, bucket: 2_048),
        (tokens: 2_048, bucket: 2_048),
        (tokens: 2_049, bucket: 2_048),
    ]
    for boundary in singleBoundaries {
        let plan = GlossTextEmbedder.embeddingPlan(
            tokenCounts: [boundary.tokens, boundary.tokens],
            buckets: textSingleBuckets,
            batchPairs: textBatchPairs)
        #expect(plan == [
            .single(index: 0, bucket: boundary.bucket),
            .single(index: 1, bucket: boundary.bucket),
        ], "\(boundary.tokens)-token rows planned as \(plan)")
    }

    #expect(GlossTextEmbedder.embeddingPlan(
        tokenCounts: [], buckets: textSingleBuckets, batchPairs: textBatchPairs).isEmpty)
    #expect(GlossTextEmbedder.embeddingPlan(
        tokenCounts: [513], buckets: textSingleBuckets, batchPairs: textBatchPairs)
        == [.single(index: 0, bucket: 1_024)])
}

@Test func textBatchPlanGroupsRowsByFunctionAcrossLongRowsAndPartialChunks() {
    let short = 16

    // An oversized row does not split a dense batch group. The one-row remainder avoids a
    // mostly padded second native batch.
    let longFirst = GlossTextEmbedder.embeddingPlan(
        tokenCounts: [513] + [Int](repeating: short, count: 65),
        buckets: textSingleBuckets,
        batchPairs: textBatchPairs)
    #expect(longFirst == [
        .batch(rows: Array(1..<65), size: 64, bucket: 32),
        .single(index: 65, bucket: 32),
        .single(index: 0, bucket: 1_024),
    ])

    // Rows before and after an oversized row join the same group, but its sparse remainder runs
    // on single-row functions instead of padding 58 rows.
    let longMiddle = GlossTextEmbedder.embeddingPlan(
        tokenCounts: [Int](repeating: short, count: 64)
            + [1_025]
            + [Int](repeating: short, count: 6),
        buckets: textSingleBuckets,
        batchPairs: textBatchPairs)
    #expect(longMiddle == [
        .batch(rows: Array(0..<64), size: 64, bucket: 32),
    ] + (65..<71).map { .single(index: $0, bucket: 32) } + [
        .single(index: 64, bucket: 2_048),
    ])

    let longLast = GlossTextEmbedder.embeddingPlan(
        tokenCounts: [Int](repeating: short, count: 64) + [2_049],
        buckets: textSingleBuckets,
        batchPairs: textBatchPairs)
    #expect(longLast == [
        .batch(rows: Array(0..<64), size: 64, bucket: 32),
        .single(index: 64, bucket: 2_048),
    ])

    // Interleaved short rows still share a planning group, but three real rows are not enough to
    // amortize a 64-row native call.
    let interspersed = GlossTextEmbedder.embeddingPlan(
        tokenCounts: [short, 513, short, 1_025, short],
        buckets: textSingleBuckets,
        batchPairs: textBatchPairs)
    #expect(interspersed == [
        .single(index: 0, bucket: 32),
        .single(index: 2, bucket: 32),
        .single(index: 4, bucket: 32),
        .single(index: 1, bucket: 1_024),
        .single(index: 3, bucket: 2_048),
    ])
}

@Test func textBatchPlanSupportsLegacyFixedSizeAndNoBatchFallback() {
    let legacyPairs = [
        (size: 4, bucket: 32),
        (size: 4, bucket: 64),
        (size: 4, bucket: 128),
    ]
    #expect(GlossTextEmbedder.embeddingPlan(
        tokenCounts: [70, 70, 70],
        buckets: textSingleBuckets,
        batchPairs: legacyPairs
    ) == [
        .single(index: 0, bucket: 128),
        .single(index: 1, bucket: 128),
        .single(index: 2, bucket: 128),
    ])
    #expect(GlossTextEmbedder.embeddingPlan(
        tokenCounts: [70, 129, 70],
        buckets: textSingleBuckets,
        batchPairs: legacyPairs
    ) == [
        .single(index: 0, bucket: 128),
        .single(index: 2, bucket: 128),
        .single(index: 1, bucket: 256),
    ])
    #expect(GlossTextEmbedder.embeddingPlan(
        tokenCounts: [20, 513],
        buckets: textSingleBuckets,
        batchPairs: []
    ) == [
        .single(index: 0, bucket: 32),
        .single(index: 1, bucket: 1_024),
    ])
}

@Test(.enabled(if: hasAll("artifacts/coreml/text_multifunc.mlpackage", "artifacts/hf/jina-v5-omni-small"), "Requires converter text artifacts"))
func textEmbeddingParity() async throws {
    guard exists("artifacts/coreml/text_multifunc.mlpackage") else { return }
    let embedder = try await GlossTextEmbedder(
        multiFunctionModelURL: path("artifacts/coreml/text_multifunc.mlpackage"),
        tokenizerFolder: path("artifacts/hf/jina-v5-omni-small"))
    for r in try textRefs().prefix(5) {
        let prompt: GlossTextEmbedder.Prompt = r.prompt_name == "query" ? .query : .document
        let emb = try embedder.embed(r.text, prompt: prompt)
        let c = cosine(emb, r.embedding["1024"]!)
        #expect(c > 0.999, "text \"\(r.text)\" cos=\(c)")
    }
}

@Test(.enabled(if: hasAll("artifacts/coreml/audio_tower_masked_multifunc.mlpackage", "artifacts/coreml/embed_multifunc.mlpackage", "reference/audio_swift/manifest_offbucket.json"), "Requires converter audio artifacts"))
func maskedAudioArbitraryLength() async throws {
    guard exists("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
          exists("artifacts/coreml/embed_multifunc.mlpackage"),
          exists("reference/audio_swift/manifest_offbucket.json") else { return }
    struct Off: Decodable { let tag: String }
    let offs = try JSONDecoder().decode([Off].self, from: Data(contentsOf: path("reference/audio_swift/manifest_offbucket.json")))
    let embedder = try GlossAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        tokens: .jinaV5OmniSmallAudio)
    func loadF32(_ rel: String) throws -> [Float] {
        try Data(contentsOf: path(rel)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    // Off-bucket (partial-chunk) lengths are where truncate-real fails and the masked encoder must
    // match the reference — exercise 3s/5s explicitly.
    for o in offs {
        let wave = try loadF32("reference/audio_swift/off_\(o.tag)_wave.f32")
        let ref = try loadF32("reference/audio_swift/off_\(o.tag)_emb.f32")
        let c = cosine(try embedder.embed(wave, prompt: .document), ref)
        #expect(c > 0.999, "masked audio \(o.tag) cos=\(c)")
    }
}

/// Full-ANE deployment: force BOTH the encoder AND the decoder onto the Neural Engine (the
/// `decoderUnits` knob, alongside `encoderUnits`). Regression-guards that the full-ANE path actually
/// RUNS end-to-end on the ANE (no shape/compile error), is unit-norm, and stays above the model's
/// bf16 audio floor vs the fp32 reference (documented full-ANE end-to-end ≈0.9964; the 0.99 guard
/// below leaves margin for ANE fp16 run variance). Lowest-power, GPU-free mode; slower than the
/// hybrid default, but the recovered end-to-end cos stays at/above native bf16 precision.
@Test(.enabled(if: hasAll("artifacts/coreml/audio_tower_masked_multifunc.mlpackage", "artifacts/coreml/embed_multifunc.mlpackage", "reference/audio_swift/manifest_offbucket.json"), "Requires converter audio artifacts"))
func fullANEDeploymentRuns() throws {
    guard exists("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
          exists("artifacts/coreml/embed_multifunc.mlpackage"),
          exists("reference/audio_swift/manifest_offbucket.json") else { return }
    struct Off: Decodable { let tag: String }
    let offs = try JSONDecoder().decode([Off].self, from: Data(contentsOf: path("reference/audio_swift/manifest_offbucket.json")))
    func loadF32(_ rel: String) throws -> [Float] {
        try Data(contentsOf: path(rel)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let fullANE = try GlossAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        tokens: .jinaV5OmniSmallAudio,
        encoderUnits: .cpuAndNeuralEngine, decoderUnits: .cpuAndNeuralEngine)
    for o in offs {
        let wave = try loadF32("reference/audio_swift/off_\(o.tag)_wave.f32")
        let ref = try loadF32("reference/audio_swift/off_\(o.tag)_emb.f32")
        let emb = try fullANE.embed(wave, prompt: .document)
        let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 1e-3, "full-ANE \(o.tag) not L2-normalized (norm=\(norm))")
        // above the bf16 audio floor (0.994951); use 0.99 as the regression guard (ANE fp16 + run variance)
        let c = cosine(emb, ref)
        #expect(c > 0.99, "full-ANE audio \(o.tag) below floor (cos=\(c))")
    }
}

/// Full-ANE for the VISION path — the harder, slower encoder (key-mask masked ViT; ANE features drop
/// to ~0.9958 over 24 layers, which the decoder pool + L2 recover to ~0.9995 end-to-end). Guards that
/// the vision full-ANE pipeline (ViT + decoder both on the ANE via `encoderUnits`+`decoderUnits`) runs
/// end-to-end, is unit-norm, and stays above the vision bf16 floor (0.993308) vs the fp32 reference.
/// One grid only (the ANE ViT compile is expensive) — runs concurrently with the audio full-ANE test.
@Test(.enabled(if: hasAll("artifacts/coreml/vision_tower_masked_multifunc.mlpackage", "artifacts/coreml/embed_multifunc.mlpackage", "reference/vision_swift/manifest.json"), "Requires converter vision artifacts"))
func fullANEVisionRuns() throws {
    guard exists("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
          exists("artifacts/coreml/embed_multifunc.mlpackage"),
          exists("reference/vision_swift/manifest.json") else { return }
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    guard let g = grids.first else { return }
    let fullANE = try GlossImageEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        resourcesDir: path("reference/vision_swift"),
        tokens: .jinaV5OmniSmallImage,
        encoderUnits: .cpuAndNeuralEngine, decoderUnits: .cpuAndNeuralEngine)
    let emb = try fullANE.embed(rgb: try u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16, prompt: .document)
    let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
    #expect(abs(norm - 1.0) < 1e-3, "full-ANE vision \(g.tag) not L2-normalized (norm=\(norm))")
    let c = cosine(emb, try f32("g\(g.tag)_emb.f32"))
    #expect(c > 0.99, "full-ANE vision \(g.tag) below floor (cos=\(c))")
}

@Test(.enabled(if: hasAll("reference/vision_swift/manifest.json", "reference/vision_swift/pos_embed_table.f32"), "Requires converter vision references"))
func visionPositionAndPatchifyPort() throws {
    // Cheap (no Core ML): the bilinear pos_embeds + 2D RoPE + patchify match the golden exports.
    guard exists("reference/vision_swift/manifest.json"),
          exists("reference/vision_swift/pos_embed_table.f32") else { return }
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    func maxAbs(_ a: [Float], _ b: [Float]) -> Float { var m: Float = 0; for i in 0..<min(a.count, b.count) { m = Swift.max(m, abs(a[i] - b[i])) }; return m }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    let positions = try VisionPositions(metaURL: vp("meta.json"), posTableURL: vp("pos_embed_table.f32"), invFreqURL: vp("rope_inv_freq.f32"))
    let prep = GlossImagePreprocessor()
    for g in grids {
        let (pe, cv, sv, _) = positions.compute(gh: g.gh, gw: g.gw)
        #expect(maxAbs(pe, try f32("g\(g.tag)_pos_embeds.f32")) < 1e-4)
        #expect(maxAbs(cv, try f32("g\(g.tag)_rope_cos.f32")) < 1e-4)
        #expect(maxAbs(sv, try f32("g\(g.tag)_rope_sin.f32")) < 1e-4)
        let (pv, _, _) = try prep.pixelValues(rgb: try u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16)
        #expect(maxAbs(pv, try f32("g\(g.tag)_pixel_values.f32")) < 1e-4)
    }
}

@Test(.enabled(if: hasAll("artifacts/coreml/vision_tower_masked_multifunc.mlpackage", "artifacts/coreml/embed_multifunc.mlpackage", "reference/vision_swift/manifest.json"), "Requires converter vision artifacts"))
func visionMaskedArbitraryResolution() throws {
    guard exists("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
          exists("artifacts/coreml/embed_multifunc.mlpackage"),
          exists("reference/vision_swift/manifest.json") else { return }
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    // exercises the resourcesDir convenience init (vision_swift holds meta/pos_embed_table/rope_inv_freq)
    let embedder = try GlossImageEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        resourcesDir: path("reference/vision_swift"),
        tokens: .jinaV5OmniSmallImage)
    // Non-square grids (40x30/30x40) are the cases the fixed-square MVP could not handle.
    for g in grids {
        let c = cosine(try embedder.embed(rgb: try u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16, prompt: .document), try f32("g\(g.tag)_emb.f32"))
        #expect(c > 0.999, "vision \(g.tag) cos=\(c)")
    }
}

@Test(.enabled(if: hasAll("artifacts/coreml/vision_tower_video_multifunc.mlpackage", "reference/video_swift/manifest.json", "reference/vision_swift/meta.json"), "Requires converter video artifacts"))
func videoOnDevicePath() throws {
    guard exists("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
          exists("reference/video_swift/manifest.json"),
          exists("reference/vision_swift/meta.json") else { return }
    struct Case: Decodable { let tag: String; let nframes: Int; let fsize: Int; let t: Int; let gh: Int; let gw: Int }
    func vd(_ r: String) -> URL { path("reference/video_swift/\(r)") }
    func rs(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ u: URL) throws -> [Float] { try Data(contentsOf: u).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ u: URL) throws -> [UInt8] { try Data(contentsOf: u).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: vd("manifest.json")))
    let embedder = try GlossVideoEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        metaURL: rs("meta.json"), posTableURL: rs("pos_embed_table.f32"), invFreqURL: rs("rope_inv_freq.f32"),
        tokens: .jinaV5OmniSmallVideo)
    for c in cases {
        let ref = try f32(vd("v\(c.tag)_emb.f32"))
        // full path from raw frames (frame-patchify + block-diagonal ViT + decoder)
        let raw = try u8(vd("v\(c.tag)_frames.u8")); let fb = c.fsize * c.fsize * 3
        let frames = (0..<c.nframes).map { Array(raw[$0 * fb ..< ($0 + 1) * fb]) }
        #expect(cosine(try embedder.embed(frames: frames, h: c.fsize, w: c.fsize, prompt: .document), ref) > 0.999, "video \(c.tag)")
    }
}

/// End-to-end `embed(videoURL:)`: a synthetic .mp4 -> adaptive extraction -> real ViT + decoder.
/// The default serving profile must equal its manually composed frame path.
@Test(.enabled(if: hasAll("artifacts/coreml/vision_tower_video_multifunc.mlpackage", "reference/vision_swift/meta.json"), "Requires converter video artifacts"))
func videoURLEndToEnd() throws {
    guard exists("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
          exists("reference/vision_swift/meta.json") else { return }
    guard let url = makeSyntheticMP4(frames: 12, size: 256) else { return }   // skip if no encoder
    defer { try? FileManager.default.removeItem(at: url) }
    func rs(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    let embedder = try GlossVideoEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        metaURL: rs("meta.json"), posTableURL: rs("pos_embed_table.f32"), invFreqURL: rs("rope_inv_freq.f32"),
        tokens: .jinaV5OmniSmallVideo)
    let emb = try embedder.embed(videoURL: url)
    #expect(emb.count == 1024)
    let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
    #expect(abs(norm - 1.0) < 1e-3, "embed(videoURL:) not L2-normalized (norm=\(norm))")
    // Determinism + equivalence to the manual compose (extractFrames + embed(frames:)).
    let input = try GlossVideoFile.extractFrames(url, maxPatches: embedder.encoder.patchBuckets.last!,
                                                 preprocessor: embedder.preprocessor)
    let manual = try embedder.embed(
        frames: input.frames, h: input.h, w: input.w)
    #expect(cosine(emb, manual) > 0.99999, "embed(videoURL:) != manual compose")
}

@Test(.enabled(if: hasAll("artifacts/coreml/audio_tower_masked_multifunc.mlpackage", "reference/audio_swift/manifest_offbucket.json"), "Requires converter audio artifacts"))
func concurrentEmbedIsSafeAndConsistent() throws {
    // A SHARED masked-audio embedder hit from many threads at once must not race on its lazy model
    // caches and must return identical, correct embeddings. (Exercises the NSLock cache guards.)
    guard exists("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
          exists("reference/audio_swift/manifest_offbucket.json") else { return }
    let embedder = try GlossAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        tokens: .jinaV5OmniSmallAudio)
    func load(_ tag: String) throws -> [Float] {
        try Data(contentsOf: path("reference/audio_swift/off_\(tag)_wave.f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    // mix of durations -> different frame + S buckets loaded concurrently (max cache contention)
    let waves = [try load("3s"), try load("6s"), try load("20s")]
    let serial = try waves.map { try embedder.embed($0) }
    let results = NSMutableArray(); let rlock = NSLock()
    DispatchQueue.concurrentPerform(iterations: 24) { i in
        if let e = try? embedder.embed(waves[i % waves.count]) { rlock.lock(); results.add(e); rlock.unlock() }
    }
    #expect(results.count == 24, "some concurrent embeds failed (\(results.count)/24)")
    for case let e as [Float] in results {
        // every concurrent result must match one of the serial baselines exactly (no torn cache)
        #expect(serial.contains { cosine($0, e) > 0.99999 }, "concurrent embed diverged")
    }
}

/// Make a tiny synthetic H.264 .mp4 (distinct gray per frame). Returns nil if encoding is
/// unavailable (headless/CI) so the round-trip test skips rather than flaking.
private func makeSyntheticMP4(frames: Int, size: Int) -> URL? {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("jina_\(UUID().uuidString).mp4")
    guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
    let settings: [String: Any] = [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: size, AVVideoHeightKey: size]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
    guard writer.canAdd(input) else { return nil }
    writer.add(input)
    guard writer.startWriting() else { return nil }
    writer.startSession(atSourceTime: .zero)
    for i in 0..<frames {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32ARGB, nil, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        let g = Int32(20 + i * (200 / max(frames, 1)))
        for y in 0..<size { memset(CVPixelBufferGetBaseAddress(buf)! + y * CVPixelBufferGetBytesPerRow(buf), g, size * 4) }
        CVPixelBufferUnlockBaseAddress(buf, [])
        while !input.isReadyForMoreMediaData { usleep(2000) }
        _ = adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 10))
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    return writer.status == .completed ? url : nil
}

/// Video file decode: error cases (always) + a synthetic .mp4 frame-extraction round-trip (skips if
/// the platform can't encode). Validates the embed(videoURL:) extractor mechanics.
@Test func videoFileExtraction() throws {
    let prep = GlossImagePreprocessor()
    // odd / empty frame counts and a bad URL must throw, not crash
    #expect(throws: GlossVideoFile.DecodeError.self) {
        _ = try GlossVideoFile.extractSquareFrames(URL(fileURLWithPath: "/nonexistent.mp4"), count: 4, size: 64, preprocessor: prep)
    }
    #expect(throws: GlossVideoFile.DecodeError.self) {
        _ = try GlossVideoFile.extractSquareFrames(FileManager.default.temporaryDirectory, count: 3, size: 64, preprocessor: prep)
    }
    guard let url = makeSyntheticMP4(frames: 6, size: 64) else { return }   // skip if no encoder
    defer { try? FileManager.default.removeItem(at: url) }
    let frames = try GlossVideoFile.extractSquareFrames(url, count: 4, size: 64, preprocessor: prep)
    #expect(frames.count == 4)
    #expect(frames.allSatisfy { $0.count == 64 * 64 * 3 })
    // distinct gray per source frame -> extracted frames should not all be identical
    #expect(Set(frames.map { $0.first ?? 0 }).count > 1, "extracted frames are all identical")
    let adaptive = try GlossVideoFile.extractFrames(url, maxPatches: 2_048, preprocessor: prep)
    #expect(adaptive.frames.count == 4)
    #expect(adaptive.frames.allSatisfy { $0.count == adaptive.h * adaptive.w * 3 })
    #expect(Set(adaptive.frames.map { $0.first ?? 0 }).count > 1)
}

@Test func videoServingPlanFollowsSourceSamplingWithinNativeBudget() throws {
    let prep = GlossImagePreprocessor()
    let short = try GlossVideoFile.plan(totalFrames: 8, sourceFPS: 8, height: 64, width: 64,
                                         maxPatches: 2_048, preprocessor: prep)
    #expect(short.frameIndices == [0, 2, 5, 7])
    #expect(short.height == 256 && short.width == 256)
    #expect(short.patches == 512)

    let odd = try GlossVideoFile.plan(totalFrames: 3, sourceFPS: 3, height: 128, width: 128,
                                       maxPatches: 2_048, preprocessor: prep)
    #expect(odd.frameIndices == [0, 1, 2, 2], "odd source samples repeat the final frame")

    let landscape = try GlossVideoFile.plan(totalFrames: 40, sourceFPS: 10, height: 1_080, width: 1_920,
                                             maxPatches: 2_048, preprocessor: prep)
    #expect(landscape.frameIndices == [0, 6, 11, 17, 22, 28, 33, 39])
    #expect(landscape.width > landscape.height, "video geometry must preserve orientation")
    #expect(landscape.patches <= 2_048)

    let long = try GlossVideoFile.plan(totalFrames: 900, sourceFPS: 30, height: 1_080, width: 1_920,
                                        maxPatches: 2_048, preprocessor: prep)
    #expect(long.frameIndices.count == 32, "native frame cap must cover the whole clip")
    #expect(long.frameIndices.first == 0 && long.frameIndices.last == 899)
    #expect(long.patches <= 2_048)
    #expect(long.width > long.height)
    #expect(throws: GlossVideoFile.DecodeError.self) {
        _ = try GlossVideoFile.plan(totalFrames: 1_830, sourceFPS: 30, height: 1080, width: 1920,
                                    maxPatches: 2_048, preprocessor: prep)
    }
}

/// Bad input must THROW (recoverable), not crash the host app — no model artifacts needed.
@Test func badInputThrowsNotCrashes() throws {
    let mel = try GlossMelFrontend()   // bundled constants
    #expect(throws: GlossMelFrontend.MelError.self) {
        _ = try mel.packedMel([Float](repeating: 0, count: 50))   // < FFT half-window
    }
    _ = try mel.packedMel([Float](repeating: 0, count: 8000))     // valid clip: does not throw
    let prep = GlossImagePreprocessor()
    #expect(throws: GlossImagePreprocessor.ImageError.self) {
        _ = try prep.videoPixelValues(frames: [[UInt8](repeating: 0, count: 128 * 128 * 3)], h: 128, w: 128)  // odd frame count
    }
}

@Test(.enabled(if: hasAll("artifacts/coreml/text_multifunc.mlpackage", "artifacts/hf/jina-v5-omni-small"), "Requires converter text artifacts"))
func matryoshkaDimsAreUnitNorm() async throws {
    guard exists("artifacts/coreml/text_multifunc.mlpackage") else { return }
    let embedder = try await GlossTextEmbedder(
        multiFunctionModelURL: path("artifacts/coreml/text_multifunc.mlpackage"),
        tokenizerFolder: path("artifacts/hf/jina-v5-omni-small"))
    for d in [64, 256, 512] {
        let emb = try embedder.embed("semantic search", prompt: .query, dim: d)
        #expect(emb.count == d)
        var n: Float = 0; for x in emb { n += x * x }
        #expect(abs(n.squareRoot() - 1.0) < 1e-4)
    }
}

/// Long-document embedding through the large text buckets (1024/2048): a ~1200-token document
/// must select bucket_2048 and embed with unit norm. Guards the larger buckets on the native
/// 32k bundle. Artifact-gated.
@Test(.enabled(if: exists("artifacts/JinaV5OmniSmall.w8a16.bundle/manifest.json"), "Requires local full bundle under artifacts"))
func longDocumentEmbedding() async throws {
    guard exists("artifacts/JinaV5OmniSmall.w8a16.bundle/manifest.json") else { return }
    let bundle = try GlossModelBundle(url: path("artifacts/JinaV5OmniSmall.w8a16.bundle"))
    guard let text = bundle.manifest.text else { return }
    let embedder = try await GlossTextEmbedder(
        multiFunctionModelURL: bundle.resolve(text.model),
        tokenizerFolder: bundle.resolve(text.tokenizer),
        buckets: text.buckets,
        prompts: GlossTextEmbedder.PromptStrings(
            query: bundle.manifest.prompts?.query ?? "",
            document: bundle.manifest.prompts?.document ?? ""),
        padTokenID: bundle.manifest.tokens.padID,
        batchSize: text.batch?.size,
        batchBuckets: text.batch?.buckets ?? [],
        batchSizes: text.batch?.sizes ?? [])
    // ~1200 tokens: a repeated sentence until the tokenizer length is comfortably past 1024.
    let sentence = "The quick brown fox jumps over the lazy dog near the riverbank every morning. "
    var doc = ""
    while doc.count < 9000 { doc += sentence }   // ≈1200 tokens
    let emb = try embedder.embed(doc, prompt: .document)
    #expect(emb.count == bundle.manifest.embeddingDimension)
    var n: Float = 0
    for x in emb { n += x * x }
    #expect(abs(n.squareRoot() - 1.0) < 1e-3, "long-doc embedding not unit-norm (norm=\(n.squareRoot()))")
}

/// Mixed bulk requests must preserve the single-row semantics of rows that exceed the largest
/// batch bucket. This exercises both a 1024/2048 single bucket and true >2048 keep-first overflow,
/// with task-prefix selection and Matryoshka truncation applied identically on both paths.
@Test(.enabled(if: exists("artifacts/JinaV5OmniSmall.w8a16.bundle/manifest.json"), "Requires local full bundle under artifacts"))
func longRowsInTextBatchMatchSingleSemantics() async throws {
    guard exists("artifacts/JinaV5OmniSmall.w8a16.bundle/manifest.json") else { return }
    let bundle = try GlossModelBundle(url: path("artifacts/JinaV5OmniSmall.w8a16.bundle"))
    guard let text = bundle.manifest.text, let batch = text.batch else { return }
    let taskPrompt = GlossTextEmbedder.PromptStrings(
        query: "Find supporting evidence: ", document: "Supporting evidence: ")
    let embedder = try await GlossTextEmbedder(
        multiFunctionModelURL: bundle.resolve(text.model),
        tokenizerFolder: bundle.resolve(text.tokenizer),
        buckets: text.buckets,
        prompts: .init(
            query: bundle.manifest.prompts?.query ?? "",
            document: bundle.manifest.prompts?.document ?? ""),
        padTokenID: bundle.manifest.tokens.padID,
        batchSize: batch.size,
        batchBuckets: batch.buckets,
        batchSizes: batch.sizes ?? [],
        taskPrompts: ["qa": taskPrompt])

    func makeText(atLeastTokenCount target: Int) -> String {
        var words = [String]()
        while true {
            let base = words.count
            for i in 0..<64 {
                words.append("evidence\(base + i) relates concept\((base + i) % 17)")
            }
            let candidate = words.joined(separator: " ")
            if embedder.tokenizer.encode(taskPrompt.query + candidate).count >= target {
                return candidate
            }
        }
    }

    let long = makeText(atLeastTokenCount: 900)
    let overflow = makeText(atLeastTokenCount: 2_100)
    let longCount = embedder.tokenizer.encode(taskPrompt.query + long).count
    let overflowCount = embedder.tokenizer.encode(taskPrompt.query + overflow).count
    #expect(longCount > 512 && longCount <= 2_048, "long probe has \(longCount) tokens")
    #expect(overflowCount > 2_048, "overflow probe has \(overflowCount) tokens")

    let texts = ["short alpha", long, "short beta", overflow, "short gamma"]
    let bulk = try embedder.embed(texts: texts, prompt: .query, task: "qa", dim: 128)
    #expect(bulk.count == texts.count)
    for (index, text) in texts.enumerated() {
        let single = try embedder.embed(text, prompt: .query, task: "qa", dim: 128)
        #expect(bulk[index].count == 128)
        #expect(single.count == 128)
        #expect(bulk[index].allSatisfy { $0.isFinite }, "bulk row \(index) contains non-finite values")
        #expect(single.allSatisfy { $0.isFinite }, "single row \(index) contains non-finite values")
        let bulkNorm = bulk[index].reduce(Float.zero) { $0 + $1 * $1 }.squareRoot()
        let singleNorm = single.reduce(Float.zero) { $0 + $1 * $1 }.squareRoot()
        #expect(abs(bulkNorm - 1) < 1e-3, "bulk row \(index) norm=\(bulkNorm)")
        #expect(abs(singleNorm - 1) < 1e-3, "single row \(index) norm=\(singleNorm)")
        let similarity = cosine(bulk[index], single)
        #expect(similarity.isFinite, "row \(index) cosine is non-finite")
        #expect(similarity > 0.999, "row \(index) bulk/single cosine=\(similarity)")
    }

    #expect(try embedder.embed(texts: [], prompt: .query, task: "qa", dim: 128).isEmpty)
    let one = try embedder.embed(texts: [long], prompt: .query, task: "qa", dim: 128)
    #expect(one.count == 1)
    let longSingle = try embedder.embed(long, prompt: .query, task: "qa", dim: 128)
    let oneSimilarity = cosine(one[0], longSingle)
    #expect(oneSimilarity.isFinite && oneSimilarity > 0.999)
}
