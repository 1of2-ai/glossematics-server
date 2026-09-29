import Foundation
import Testing
@testable import gloss_server

/// Full-model gates for a real BidirLM bundle. Enabled with `GLOSS_BIDIRLM_BUNDLE`; the compute
/// mode comes from `GLOSS_BIDIRLM_COMPUTE` (ane|gpu|cpu, default ane). Goldens are independent
/// FP32 evaluations of the untouched source model (`reference/bidirlm/text_reference.json`).
private struct TextReference: Decodable {
    struct TokenizerCase: Decodable { let text: String; let ids: [Int32] }
    struct Case: Decodable { let name: String; let text: String?; let tokens: [Int32]; let embedding: [Float] }
    let tokenizer: [TokenizerCase]
    let cases: [Case]
}

private func referenceURL(_ name: String) -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("reference/bidirlm").appendingPathComponent(name)
}

private func productionBundle() -> URL? {
    ProcessInfo.processInfo.environment["GLOSS_BIDIRLM_BUNDLE"].map { URL(fileURLWithPath: $0, isDirectory: true) }
}

private func computeMode() -> ComputeMode {
    ComputeMode(rawValue: ProcessInfo.processInfo.environment["GLOSS_BIDIRLM_COMPUTE"] ?? "ane") ?? .ane
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<min(a.count, b.count) {
        dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
    }
    return dot / (na.squareRoot() * nb.squareRoot())
}

/// Media goldens exported by `qualify_ane_media.py --export`: the exact PNG/WAV bytes a client
/// sends, the processor's token ids and tensor samples, and the FP32 source embedding.
private struct MediaReference: Decodable {
    struct Part: Decodable { let type: String; let text: String?; let png: String?; let wav: String? }
    struct Sample: Decodable { let rows: [Int]?; let frames: [Int]?; let values: [[Float]] }
    struct Case: Decodable {
        let name: String
        let parts: [Part]
        let input_ids: [Int32]
        let embedding: [Float]
        let pixel_values_sample: Sample?
        let pixel_values_sum: Double?
        let mel_frames: Int?
        let mel_sample: Sample?
    }
    let cases: [Case]
}

private func mediaReferenceAvailable() -> Bool {
    FileManager.default.fileExists(atPath: referenceURL("media_reference.json").path)
}

/// The full-model gates share the machine's Neural Engine program slots and one process log,
/// so they run one at a time.
@Suite(.serialized) struct BidirLMFullModel {
@Test(.enabled(if: productionBundle() != nil, "Set GLOSS_BIDIRLM_BUNDLE to a sealed BidirLM bundle"))
func bidirlmTextMatchesFP32Source() async throws {
    guard let url = productionBundle() else { return }
    let reference = try JSONDecoder().decode(TextReference.self, from: Data(contentsOf: referenceURL("text_reference.json")))
    let bundle = try BidirLMBundle.load(from: url)
    let tokenizer = try BidirLMTokenizer(folder: bundle.root.appendingPathComponent(bundle.manifest.tokenizer),
                                         manifest: bundle.manifest, instances: 1)
    for item in reference.tokenizer {
        #expect(tokenizer.templated(item.text) == item.ids, "tokenization differs for \(item.text.prefix(40))")
    }
    let mode = computeMode()
    let backend = BidirLMBackend(bundle: bundle, mode: mode, tokenizer: tokenizer)
    try await backend.prepare(verifyPlacement: mode == .ane)
    if mode == .ane {
        let placement = try #require(await backend.placement)
        #expect(placement.strictANE, "placement: \(placement.firstProblem ?? "")")
    }
    let maxTokens = Int(ProcessInfo.processInfo.environment["GLOSS_BIDIRLM_MAX_TOKENS"] ?? "") ?? 32_768

    // Short and medium texts are packed into one execution each; long texts run alone.
    let groups: [[TextReference.Case]] = [
        reference.cases.filter { $0.name.hasPrefix("short_") },
        reference.cases.filter { $0.name.hasPrefix("medium_") },
    ] + reference.cases.filter { $0.name.hasPrefix("single_") && $0.tokens.count <= maxTokens }.map { [$0] }
    for group in groups where !group.isEmpty {
        for item in group where item.text != nil {
            #expect(tokenizer.templated(item.text!) == item.tokens, "\(item.name) tokenization")
        }
        let started = Date()
        let vectors = try await backend.embed(tokenRows: group.map(\.tokens))
        let seconds = Date().timeIntervalSince(started)
        for (item, vector) in zip(group, vectors) {
            let c = cosine(vector, item.embedding)
            print("\(mode.rawValue) \(item.name) tokens=\(item.tokens.count) cosine=\(c) seconds=\(seconds)")
            #expect(c >= 0.995, "\(item.name): cosine \(c)")
        }
    }
}

@Test(.enabled(if: mediaReferenceAvailable(), "run qualify_ane_media.py --export first"))
func mediaPreprocessingMatchesTheProcessor() throws {
    let reference = try JSONDecoder().decode(MediaReference.self, from: Data(contentsOf: referenceURL("media_reference.json")))
    let vision = BidirLMManifest.Vision.reference
    let mel = try GlossMelFrontend()
    for item in reference.cases where item.parts.count == 1 {
        let part = item.parts[0]
        if let png = part.png, let sample = item.pixel_values_sample, let rows = sample.rows {
            let image = try BidirLMMediaInputs.image(Data(base64Encoded: png)!, config: vision)
            var worst: Float = 0
            for (row, values) in zip(rows, sample.values) {
                for (j, v) in values.enumerated() { worst = max(worst, abs(image.pixels[row * 1536 + j] - v)) }
            }
            let sum = image.pixels.reduce(0.0) { $0 + Double($1) }
            print("\(item.name) grid=\(image.gridH)x\(image.gridW) worst=\(worst) sum=\(sum) ref=\(item.pixel_values_sum ?? .nan)")
            #expect(image.tokens + 7 == item.input_ids.count, "\(item.name) token count")
            #expect(worst < 1e-5, "\(item.name): resampled pixels differ by \(worst)")
            #expect(abs(sum - (item.pixel_values_sum ?? 0)) < 1e-2 * Double(image.pixels.count) * 1e-4 + 1, "\(item.name) pixel sum")
        }
        if let wav = part.wav, let sample = item.mel_sample, let frames = sample.frames {
            let samples = try BidirLMMediaInputs.samples16k(Data(base64Encoded: wav)!, maximumSeconds: 3600)
            let clip = try BidirLMMediaInputs.audio(samples, frontend: mel)
            #expect(clip.frames == item.mel_frames, "\(item.name) frame count")
            var worst: Float = 0
            for (f, values) in zip(frames, sample.values) {
                for (m, v) in values.enumerated() { worst = max(worst, abs(clip.mel[m * clip.frames + f] - v)) }
            }
            print("\(item.name) frames=\(clip.frames) worst mel diff=\(worst)")
            // The reference used float samples; the WAV carries 16-bit PCM of the same signal.
            #expect(worst < 2e-3, "\(item.name): log-mel differs by \(worst)")
        }
    }
}

@Test(.enabled(if: productionBundle() != nil && mediaReferenceAvailable(), "Set GLOSS_BIDIRLM_BUNDLE"))
func bidirlmMediaMatchesFP32Source() async throws {
    guard let url = productionBundle() else { return }
    let reference = try JSONDecoder().decode(MediaReference.self, from: Data(contentsOf: referenceURL("media_reference.json")))
    let bundle = try BidirLMBundle.load(from: url)
    let tokenizer = try BidirLMTokenizer(folder: bundle.root.appendingPathComponent(bundle.manifest.tokenizer),
                                         manifest: bundle.manifest, instances: 1)
    let mode = computeMode()
    let backend = BidirLMBackend(bundle: bundle, mode: mode, tokenizer: tokenizer)
    try await backend.prepare(verifyPlacement: false)
    let mel = try GlossMelFrontend()
    for item in reference.cases {
        var parts = [MediaSequenceBuilder.Part]()
        for part in item.parts {
            if let text = part.text { parts.append(.text(text)) }
            if let png = part.png {
                parts.append(.media(.image(try BidirLMMediaInputs.image(Data(base64Encoded: png)!, config: bundle.manifest.vision!))))
            }
            if let wav = part.wav {
                let samples = try BidirLMMediaInputs.samples16k(Data(base64Encoded: wav)!, maximumSeconds: 3600)
                parts.append(.media(.audio(try BidirLMMediaInputs.audio(samples, frontend: mel))))
            }
        }
        let pending = try MediaSequenceBuilder.build(parts, tokenizer: tokenizer, tokens: bundle.manifest.mediaTokens!, kind: "message")
        #expect(pending.ids == item.input_ids, "\(item.name) token ids")
        var features = [BidirLMMediaEncoder.Features]()
        for span in pending.spans {
            let run = try await backend.startTower(span.input)
            var f: BidirLMMediaEncoder.Features?
            while f == nil { f = try await backend.step(run) }
            features.append(f!)
        }
        let sequence = try pending.sequence(features: features, deepstackLayers: bundle.manifest.imageDeepstackSets)
        let vector: [Float]
        if PackPlanner.isShort(sequence.count) {
            vector = try await backend.embed(sequences: [sequence])[0]
        } else {
            let run = try await backend.startLong(sequence)
            var v: [Float]?
            while v == nil { v = try await backend.step(run) }
            vector = v!
        }
        let c = cosine(vector, item.embedding)
        print("\(mode.rawValue) \(item.name) tokens=\(sequence.count) cosine=\(c)")
        #expect(c >= 0.99, "\(item.name): cosine \(c)")
    }
}
}
