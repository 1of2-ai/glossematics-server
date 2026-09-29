import CoreML
import Foundation
import Testing
@testable import gloss_server

@Test func streamedManifestRejectsOldSummariesAndPartialModels() throws {
    let names = ["head", "summary_b512", "summary_b4096", "merge_n2", "merge_n4", "merge_n8"]
        + (0..<28).map { "boundary_\($0)" }
    let fields: [String: Any] = [
        "format": "bidirlm-streamed-v2", "revision": BidirLMContract.revision, "layers": 28,
        "geometry": ["chunk": 512, "key_block": 4096, "capacity": 32768],
        "key_buckets": [512, 4096, 8192, 16384, 32768], "weight_precision": "fp16", "rms_fp32": false,
        "deepstack_layers": 3, "pad_token_id": 151643, "mrope_section": [24, 20, 20],
        "embeddings": ["file": "token_embeddings.f16", "shape": [151936, 2048]],
        "pooling": "masked_mean_l2_float64_host", "attention_schedule": "summary_then_merge",
        "attention_summary": "max_sum_normalized_output",
        "native_tables": ["cos": "cos.f16", "sin": "sin.f16", "inverseFrequency": "inv_freq.f32"],
        "stages": Dictionary(uniqueKeysWithValues: names.map { ($0, ["compiled": "stages/\($0).mlmodelc"]) })]
    func manifest(_ fields: [String: Any]) throws -> StreamedTextManifest {
        try JSONDecoder().decode(StreamedTextManifest.self, from: JSONSerialization.data(withJSONObject: fields))
    }
    try manifest(fields).validate()
    for (key, value) in [("format", "bidirlm-streamed-v1" as Any), ("layers", 27),
                         ("attention_summary", "max_mean_numerator"), ("weight_precision", "int4"), ("rms_fp32", true)] {
        var changed = fields; changed[key] = value
        #expect(throws: (any Error).self) { try manifest(changed).validate() }
    }
    // The W8A16 release recipe is its own embedding space; FP16 stays separate.
    var w8 = fields; w8["weight_precision"] = "w8"
    let release = try manifest(w8)
    try release.validate()
    #expect(release.expectedSpaceID == StreamedTextManifest.spaceIDW8)
    #expect(release.expectedSpaceID != StreamedTextManifest.spaceID)
    #expect(release.expectedPrecision == .init(weights: "int8", activations: "float16"))
    #expect(try manifest(fields).expectedPrecision == .init(weights: "float16", activations: "float16"))
    // Stages may name a function of the shared multifunction artifact.
    var named = w8
    named["stages"] = Dictionary(uniqueKeysWithValues: names.map {
        ($0, ["compiled": "BidirLMOmniLanguageANE.mlmodelc", "function": "language_ane_\($0)"]) })
    let multifunction = try manifest(named)
    try multifunction.validate()
    #expect(multifunction.stages["boundary_3"]?.functionName == "language_ane_boundary_3")
    #expect(try manifest(fields).stages["head"]?.functionName == "main")
}

@Test func streamedHalfSurfaceHonorsPixelBufferStrides() throws {
    let a = try HalfSurface([1, 3, 2, 5], zeroed: true)
    #expect(a.array.dataType == .float16)
    #expect(a.array.strides.map(\.intValue) == a.strides)
    try a.withMutable { p, s in
        for h in 0..<3 { for d in 0..<2 { for t in 0..<5 { p[h * s[1] + d * s[2] + t] = Float16(100 * h + 10 * d + t) } } }
    }
    let b = try HalfSurface(a.shape)
    try b.copy(from: a.array)
    try b.withBuffer { p, s in
        #expect(p[2 * s[1] + s[2] + 4] == 214)
    }
    #expect(throws: (any Error).self) { try HalfSurface([1, 2, 0, 4]) }
}

@Test func streamedBanksPreserveIdentityLayoutsAndLayerSeparation() throws {
    let valid = (0..<64).map { $0 % 3 != 0 }
    let bank = try StreamedKVBank(keys: 64, block: 32, valid: valid, heads: 2, dim: 8)
    let next = try StreamedKVBank(keys: 64, block: 32, valid: valid, heads: 2, dim: 8)
    let k = try HalfSurface([1, 2, 8, 16]), v = try HalfSurface([1, 2, 8, 16])
    for (tensor, sign) in [(k, Float16(1)), (v, Float16(-1))] {
        try tensor.withMutable { p, s in
            for h in 0..<2 { for d in 0..<8 { for t in 0..<16 { p[h * s[1] + d * s[2] + t] = sign * Float16(h * 128 + d * 16 + t) } } }
        }
    }
    let identity = bank.keys[0][1].array
    try bank.store(start: 16, key: k, value: v)
    #expect(bank.keys[0][1].array === identity)
    try bank.keys[0][1].withBuffer { p, s in #expect(p[31 * s[1] + 7] == 255) }
    try bank.values[0][1].withBuffer { p, s in #expect(p[7 * s[1] + 31] == -255) }
    try bank.masks[1].withBuffer { p, s in
        for t in 0..<32 { #expect(p[t * s[1]] == (valid[t + 32] ? 1 : 0)) }
    }
    try next.keys[0][1].withBuffer { p, s in #expect(p[31 * s[1] + 7] == 0) }
    #expect(throws: (any Error).self) { try bank.store(start: 24, key: k, value: v) }
}

private struct StreamedFreshRequest: Decodable {
    let model: String
    let output: String
    let ids: [Int32]
    let valid: [Bool]
    let positions: [Int]
    let compute: String
}

/// Only the fresh Python verifier supplies this request. No checked-in numeric fixtures or
/// qualification reports are read. This calls the same encoder used by the server backend.
@Test(.enabled(if: ProcessInfo.processInfo.environment["GLOSS_STREAMED_REQUEST"] != nil,
               "requires a newly generated full-model verification request"))
func streamedFreshFullModel() throws {
    let requestPath = try #require(ProcessInfo.processInfo.environment["GLOSS_STREAMED_REQUEST"])
    let request = try JSONDecoder().decode(StreamedFreshRequest.self, from: Data(contentsOf: URL(fileURLWithPath: requestPath)))
    let root = URL(fileURLWithPath: request.model)
    let manifest = try JSONDecoder().decode(StreamedTextManifest.self, from: Data(contentsOf: root.appendingPathComponent("streamed.json")))
    try manifest.validate()
    try manifest.validateFiles(root: root)
    let mode = try #require(ComputeMode(rawValue: request.compute))
    let started = Date()
    func progress(_ event: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])
        data.append(10)
        try FileHandle.standardOutput.write(contentsOf: data)
    }
    try progress(["phase": "native_loading", "pid": ProcessInfo.processInfo.processIdentifier])
    let residency = ProgramResidency(root: root, mode: mode, budget: 64, verifyNeuralEngine: false)
    let encoder = try StreamedTextEncoder(residency: residency, manifest: manifest)
    let loadSeconds = Date().timeIntervalSince(started)
    try progress(["phase": "native_loaded", "seconds": loadSeconds])
    var layers: [[[Float]]] = []
    let predicting = Date()
    let run = try encoder.makeRun(LMSequence(ids: request.ids), valid: request.valid) { layer, chunks in
        let rows = try request.positions.map { position -> [Float] in
            guard (0..<request.ids.count).contains(position) else {
                throw BidirLMBundle.Failure.invalid("sample position outside sequence")
            }
            return try chunks[position / 512].withBuffer { p, s in
                (0..<2048).map { Float(p[$0 * s[1] + position % 512]) }
            }
        }
        layers.append(rows)
        try progress(["phase": "native_layer", "layer": layer, "seconds": Date().timeIntervalSince(predicting)])
    }
    while try !run.step() {}
    let result: [String: Any] = ["embedding": try run.result(), "layers": layers,
                               "load_seconds": loadSeconds, "prediction_seconds": Date().timeIntervalSince(predicting),
                               "predictions": encoder.predictions, "declined_output_backings": encoder.declinedOutputBackings,
                               "steps": run.completedSteps, "total_steps": run.totalSteps,
                               "actual_ane_execution_verified": false]
    #expect(run.completedSteps == run.totalSteps)
    #expect(layers.count == 28)
    try JSONSerialization.data(withJSONObject: result).write(to: URL(fileURLWithPath: request.output), options: .atomic)
}
