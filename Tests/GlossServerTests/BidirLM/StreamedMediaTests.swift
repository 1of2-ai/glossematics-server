import Foundation
import Testing
@testable import gloss_server

private func mediaFields() -> [String: Any] {
    func tower(_ family: String, keys: [Int]) -> [String: Any] {
        let names = StreamedMediaManifest.expectedStages(family, keys: keys)
        let model = family == "vision" ? "BidirLMOmniVisionANE.mlmodelc" : "BidirLMOmniAudioANE.mlmodelc"
        var t: [String: Any] = ["keys": keys, "stages": Dictionary(uniqueKeysWithValues: names.map {
            ($0, ["compiled": model, "function": "\(family)_ane_\($0)"]) })]
        if family == "vision" { t["deepstack_blocks"] = [8, 16] }
        return t
    }
    return ["format": StreamedMediaManifest.format, "revision": BidirLMContract.revision,
            "weight_precision": "w8", "weight_tensors": "complete_release_shared", "chunk": 512,
            "key_block": 4096, "layers": 24, "hidden": 1024, "heads": 16, "head_dim": 64, "out_dim": 2048,
            "attention_summary": "max_sum_normalized_output",
            "vision": tower("vision", keys: StreamedMediaManifest.visionKeys),
            "audio": tower("audio", keys: StreamedMediaManifest.audioKeys)]
}

private func decode(_ fields: [String: Any]) throws -> StreamedMediaManifest {
    try JSONDecoder().decode(StreamedMediaManifest.self, from: JSONSerialization.data(withJSONObject: fields))
}

@Test func stagedMediaManifestAcceptsExactStagesAndRejectsDrift() throws {
    let media = try decode(mediaFields())
    try media.validate(revision: BidirLMContract.revision)
    #expect(media.vision.stages.count == 32 && media.audio.stages.count == 33 && media.stageCount == 65)
    #expect(media.functions.count == 65)
    #expect(media.vision.stages["boundary_3"]?.functionName == "vision_ane_boundary_3")
    // Residency: towers and every bucket fit beside the 34 pinned language programs (budget 64).
    let sets = Dictionary(uniqueKeysWithValues: media.programSets.map { ($0.0, $0.1.count) })
    #expect(sets["vision.tower"] == 28 && sets["audio.tower"] == 25 && sets["audio.front"] == 1)
    #expect(sets["vision.attn.4096"] == 1 && sets["audio.attn.32768"] == 2 && sets["audio.attn.4096"] == 1)
    #expect(34 + sets["vision.tower"]! + sets["vision.attn.4096"]! <= 64)
    #expect(34 + sets["audio.tower"]! + sets["audio.front"]! + sets["audio.attn.32768"]! <= 64)
    #expect(StreamedMediaManifest.attentionStages(keys: 16384) == ["summary_b4096", "merge_n4"])

    var fields = mediaFields()
    fields["weight_precision"] = "fp16"
    #expect(throws: (any Error).self) { try decode(fields).validate(revision: BidirLMContract.revision) }
    fields = mediaFields()
    var vision = fields["vision"] as! [String: Any]
    var stages = vision["stages"] as! [String: Any]
    stages.removeValue(forKey: "merger_deepstack_1")
    vision["stages"] = stages
    fields["vision"] = vision
    #expect(throws: (any Error).self) { try decode(fields).validate(revision: BidirLMContract.revision) }
    fields = mediaFields()
    var audio = fields["audio"] as! [String: Any]
    var audioStages = audio["stages"] as! [String: Any]
    audioStages["front"] = ["compiled": "../escape.mlmodelc", "function": "audio_ane_front"]
    audio["stages"] = audioStages
    fields["audio"] = audio
    #expect(throws: (any Error).self) { try decode(fields).validate(revision: BidirLMContract.revision) }
    fields = mediaFields()
    fields["weight_tensors"] = "source_fp16"
    #expect(throws: (any Error).self) { try decode(fields).validate(revision: BidirLMContract.revision) }
    #expect(throws: (any Error).self) { try decode(mediaFields()).validate(revision: "other") }
}
