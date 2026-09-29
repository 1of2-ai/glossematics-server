import Foundation
import Testing
@testable import gloss_server

/// Image and audio embeddings against the FP32 source model (`reference/jina/fp32_reference.json`,
/// captured by `GlossematicsCoreML/python/parity/export_jina_fp32_http_refs.py`, with its media in
/// `reference/jina/media/`). Gated on `GLOSS_JINA_BUNDLE`: it embeds every image and audio case
/// with both retrieval roles on the real bundle and holds the cosine at or above the per-category
/// floors `Scripts/smoke_jina.py` uses (`FP32_FLOORS`), which are "no worse than the shipped W8A16
/// quality".
///
/// It pins the three preprocessing fixes that gate found: half-to-even smart-resize rounding
/// (`hd_1280x720`), `ceil(samples / hop)` mel frames (`odd_2.5s`), and mean stereo downmix
/// (`3s_44k_stereo`).

private let repositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

private let floors: [String: Double] = [
    "image_exact": 0.9935,
    "image_downscaled": 0.994,
    "image_upscaled": 0.9695,
    "audio_exact": 0.9965,
    "audio_resampled": 0.9615,
]

private struct MediaCase: Decodable {
    struct Role: Decodable { let embedding: [Float] }
    let name: String
    let file: String
    let path: String
    let roles: [String: Role]
}

private struct FP32Reference: Decodable {
    let format: String
    let image: [MediaCase]
    let audio: [MediaCase]
}

private func realBundle() -> URL? {
    ProcessInfo.processInfo.environment["GLOSS_JINA_BUNDLE"].map { URL(fileURLWithPath: $0, isDirectory: true) }
}

@Suite(.serialized) struct Fp32MediaReference {
    @Test(.enabled(if: realBundle() != nil, "Set GLOSS_JINA_BUNDLE to the production bundle"))
    func imagesAndAudioMeetTheFP32Floors() async throws {
        let bundle = try #require(realBundle())
        let jina = repositoryRoot.appendingPathComponent("reference/jina")
        let reference = try JSONDecoder().decode(
            FP32Reference.self, from: Data(contentsOf: jina.appendingPathComponent("fp32_reference.json")))
        #expect(reference.format == "glossematics-jina-fp32-reference-v1")
        let model = try await OmniSmall.load(from: bundle)

        var rows = [String]()
        for (kind, cases) in [("image", reference.image), ("audio", reference.audio)] {
            for mediaCase in cases {
                let bytes = try Data(contentsOf: jina.appendingPathComponent(mediaCase.file))
                let input: OmniSmall.Input = kind == "image" ? .imageData(bytes) : .audioData(bytes)
                let category = "\(kind)_\(mediaCase.path)"
                let floor = try #require(floors[category], "no floor for \(category)")
                for role in ["query", "document"] {
                    let served = role == "query"
                        ? try await model.embedQuery(input).values
                        : try await model.embedDocument(input).values
                    let expected = try #require(mediaCase.roles[role]).embedding
                    let similarity = cosine(served, expected)
                    rows.append("\(kind):\(mediaCase.name)/\(role) cosine \(String(format: "%.6f", similarity)) (floor \(floor), \(category))")
                    #expect(similarity >= floor,
                            "\(kind):\(mediaCase.name)/\(role) cosine \(similarity) is below the \(category) floor \(floor)")
                }
            }
        }
        print("FP32 media parity:\n  " + rows.joined(separator: "\n  "))
    }
}
