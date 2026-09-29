import XCTest
@testable import gloss_server

final class DocsPageTests: XCTestCase {
    private var context: DocsPage.Context {
        .init(
            baseURL: "http://127.0.0.1:11435",
            modelID: "BidirLM/BidirLM-Omni-2.5B-Embedding",
            dimensions: 2048,
            maxTokens: 32768,
            compute: "ane",
            modalities: ["text", "image", "audio", "message"],
            maxBatch: 2048,
            maxRequestTokens: 131_072,
            maxBodyMB: 64,
            maxTotalBodyMB: 256,
            maxQueueRequests: 128,
            maxQueueItems: 8192,
            batchWindowMS: 2,
            keepWarmSeconds: 60,
            maxConnections: 256,
            ioTimeoutSeconds: 30,
            shutdownGraceSeconds: 15,
            accessLogMode: "errors",
            spaceID: "BidirLM/BidirLM-Omni-2.5B-Embedding:447a6e31:2048:w8a16:ane-chunked-mean-v1")
    }

    func testResourceIsPackagedIntoTheBundle() { XCTAssertTrue(DocsPage.templateAvailable) }
    func testRenderSubstitutesEveryToken() {
        let html = DocsPage.render(context) ?? ""
        XCTAssertFalse(html.contains("{{"))
        XCTAssertTrue(html.contains("2048d"))
        XCTAssertTrue(html.contains("every 60s"))
        XCTAssertTrue(html.contains("input_audio"))
        XCTAssertTrue(html.contains("unsupported_modality"))
        XCTAssertTrue(html.contains("cpuAndNeuralEngine"))
        XCTAssertTrue(html.contains("32768 tokens"))
        XCTAssertTrue(html.contains("text · image · audio · message"))
    }
    func testJinaPageRendersTheJinaContract() {
        var jina = context
        jina = .init(
            baseURL: jina.baseURL, modelID: "jinaai/jina-embeddings-v5-omni-small", dimensions: 1024,
            maxTokens: 32768, compute: "auto", modalities: ["text", "image", "audio", "video"],
            maxBatch: jina.maxBatch, maxRequestTokens: jina.maxRequestTokens, maxBodyMB: jina.maxBodyMB,
            maxTotalBodyMB: jina.maxTotalBodyMB, maxQueueRequests: jina.maxQueueRequests,
            maxQueueItems: jina.maxQueueItems, batchWindowMS: jina.batchWindowMS,
            keepWarmSeconds: jina.keepWarmSeconds, maxConnections: jina.maxConnections,
            ioTimeoutSeconds: jina.ioTimeoutSeconds, shutdownGraceSeconds: jina.shutdownGraceSeconds,
            accessLogMode: jina.accessLogMode, spaceID: "glossematics:omni-small:sha256:abc",
            family: .jinaOmniSmall, matryoshka: [1024, 32, 64, 128, 256, 512])
        XCTAssertNil(DocsPage.validationError(jina))
        let html = DocsPage.render(jina) ?? ""
        XCTAssertFalse(html.contains("{{"))
        XCTAssertTrue(html.contains("32 / 64 / 128 / 256 / 512 / 1024"))
        XCTAssertTrue(html.contains("retrieval.query"))
        XCTAssertTrue(html.contains("input_video"))
        XCTAssertTrue(html.contains("text · image · audio · video"))
        XCTAssertFalse(html.contains("BidirLM-Omni-2.5B"))
    }
    func testValuesAreEscaped() {
        XCTAssertEqual(DocsPage.escape("&<>\""), "&amp;&lt;&gt;&quot;")
    }
    func testFixtureIsLabeledInDocs() {
        var fixture = context
        fixture.isDummy = true
        let html = DocsPage.render(fixture) ?? ""
        XCTAssertTrue(html.contains("Core ML golden fixture"))
        XCTAssertTrue(html.contains("not suitable for retrieval"))
    }
    func testResponseHasNonceCSP() {
        let response = DocsPage.response(context)
        XCTAssertEqual(response.status, 200)
        let csp = response.extraHeaders.first { $0.0 == "Content-Security-Policy" }?.1 ?? ""
        XCTAssertTrue(csp.contains("script-src 'nonce-"))
        XCTAssertEqual(response.extraHeaders.first { $0.0 == "X-Frame-Options" }?.1, "DENY")
    }
}
