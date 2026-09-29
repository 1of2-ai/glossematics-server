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
