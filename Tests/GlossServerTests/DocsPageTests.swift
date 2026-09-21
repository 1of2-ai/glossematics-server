import XCTest
@testable import gloss_server

final class DocsPageTests: XCTestCase {
    private var context: DocsPage.Context {
        DocsPage.Context(
            baseURL: "http://127.0.0.1:11435",
            modelID: "jinaai/jina-embeddings-v5-omni-small",
            defaultDimensions: 1024,
            matryoshka: [1024, 32, 512],
            maxBatch: 2048,
            maxBodyMB: 64,
            spaceID: "glossematics:omni-small:sha256:abc")
    }

    func testResourceIsPackagedIntoTheBundle() {
        // Fails when Resources/docs.html is missing from the target's resource bundle.
        XCTAssertTrue(DocsPage.templateAvailable, "docs.html did not load from Bundle.module")
    }

    func testRenderSubstitutesEveryToken() {
        let html = DocsPage.render(context)
        XCTAssertNotNil(html)
        if let html {
            // Any leftover token necessarily contains "{{"; "}}" alone is ambiguous because
            // adjacent JS/CSS closing braces produce the same sequence.
            XCTAssertFalse(html.contains("{{"), "unsubstituted tokens remain in rendered page")
            XCTAssert(html.contains("http://127.0.0.1:11435"))
            XCTAssert(html.contains("32 / 512 / 1024"))
            XCTAssert(html.contains("jinaai/jina-embeddings-v5-omni-small"))
            XCTAssert(html.contains("2048"))
            XCTAssert(html.contains("glossematics:omni-small:sha256:abc"))
        }
    }

    func testSectionAnchorsAndNavAgree() {
        let html = DocsPage.render(context) ?? ""
        let targets = ["content-get-started", "content-create-embeddings", "content-roles",
                       "content-models", "content-health", "content-errors"]
        for target in targets {
            XCTAssert(html.contains("id=\"\(target)\""), "missing section \(target)")
            XCTAssert(html.contains("data-target=\"\(target)\""), "missing nav link for \(target)")
        }
    }

    func testValuesAreHTMLEscaped() {
        let escaped = DocsPage.escape("&<>\"")
        XCTAssertEqual(escaped, "&amp;&lt;&gt;&quot;")
        // A model id containing markup must not reach the page raw.
        let hostile = DocsPage.Context(
            baseURL: "http://127.0.0.1:11435",
            modelID: "\"><script>alert(1)</script>",
            defaultDimensions: 1024,
            matryoshka: [1024],
            maxBatch: 2048,
            maxBodyMB: 64,
            spaceID: nil)
        let html = DocsPage.render(hostile) ?? ""
        XCTAssertFalse(html.contains("<script>alert(1)"), "unescaped value reached the page")
        XCTAssert(html.contains("&quot;&gt;&lt;script&gt;"))
    }

    func testResponseShape() {
        let response = DocsPage.response(context)
        XCTAssertEqual(response.status, 200)
        XCTAssertTrue(response.contentType.hasPrefix("text/html"))
        XCTAssertEqual(response.extraHeaders.first { $0.0 == "Cache-Control" }?.1, "no-store")
    }
}
