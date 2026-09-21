import Foundation

/// The /docs page.
///
/// The page's source of truth is `Resources/docs.html` — an editable HTML document in the repo
/// containing `{{TOKEN}}` placeholders. The file is packaged into the executable's SwiftPM
/// resource bundle at build time; serving substitutes each token with an HTML-escaped value
/// taken from the live server configuration, so the docs always describe the running daemon.
///
/// Layout adapted from the "API Documentation HTML Template" (MIT, © 2016 Florian Nicolas,
/// github.com/floriannicolas/API-Documentation-HTML-Template).
enum DocsPage {
    static let version = "1.0.0"

    struct Context {
        let baseURL: String
        let modelID: String
        let defaultDimensions: Int
        let matryoshka: [Int]
        let maxBatch: Int
        let maxBodyMB: Int
        let spaceID: String?
    }

    /// Loaded once from the packaged resource bundle. Nil means the resource did not make it
    /// into the bundle (a build misconfiguration) and /docs answers 500.
    private static let template: String? = {
        guard let url = Bundle.module.url(forResource: "docs", withExtension: "html") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }()

    static var templateAvailable: Bool { template != nil }

    static func response(_ context: Context) -> HTTPResponse {
        guard let html = render(context) else {
            return .error(500, APIError.apiError(
                "documentation resource missing from the executable bundle"))
        }
        return HTTPResponse(
            status: 200,
            contentType: "text/html; charset=utf-8",
            extraHeaders: [("Cache-Control", "no-store")],
            body: Data(html.utf8))
    }

    /// Token table for a context, exposed for tests. Values are substituted as-is; escaping
    /// happens inside `render`.
    static func tokens(_ context: Context) -> [(token: String, value: String)] {
        [
            ("BASE_URL", context.baseURL),
            ("MODEL_ID", context.modelID),
            ("DEFAULT_DIMENSIONS", String(context.defaultDimensions)),
            ("MATRYOSHKA", context.matryoshka.sorted().map(String.init).joined(separator: " / ")),
            ("MAX_BATCH", String(context.maxBatch)),
            ("MAX_BODY_MB", String(context.maxBodyMB)),
            ("SPACE_ID", context.spaceID ?? "n/a"),
            ("PORT", URL(string: context.baseURL)?.port.map(String.init) ?? "11435"),
            ("VERSION", version),
        ]
    }

    static func render(_ context: Context) -> String? {
        guard var html = template else { return nil }
        for (token, value) in tokens(context) {
            html = html.replacingOccurrences(of: "{{\(token)}}", with: escape(value))
        }
        return html
    }

    static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
