import Foundation

enum DocsPage {
    static let version = BuildInfo.version

    struct Context: Sendable {
        let baseURL: String; let modelID: String; let dimensions: Int; let maxTokens: Int
        let compute: String; let modalities: [String]
        let maxBatch: Int; let maxRequestTokens: Int; let maxBodyMB: Int; let maxTotalBodyMB: Int
        let maxQueueRequests: Int; let maxQueueItems: Int; let batchWindowMS: Double; let keepWarmSeconds: Int
        let maxConnections: Int; let ioTimeoutSeconds: Int; let shutdownGraceSeconds: Int; let accessLogMode: String
        let spaceID: String?
        var isDummy = false
        /// Selects the page: `docs.html` (BidirLM) or `docs-jina.html`.
        var family: ModelFamily = .bidirlm
        /// Matryoshka output sizes (Jina); `dimensions` is then the default size.
        var matryoshka: [Int] = []
    }

    static func resourceName(_ family: ModelFamily) -> String {
        switch family {
        case .bidirlm: "docs"
        case .jinaOmniSmall: "docs-jina"
        }
    }

    private static let templates: [ModelFamily: String] = {
        var loaded = [ModelFamily: String]()
        for family in ModelFamily.allCases {
            if let url = Bundle.module.url(forResource: resourceName(family), withExtension: "html"),
               let text = try? String(contentsOf: url, encoding: .utf8) {
                loaded[family] = text
            }
        }
        return loaded
    }()

    static func template(for family: ModelFamily) -> String? { templates[family] }

    static var templateAvailable: Bool { templates[.bidirlm] != nil }

    static func validationError(_ context: Context) -> String? {
        guard let template = template(for: context.family) else {
            return "\(resourceName(context.family)).html is missing from the executable resource bundle"
        }
        let known = Set(tokens(context).map(\.token))
        let unknown = placeholderNames(in: template).subtracting(known)
        return unknown.isEmpty ? nil : "docs.html contains unknown placeholders: \(unknown.sorted().joined(separator: ", "))"
    }

    static func response(_ context: Context) -> HTTPResponse {
        let nonce = UUID().uuidString.replacingOccurrences(of: "-", with: "")
        guard let html = render(context, scriptNonce: nonce) else {
            return .error(500, .apiError("documentation resource missing or invalid"))
        }
        return .init(
            status: 200,
            contentType: "text/html; charset=utf-8",
            extraHeaders: [
                ("Cache-Control", "no-store"),
                ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-\(nonce)'; img-src data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
                ("Referrer-Policy", "no-referrer"),
                ("X-Frame-Options", "DENY"),
                ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
            ],
            body: Data(html.utf8))
    }

    static func tokens(_ c: Context, scriptNonce: String = "docs-preview") -> [(token: String, value: String)] {
        [
            ("BASE_URL", c.baseURL), ("MODEL_ID", c.modelID), ("DIMENSIONS", String(c.dimensions)),
            ("MAX_TOKENS", String(c.maxTokens)), ("COMPUTE", c.compute),
            ("MODALITIES", c.modalities.joined(separator: " · ")), ("MAX_BATCH", String(c.maxBatch)),
            ("MAX_REQUEST_TOKENS", String(c.maxRequestTokens)),
            ("MAX_BODY_MB", String(c.maxBodyMB)), ("MAX_TOTAL_BODY_MB", String(c.maxTotalBodyMB)), ("MAX_QUEUE_REQUESTS", String(c.maxQueueRequests)),
            ("MAX_QUEUE_ITEMS", String(c.maxQueueItems)), ("BATCH_WINDOW_MS", String(format: "%.2f", c.batchWindowMS)),
            ("KEEP_WARM_LABEL", c.keepWarmSeconds > 0 ? "every \(c.keepWarmSeconds)s" : "disabled"), ("MAX_CONNECTIONS", String(c.maxConnections)),
            ("IO_TIMEOUT_SECONDS", String(c.ioTimeoutSeconds)), ("SHUTDOWN_GRACE_SECONDS", String(c.shutdownGraceSeconds)), ("ACCESS_LOG_MODE", c.accessLogMode),
            ("SPACE_ID", c.spaceID ?? "n/a"), ("VERSION", version), ("CSP_NONCE", scriptNonce),
            ("MATRYOSHKA", c.matryoshka.sorted().map(String.init).joined(separator: " / ")),
            ("SERVING_MODE", c.isDummy ? "Core ML golden fixture" : "Production local microservice"),
            ("FIXTURE_NOTICE", c.isDummy
                ? "This process serves a deterministic Core ML test fixture. Its embeddings are not suitable for retrieval."
                : "This process serves a validated local Core ML bundle."),
        ]
    }

    static func render(_ context: Context, scriptNonce: String = "docs-preview") -> String? {
        guard validationError(context) == nil, var html = template(for: context.family) else { return nil }
        for (token, value) in tokens(context, scriptNonce: scriptNonce) {
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

    private static func placeholderNames(in html: String) -> Set<String> {
        guard let regex = try? NSRegularExpression(pattern: #"\{\{([A-Z0-9_]+)\}\}"#) else { return [] }
        let range = NSRange(html.startIndex..<html.endIndex, in: html)
        return Set(regex.matches(in: html, range: range).compactMap { match in
            guard let r = Range(match.range(at: 1), in: html) else { return nil }
            return String(html[r])
        })
    }
}
