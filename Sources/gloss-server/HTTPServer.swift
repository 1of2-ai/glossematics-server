import Foundation
import Network

// Minimal HTTP/1.1 server over Network.framework, bound to the loopback interface only.
// It exists to serve one endpoint family on localhost, so it intentionally implements just
// what that requires: request-line + header parsing, Content-Length bodies (with
// Expect: 100-continue), keep-alive, and JSON responses. No TLS, no chunked request bodies.

struct HTTPRequest: Sendable {
    var method: String
    var path: String
    var query: String
    var version: String
    var headers: [String: String]
    var body: Data

    /// HTTP/1.1 defaults to keep-alive; HTTP/1.0 defaults to close. Header overrides both.
    var wantsKeepAlive: Bool {
        let connection = headers["connection"]?.lowercased()
        if version == "HTTP/1.0" { return connection?.contains("keep-alive") == true }
        return connection?.contains("close") != true
    }
}

struct HTTPResponse: Sendable {
    var status: Int
    var contentType: String
    var extraHeaders: [(String, String)]
    var body: Data

    init(status: Int, contentType: String = "application/json", extraHeaders: [(String, String)] = [], body: Data) {
        self.status = status
        self.contentType = contentType
        self.extraHeaders = extraHeaders
        self.body = body
    }

    static func json<T: Encodable>(_ status: Int, _ payload: T, extraHeaders: [(String, String)] = []) -> HTTPResponse {
        let encoder = JSONEncoder()
        let data = (try? encoder.encode(payload)) ?? Data("{}".utf8)
        return HTTPResponse(status: status, extraHeaders: extraHeaders, body: data)
    }

    static func rawJSON(_ status: Int, _ utf8: String, extraHeaders: [(String, String)] = []) -> HTTPResponse {
        HTTPResponse(status: status, extraHeaders: extraHeaders, body: Data(utf8.utf8))
    }

    static func error(_ status: Int, _ apiError: APIError, extraHeaders: [(String, String)] = []) -> HTTPResponse {
        json(status, apiError, extraHeaders: extraHeaders)
    }

    static let reasonPhrases: [Int: String] = [
        100: "Continue", 200: "OK", 400: "Bad Request", 404: "Not Found",
        405: "Method Not Allowed", 408: "Request Timeout", 413: "Content Too Large",
        415: "Unsupported Media Type", 422: "Unprocessable Content",
        431: "Request Header Fields Too Large", 500: "Internal Server Error",
        503: "Service Unavailable",
    ]

    static func reason(_ status: Int) -> String {
        reasonPhrases[status] ?? "Status \(status)"
    }
}

enum HTTPError: Error, CustomStringConvertible {
    case connectionClosed
    case headerTooLarge
    case bodyTooLarge(limit: Int)
    case malformedRequest(String)

    var description: String {
        switch self {
        case .connectionClosed: return "connection closed"
        case .headerTooLarge: return "request headers exceed the size limit"
        case let .bodyTooLarge(limit): return "request body exceeds the \(limit)-byte limit"
        case let .malformedRequest(reason): return reason
        }
    }
}

/// One accepted client connection. All buffer state is owned by the single pump task;
/// `nw` and `handler` are immutable after init, so unchecked Sendable is sound here.
final class HTTPConnection: @unchecked Sendable {
    private let nw: NWConnection
    private let handler: @Sendable (HTTPRequest) async -> HTTPResponse
    private let maxBodyBytes: Int
    private var pending = [UInt8]()

    init(nw: NWConnection, handler: @escaping @Sendable (HTTPRequest) async -> HTTPResponse, maxBodyBytes: Int) {
        self.nw = nw
        self.handler = handler
        self.maxBodyBytes = maxBodyBytes
    }

    func start(queue: DispatchQueue) {
        nw.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.pump()
            case .failed, .cancelled:
                self.nw.stateUpdateHandler = nil
            default:
                break
            }
        }
        nw.start(queue: queue)
    }

    func cancel() {
        nw.cancel()
    }

    private func pump() {
        Task { [self] in
            do {
                while true {
                    let request = try await readRequest()
                    let keepAlive = request.wantsKeepAlive
                    let response = await handler(request)
                    try await write(response, keepAlive: keepAlive)
                    if !keepAlive {
                        try await writeFinal()
                        break
                    }
                }
            } catch is CancellationError {
                // Client went away mid-request.
            } catch {
                // Best-effort error response; the connection is likely unusable afterwards.
                if let httpError = error as? HTTPError {
                    let status: Int
                    switch httpError {
                    case .bodyTooLarge: status = 413
                    case .headerTooLarge: status = 431
                    case .malformedRequest: status = 400
                    case .connectionClosed: status = 408
                    }
                    let body = try? JSONEncoder().encode(
                        APIError.invalidRequest(httpError.description))
                    let head = "HTTP/1.1 \(status) \(HTTPResponse.reason(status))\r\n"
                        + "Content-Type: application/json\r\n"
                        + "Content-Length: \(body?.count ?? 0)\r\n"
                        + "Connection: close\r\n\r\n"
                    var payload = Data(head.utf8)
                    if let body { payload.append(body) }
                    try? await send(payload)
                }
            }
            nw.cancel()
        }
    }

    // MARK: - Receive

    private func receiveChunk() async throws -> [UInt8] {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                nw.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { data, _, isComplete, error in
                    if let data, !data.isEmpty {
                        continuation.resume(returning: [UInt8](data))
                    } else if let error {
                        continuation.resume(throwing: error)
                    } else {
                        // EOF with no data (isComplete) or a spurious empty read: peer is gone.
                        continuation.resume(throwing: HTTPError.connectionClosed)
                    }
                }
            }
        } onCancel: {
            nw.cancel()
        }
    }

    private func readRequest() async throws -> HTTPRequest {
        // 1) Headers, terminated by CRLFCRLF.
        let delimiter: [UInt8] = Array("\r\n\r\n".utf8)
        var searchFloor = 0
        var headerEnd = find(delimiter, in: pending, from: searchFloor)
        while headerEnd == nil {
            guard pending.count < 64 * 1024 else { throw HTTPError.headerTooLarge }
            let chunk = try await receiveChunk()
            searchFloor = max(0, pending.count - delimiter.count + 1)
            pending.append(contentsOf: chunk)
            headerEnd = find(delimiter, in: pending, from: searchFloor)
        }
        let headBytes = Array(pending[0..<headerEnd!])
        pending.removeFirst(headerEnd! + delimiter.count)

        let head = String(decoding: headBytes, as: UTF8.self)
        var lines = head.split(separator: "\r\n", omittingEmptySubsequences: false)[...]
        guard let requestLine = lines.first else {
            throw HTTPError.malformedRequest("empty request head")
        }
        lines = lines.dropFirst()
        let parts = requestLine.split(separator: " ")
        guard parts.count == 3, parts[0].count > 0, parts[2].hasPrefix("HTTP/") else {
            throw HTTPError.malformedRequest("malformed request line")
        }
        var headers = [String: String]()
        headers.reserveCapacity(lines.count)
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[line.startIndex..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            headers[name.lowercased()] = value
        }

        let target = String(parts[1])
        let (path, query) = splitTarget(target)
        let request = HTTPRequest(
            method: String(parts[0]),
            path: path,
            query: query,
            version: String(parts[2]),
            headers: headers,
            body: Data())

        // 2) Interim 100 Continue so curl/requests don't stall waiting to send large bodies.
        if headers["expect"]?.lowercased().contains("100-continue") == true {
            try await send(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
        }

        // 3) Body by Content-Length. Chunked request bodies are not supported by design.
        var body = pending
        pending = []
        if let transferEncoding = headers["transfer-encoding"],
           transferEncoding.lowercased().contains("chunked") {
            throw HTTPError.malformedRequest(
                "chunked request bodies are not supported; send Content-Length")
        }
        let contentLength = Int(headers["content-length"] ?? "0") ?? -1
        guard contentLength >= 0 else {
            throw HTTPError.malformedRequest("invalid Content-Length")
        }
        guard contentLength <= maxBodyBytes else {
            throw HTTPError.bodyTooLarge(limit: maxBodyBytes)
        }
        while body.count < contentLength {
            body.append(contentsOf: try await receiveChunk())
        }
        if body.count > contentLength {
            // Keep any pipelined bytes for the next request.
            pending = Array(body[contentLength...])
            body = Array(body[0..<contentLength])
        }

        var finalized = request
        finalized.body = Data(body)
        return finalized
    }

    // MARK: - Send

    private func write(_ response: HTTPResponse, keepAlive: Bool) async throws {
        var head = "HTTP/1.1 \(response.status) \(HTTPResponse.reason(response.status))\r\n"
        head += "Content-Type: \(response.contentType)\r\n"
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n"
        for (name, value) in response.extraHeaders {
            head += "\(name): \(value)\r\n"
        }
        head += "\r\n"
        var payload = Data(head.utf8)
        payload.append(response.body)
        if keepAlive {
            try await send(payload)
        } else {
            // Ask the stack to close only after every buffered byte has drained.
            try await sendFinal(payload)
        }
    }

    private func writeFinal() async throws {
        nw.send(content: nil, contentContext: .finalMessage, completion: .contentProcessed { _ in })
    }

    private func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            nw.send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            })
        }
    }

    private func sendFinal(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            nw.send(
                content: data, contentContext: .finalMessage, isComplete: true,
                completion: .contentProcessed { error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume()
                    }
                })
        }
    }

    // MARK: - Parsing helpers

    private func find(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        let upper = haystack.count - needle.count
        guard start <= upper else { return nil }
        var index = start
        while index <= upper {
            if haystack[index] == needle[0] {
                var matched = 1
                while matched < needle.count, haystack[index + matched] == needle[matched] {
                    matched += 1
                }
                if matched == needle.count { return index }
            }
            index += 1
        }
        return nil
    }

    private func splitTarget(_ target: String) -> (String, String) {
        guard let questionMark = target.firstIndex(of: "?") else {
            return (target, "")
        }
        return (String(target[target.startIndex..<questionMark]),
                String(target[target.index(after: questionMark)...]))
    }
}

/// Listener bound to the loopback interface. Not a daemon-facing multiplexer: one accept loop,
/// one task per connection.
final class HTTPServer: @unchecked Sendable {
    let port: UInt16
    private let listener: NWListener
    private let handler: @Sendable (HTTPRequest) async -> HTTPResponse
    private let queue: DispatchQueue
    private let maxBodyBytes: Int
    private let lock = NSLock()
    private var connections: [ObjectIdentifier: HTTPConnection] = [:]
    private var started = false

    init(port: UInt16, maxBodyBytes: Int, handler: @escaping @Sendable (HTTPRequest) async -> HTTPResponse) throws {
        self.port = port
        self.handler = handler
        self.maxBodyBytes = maxBodyBytes
        self.queue = DispatchQueue(label: "glossematics.http.server")
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!)
        self.listener = try NWListener(using: parameters)
    }

    func start() {
        lock.lock()
        let alreadyStarted = started
        started = true
        lock.unlock()
        guard !alreadyStarted else { return }

        listener.newConnectionHandler = { [weak self] nw in
            guard let self else { return }
            let connection = HTTPConnection(
                nw: nw, handler: self.handler, maxBodyBytes: self.maxBodyBytes)
            self.lock.lock()
            self.connections[ObjectIdentifier(connection)] = connection
            self.lock.unlock()
            connection.start(queue: self.queue)
        }
        listener.start(queue: queue)
    }

    func stop() {
        listener.cancel()
        lock.lock()
        let all = Array(connections.values)
        connections.removeAll()
        lock.unlock()
        for connection in all { connection.cancel() }
    }
}
