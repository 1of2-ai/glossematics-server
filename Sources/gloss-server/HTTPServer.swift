import Foundation
import Network

private final class HTTPDateClock: @unchecked Sendable {
    static let shared = HTTPDateClock()
    private let lock = NSLock()
    private let formatter: DateFormatter
    private var cachedSecond: Int64 = -1
    private var cachedValue = ""

    private init() {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        self.formatter = formatter
    }

    func now() -> String {
        lock.withLock {
            let date = Date()
            let second = Int64(date.timeIntervalSince1970)
            if second != cachedSecond {
                cachedSecond = second
                cachedValue = formatter.string(from: date)
            }
            return cachedValue
        }
    }
}

struct HTTPRequest: Sendable {
    var method: String
    var path: String
    var query: String
    var version: String
    var headers: [String: String]
    var body: Data
    var requestID: String

    var wantsKeepAlive: Bool {
        let value = headers["connection"]?.lowercased()
        if version == "HTTP/1.0" { return value?.contains("keep-alive") == true }
        return value?.contains("close") != true
    }
}

struct HTTPResponse: Sendable {
    var status: Int
    var contentType: String
    var extraHeaders: [(String, String)]
    var body: Data
    var onComplete: (@Sendable () async -> Void)?

    init(
        status: Int,
        contentType: String = "application/json",
        extraHeaders: [(String, String)] = [],
        body: Data,
        onComplete: (@Sendable () async -> Void)? = nil
    ) {
        self.status = status
        self.contentType = contentType
        self.extraHeaders = extraHeaders
        self.body = body
        self.onComplete = onComplete
    }

    static func json<T: Encodable>(
        _ status: Int,
        _ payload: T,
        extraHeaders: [(String, String)] = []
    ) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        do {
            return HTTPResponse(
                status: status,
                extraHeaders: extraHeaders,
                body: try encoder.encode(payload))
        } catch {
            return HTTPResponse(
                status: 500,
                body: Data(#"{"error":{"message":"failed to encode server response","type":"api_error","param":null,"code":null}}"#.utf8))
        }
    }

    static func error(
        _ status: Int,
        _ error: APIError,
        extraHeaders: [(String, String)] = []
    ) -> HTTPResponse {
        json(status, error, extraHeaders: extraHeaders)
    }

    static let reasonPhrases: [Int: String] = [
        100: "Continue", 200: "OK", 302: "Found", 400: "Bad Request",
        404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout",
        413: "Content Too Large", 415: "Unsupported Media Type", 417: "Expectation Failed",
        431: "Request Header Fields Too Large", 500: "Internal Server Error",
        503: "Service Unavailable",
    ]

    static func reason(_ status: Int) -> String {
        reasonPhrases[status] ?? "Status \(status)"
    }
}

enum HTTPError: Error, CustomStringConvertible {
    case connectionClosed
    case requestTimeout
    case headerTooLarge
    case tooManyHeaders
    case bodyTooLarge(limit: Int)
    case aggregateBodyBudgetExhausted(limit: Int)
    case expectationFailed(String)
    case malformedRequest(String)

    var description: String {
        switch self {
        case .connectionClosed: "connection closed"
        case .requestTimeout: "request I/O timed out"
        case .headerTooLarge: "request headers exceed the size limit"
        case .tooManyHeaders: "request contains too many headers"
        case let .bodyTooLarge(limit): "request body exceeds the \(limit)-byte limit"
        case let .aggregateBodyBudgetExhausted(limit):
            "server request-body budget is full (\(limit) bytes); retry shortly"
        case let .expectationFailed(reason): reason
        case let .malformedRequest(reason): reason
        }
    }
}

enum HTTPServerError: Error, CustomStringConvertible {
    case startTimedOut
    case listenerFailed(String)

    var description: String {
        switch self {
        case .startTimedOut: "HTTP listener did not become ready before timeout"
        case let .listenerFailed(reason): "HTTP listener failed: \(reason)"
        }
    }
}

actor BodyBudget {
    nonisolated let limit: Int
    private var reserved = 0

    init(limit: Int) { self.limit = limit }

    func tryReserve(_ bytes: Int) -> Bool {
        guard bytes >= 0, reserved + bytes <= limit else { return false }
        reserved += bytes
        return true
    }

    func release(_ bytes: Int) {
        reserved = max(0, reserved - max(0, bytes))
    }
}

private struct ReceivedRequest {
    var request: HTTPRequest
    var bodyReservation: Int
}

final class HTTPConnection: @unchecked Sendable {
    private let nw: NWConnection
    private let handler: @Sendable (HTTPRequest) async -> HTTPResponse
    private let maxBodyBytes: Int
    private let bodyBudget: BodyBudget
    private let ioTimeoutNanoseconds: UInt64
    private let onClose: @Sendable (ObjectIdentifier) -> Void
    private let stateLock = NSLock()
    private var pumpTask: Task<Void, Never>?
    private var finished = false
    private var draining = false
    private var handlingRequest = false
    private var pending = [UInt8]()
    private let maxKeepAliveRequests = 100

    init(
        nw: NWConnection,
        handler: @escaping @Sendable (HTTPRequest) async -> HTTPResponse,
        maxBodyBytes: Int,
        bodyBudget: BodyBudget,
        ioTimeoutSeconds: Int,
        onClose: @escaping @Sendable (ObjectIdentifier) -> Void
    ) {
        self.nw = nw
        self.handler = handler
        self.maxBodyBytes = maxBodyBytes
        self.bodyBudget = bodyBudget
        self.ioTimeoutNanoseconds = UInt64(max(1, ioTimeoutSeconds)) * 1_000_000_000
        self.onClose = onClose
    }

    func start(queue: DispatchQueue) {
        nw.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready: self.startPumpOnce()
            case .failed, .cancelled: self.finish()
            default: break
            }
        }
        nw.start(queue: queue)
    }

    func beginDrain() {
        let idleTask: Task<Void, Never>? = stateLock.withLock {
            draining = true
            return handlingRequest ? nil : pumpTask
        }
        idleTask?.cancel()
    }

    func cancel() {
        stateLock.withLock { pumpTask }?.cancel()
        nw.cancel()
    }

    private func startPumpOnce() {
        stateLock.lock()
        guard pumpTask == nil, !finished else {
            stateLock.unlock()
            return
        }
        let task = Task { [self] in await pump() }
        pumpTask = task
        stateLock.unlock()
    }

    private func finish() {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        pumpTask?.cancel()
        pumpTask = nil
        stateLock.unlock()
        nw.stateUpdateHandler = nil
        onClose(ObjectIdentifier(self))
    }

    private func isDraining() -> Bool { stateLock.withLock { draining } }
    private func setHandling(_ value: Bool) { stateLock.withLock { handlingRequest = value } }

    private func pump() async {
        var served = 0
        defer {
            nw.cancel()
            finish()
        }

        do {
            while !Task.isCancelled {
                if isDraining() { break }
                let received = try await readRequest()
                setHandling(true)
                served += 1
                var keepAlive = received.request.wantsKeepAlive && served < maxKeepAliveRequests
                let response = await handler(received.request)
                do {
                    try Task.checkCancellation()
                    if isDraining() { keepAlive = false }
                    try await write(
                        response,
                        requestID: received.request.requestID,
                        keepAlive: keepAlive)
                    await response.onComplete?()
                    await bodyBudget.release(received.bodyReservation)
                    setHandling(false)
                } catch {
                    await response.onComplete?()
                    await bodyBudget.release(received.bodyReservation)
                    setHandling(false)
                    throw error
                }
                if !keepAlive { break }
            }
        } catch is CancellationError {
        } catch let error as HTTPError {
            if case .connectionClosed = error { return }
            var headers: [(String, String)] = []
            let status: Int
            switch error {
            case .bodyTooLarge: status = 413
            case .aggregateBodyBudgetExhausted:
                status = 503
                headers.append(("Retry-After", "1"))
            case .headerTooLarge, .tooManyHeaders: status = 431
            case .expectationFailed: status = 417
            case .requestTimeout: status = 408
            case .malformedRequest: status = 400
            case .connectionClosed: return
            }
            try? await write(
                .error(status, .invalidRequest(error.description), extraHeaders: headers),
                requestID: UUID().uuidString.lowercased(),
                keepAlive: false)
        } catch {
            try? await write(
                .error(500, .apiError("transport failure")),
                requestID: UUID().uuidString.lowercased(),
                keepAlive: false)
        }
    }

    private func receiveChunk() async throws -> [UInt8] {
        try await withThrowingTaskGroup(of: [UInt8].self) { group in
            group.addTask { [self] in try await receiveRaw() }
            let timeout = ioTimeoutNanoseconds
            group.addTask {
                try await Task.sleep(nanoseconds: timeout)
                try Task.checkCancellation()
                throw HTTPError.requestTimeout
            }
            guard let first = try await group.next() else { throw HTTPError.connectionClosed }
            group.cancelAll()
            return first
        }
    }

    private func receiveRaw() async throws -> [UInt8] {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                nw.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, _, error in
                    if let data, !data.isEmpty {
                        continuation.resume(returning: [UInt8](data))
                    } else if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(throwing: HTTPError.connectionClosed)
                    }
                }
            }
        } onCancel: {
            self.nw.cancel()
        }
    }

    private func readRequest() async throws -> ReceivedRequest {
        let delimiter = Array("\r\n\r\n".utf8)
        let maxHeaderBytes = 64 * 1024
        var searchFloor = 0
        var headerEnd = find(delimiter, in: pending, from: searchFloor)
        while headerEnd == nil {
            let chunk = try await receiveChunk()
            searchFloor = max(0, pending.count - delimiter.count + 1)
            pending.append(contentsOf: chunk)
            headerEnd = find(delimiter, in: pending, from: searchFloor)
            if headerEnd == nil, pending.count > maxHeaderBytes + delimiter.count {
                throw HTTPError.headerTooLarge
            }
        }
        guard let headerEnd, headerEnd <= maxHeaderBytes else { throw HTTPError.headerTooLarge }
        let headBytes = Array(pending[0..<headerEnd])
        pending.removeFirst(headerEnd + delimiter.count)
        guard let head = String(bytes: headBytes, encoding: .utf8) else {
            throw HTTPError.malformedRequest("request headers are not valid UTF-8")
        }

        var lines = head.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { throw HTTPError.malformedRequest("empty request head") }
        let parts = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: false)
        guard parts.count == 3,
              !parts[0].isEmpty,
              !parts[1].isEmpty,
              parts[2] == "HTTP/1.1" || parts[2] == "HTTP/1.0" else {
            throw HTTPError.malformedRequest("malformed request line")
        }
        let method = String(parts[0])
        guard isHTTPToken(method), method == method.uppercased() else {
            throw HTTPError.malformedRequest("invalid HTTP method")
        }
        let target = String(parts[1])
        guard target.hasPrefix("/"), !target.contains("#"), isSafeRequestTarget(target) else {
            throw HTTPError.malformedRequest(
                "request target must be visible-ASCII origin-form without a fragment")
        }
        guard lines.count <= 100 else { throw HTTPError.tooManyHeaders }

        var headers = [String: String]()
        for line in lines where !line.isEmpty {
            guard line.first != " ", line.first != "\t", let colon = line.firstIndex(of: ":") else {
                throw HTTPError.malformedRequest("malformed HTTP header")
            }
            let rawName = String(line[..<colon])
            guard isHTTPToken(rawName) else { throw HTTPError.malformedRequest("invalid HTTP header name") }
            let name = rawName.lowercased()
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard isSafeHeaderValue(value) else {
                throw HTTPError.malformedRequest("HTTP header values contain control characters")
            }
            if headers[name] != nil {
                if name == "content-length" || name == "host" {
                    throw HTTPError.malformedRequest("duplicate \(rawName) header")
                }
                headers[name]! += ", " + value
            } else {
                headers[name] = value
            }
        }

        let version = String(parts[2])
        if version == "HTTP/1.1" {
            guard let host = headers["host"], isLoopbackHost(host) else {
                throw HTTPError.malformedRequest("Host must name localhost or loopback")
            }
        } else if let host = headers["host"], !isLoopbackHost(host) {
            throw HTTPError.malformedRequest("Host must name localhost or loopback")
        }
        if headers["transfer-encoding"] != nil {
            throw HTTPError.malformedRequest(
                "Transfer-Encoding is not supported; send one Content-Length header")
        }

        let contentLength: Int
        if let raw = headers["content-length"] {
            let bytes = raw.utf8
            guard !bytes.isEmpty,
                  bytes.allSatisfy({ (48...57).contains($0) }),
                  let parsed = Int(raw) else {
                throw HTTPError.malformedRequest("invalid Content-Length")
            }
            contentLength = parsed
        } else {
            contentLength = 0
        }
        guard contentLength <= maxBodyBytes else { throw HTTPError.bodyTooLarge(limit: maxBodyBytes) }
        guard await bodyBudget.tryReserve(contentLength) else {
            throw HTTPError.aggregateBodyBudgetExhausted(limit: bodyBudget.limit)
        }

        do {
            if let expect = headers["expect"] {
                guard expect.lowercased() == "100-continue" else {
                    throw HTTPError.expectationFailed("only Expect: 100-continue is supported")
                }
                try await send(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
            }
            var body = pending
            pending = []
            while body.count < contentLength {
                body.append(contentsOf: try await receiveChunk())
            }
            if body.count > contentLength {
                pending = Array(body[contentLength...])
                body = Array(body[..<contentLength])
            }
            let (path, query) = splitTarget(target)
            let requestID = validRequestID(headers["x-request-id"])
                ?? UUID().uuidString.lowercased()
            return ReceivedRequest(
                request: .init(
                    method: method,
                    path: path,
                    query: query,
                    version: version,
                    headers: headers,
                    body: Data(body),
                    requestID: requestID),
                bodyReservation: contentLength)
        } catch {
            await bodyBudget.release(contentLength)
            throw error
        }
    }

    private func write(_ response: HTTPResponse, requestID: String, keepAlive: Bool) async throws {
        var head = "HTTP/1.1 \(response.status) \(HTTPResponse.reason(response.status))\r\n"
        head += "Date: \(HTTPDateClock.shared.now())\r\n"
        head += "Server: \(BuildInfo.serverHeader)\r\n"
        head += "Content-Type: \(response.contentType)\r\n"
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n"
        head += "X-Request-ID: \(requestID)\r\n"
        head += "X-Content-Type-Options: nosniff\r\n"
        if !response.extraHeaders.contains(where: {
            $0.0.caseInsensitiveCompare("Cache-Control") == .orderedSame
        }) {
            head += "Cache-Control: no-store\r\n"
        }
        for (name, value) in response.extraHeaders where safeResponseHeader(name: name, value: value) {
            head += "\(name): \(value)\r\n"
        }
        head += "\r\n"
        var data = Data(head.utf8)
        data.append(response.body)
        try await sendWithTimeout(data, final: !keepAlive)
    }

    private func send(_ data: Data) async throws {
        try await sendWithTimeout(data, final: false)
    }

    private func sendWithTimeout(_ data: Data, final: Bool) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [self] in try await sendRaw(data, final: final) }
            let timeout = ioTimeoutNanoseconds
            group.addTask {
                try await Task.sleep(nanoseconds: timeout)
                try Task.checkCancellation()
                throw HTTPError.requestTimeout
            }
            _ = try await group.next()
            group.cancelAll()
        }
    }

    private func sendRaw(_ data: Data, final: Bool) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                let done = NWConnection.SendCompletion.contentProcessed { error in
                    if let error { continuation.resume(throwing: error) }
                    else { continuation.resume() }
                }
                if final {
                    nw.send(content: data, contentContext: .finalMessage, isComplete: true, completion: done)
                } else {
                    nw.send(content: data, completion: done)
                }
            }
        } onCancel: {
            self.nw.cancel()
        }
    }

    private func find(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        let upper = haystack.count - needle.count
        guard start <= upper else { return nil }
        var index = max(0, start)
        while index <= upper {
            if haystack[index] == needle[0] {
                var matched = 1
                while matched < needle.count, haystack[index + matched] == needle[matched] { matched += 1 }
                if matched == needle.count { return index }
            }
            index += 1
        }
        return nil
    }

    private func splitTarget(_ target: String) -> (String, String) {
        guard let q = target.firstIndex(of: "?") else { return (target, "") }
        return (String(target[..<q]), String(target[target.index(after: q)...]))
    }

    private func isHTTPToken(_ text: String) -> Bool {
        let allowed = CharacterSet(charactersIn:
            "!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return !text.isEmpty && text.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    private func isSafeRequestTarget(_ text: String) -> Bool {
        text.unicodeScalars.allSatisfy { (0x21...0x7e).contains($0.value) }
    }

    private func isSafeHeaderValue(_ text: String) -> Bool {
        text.unicodeScalars.allSatisfy { $0.value == 9 || (0x20...0x7e).contains($0.value) }
    }

    private func isLoopbackHost(_ raw: String) -> Bool {
        let host = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        for prefix in ["localhost", "127.0.0.1", "[::1]"] {
            if host == prefix { return true }
            if host.hasPrefix(prefix + ":") {
                return Int(host.dropFirst(prefix.count + 1)).map { (1...65535).contains($0) } ?? false
            }
        }
        return false
    }

    private func validRequestID(_ value: String?) -> String? {
        guard let value, (1...128).contains(value.count) else { return nil }
        let allowed = CharacterSet(charactersIn:
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.:")
        return value.unicodeScalars.allSatisfy { allowed.contains($0) } ? value : nil
    }

    private func safeResponseHeader(name: String, value: String) -> Bool {
        let reserved = [
            "date", "server", "content-type", "content-length", "connection",
            "x-request-id", "x-content-type-options",
        ]
        return isHTTPToken(name)
            && !reserved.contains(name.lowercased())
            && isSafeHeaderValue(value)
    }
}

final class HTTPServer: @unchecked Sendable {
    let port: UInt16
    private let listener: NWListener
    private let handler: @Sendable (HTTPRequest) async -> HTTPResponse
    private let queue = DispatchQueue(label: "glossematics.http.server", qos: .userInitiated)
    private let maxBodyBytes: Int
    private let budget: BodyBudget
    private let maxConnections: Int
    private let ioTimeoutSeconds: Int
    private let onFatal: @Sendable (String) -> Void
    private let lock = NSLock()
    private var connections: [ObjectIdentifier: HTTPConnection] = [:]
    private var started = false
    private var becameReady = false
    private var fatalDelivered = false

    init(
        port: UInt16,
        maxBodyBytes: Int,
        maxTotalBodyBytes: Int,
        maxConnections: Int,
        ioTimeoutSeconds: Int,
        handler: @escaping @Sendable (HTTPRequest) async -> HTTPResponse,
        onFatal: @escaping @Sendable (String) -> Void = { _ in }
    ) throws {
        self.port = port
        self.handler = handler
        self.maxBodyBytes = maxBodyBytes
        self.budget = BodyBudget(limit: maxTotalBodyBytes)
        self.maxConnections = maxConnections
        self.ioTimeoutSeconds = ioTimeoutSeconds
        self.onFatal = onFatal
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(
            host: .ipv4(.loopback),
            port: NWEndpoint.Port(rawValue: port)!)
        listener = try NWListener(using: parameters)
    }

    func start(timeoutSeconds: Double = 5) throws {
        lock.lock()
        guard !started else { lock.unlock(); return }
        started = true
        lock.unlock()
        let semaphore = DispatchSemaphore(value: 0)
        final class StartBox: @unchecked Sendable {
            let lock = NSLock()
            var result: Result<Void, any Error>?
        }
        let box = StartBox()
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.lock.lock()
                let first = !self.becameReady
                self.becameReady = true
                self.lock.unlock()
                if first {
                    box.lock.withLock { if box.result == nil { box.result = .success(()) } }
                    semaphore.signal()
                }
            case let .failed(error):
                if self.lock.withLock({ self.becameReady }) {
                    self.deliverFatal("listener failed after startup: \(error)")
                } else {
                    box.lock.withLock {
                        if box.result == nil {
                            box.result = .failure(HTTPServerError.listenerFailed(String(describing: error)))
                        }
                    }
                    semaphore.signal()
                }
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
        guard semaphore.wait(timeout: .now() + timeoutSeconds) == .success else {
            listener.cancel()
            throw HTTPServerError.startTimedOut
        }
        guard let result = box.lock.withLock({ box.result }) else { throw HTTPServerError.startTimedOut }
        try result.get()
    }

    func shutdown(graceSeconds: Int) async {
        listener.cancel()
        lock.withLock { Array(connections.values) }.forEach { $0.beginDrain() }
        let deadline = Date().addingTimeInterval(TimeInterval(max(0, graceSeconds)))
        while Date() < deadline {
            if lock.withLock({ connections.isEmpty }) { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        let remaining = lock.withLock { () -> [HTTPConnection] in
            let snapshot = Array(connections.values)
            connections.removeAll()
            return snapshot
        }
        remaining.forEach { $0.cancel() }
    }

    private func accept(_ nw: NWConnection) {
        if lock.withLock({ connections.count >= maxConnections }) { reject(nw); return }
        let connection = HTTPConnection(
            nw: nw,
            handler: handler,
            maxBodyBytes: maxBodyBytes,
            bodyBudget: budget,
            ioTimeoutSeconds: ioTimeoutSeconds,
            onClose: { [weak self] id in self?.remove(id) })
        let id = ObjectIdentifier(connection)
        lock.lock()
        if connections.count >= maxConnections {
            lock.unlock()
            reject(nw)
            return
        }
        connections[id] = connection
        lock.unlock()
        connection.start(queue: queue)
    }

    private func remove(_ id: ObjectIdentifier) {
        _ = lock.withLock { connections.removeValue(forKey: id) }
    }

    private func reject(_ nw: NWConnection) {
        nw.stateUpdateHandler = { state in
            guard case .ready = state else { return }
            let body = Data(
                #"{"error":{"message":"too many concurrent connections","type":"service_unavailable","param":null,"code":null}}"#.utf8)
            let requestID = UUID().uuidString.lowercased()
            let head = "HTTP/1.1 503 Service Unavailable\r\n"
                + "Date: \(HTTPDateClock.shared.now())\r\n"
                + "Server: \(BuildInfo.serverHeader)\r\n"
                + "Content-Type: application/json\r\n"
                + "Content-Length: \(body.count)\r\n"
                + "Connection: close\r\n"
                + "Retry-After: 1\r\n"
                + "X-Request-ID: \(requestID)\r\n"
                + "Cache-Control: no-store\r\n\r\n"
            var data = Data(head.utf8)
            data.append(body)
            nw.send(
                content: data,
                contentContext: .finalMessage,
                isComplete: true,
                completion: .contentProcessed { _ in nw.cancel() })
        }
        nw.start(queue: queue)
    }

    private func deliverFatal(_ message: String) {
        lock.lock()
        guard !fatalDelivered else { lock.unlock(); return }
        fatalDelivered = true
        lock.unlock()
        onFatal(message)
    }
}
