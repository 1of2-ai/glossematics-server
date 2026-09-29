import Foundation

enum BuildInfo {
    static let version = "0.2.0"
    static let serverHeader = "gloss-server/\(version)"
}

enum RetrievalRole: String, Sendable, Hashable {
    case query
    case document
}

struct RuntimeSnapshot: Sendable {
    var phase: String
    var ready: Bool
    var startedAt: Date
    var readyAt: Date?
    var startupError: String?
    var lastKeepWarmAt: Date?
    var lastKeepWarmError: String?
    var keepWarmFailures: Int
}

actor RuntimeState {
    private enum Phase {
        case starting
        case ready
        case failed(String)
        case shuttingDown
    }

    private var phase: Phase = .starting
    private let startedAt = Date()
    private var readyAt: Date?
    private var lastKeepWarmAt: Date?
    private var lastKeepWarmError: String?
    private var keepWarmFailures = 0

    func markReady() {
        guard case .starting = phase else { return }
        phase = .ready
        readyAt = Date()
    }

    func failStartup(_ error: String) {
        guard case .starting = phase else { return }
        phase = .failed(error)
    }

    func beginShutdown() { phase = .shuttingDown }

    func recordKeepWarmSuccess() {
        lastKeepWarmAt = Date()
        lastKeepWarmError = nil
    }

    func recordKeepWarmFailure(_ error: String) {
        lastKeepWarmAt = Date()
        lastKeepWarmError = error
        keepWarmFailures += 1
    }

    func isReady() -> Bool {
        if case .ready = phase { return true }
        return false
    }

    func snapshot() -> RuntimeSnapshot {
        let name: String
        let error: String?
        let ready: Bool
        switch phase {
        case .starting:
            name = "starting"; error = nil; ready = false
        case .ready:
            name = "ready"; error = nil; ready = true
        case let .failed(reason):
            name = "failed"; error = reason; ready = false
        case .shuttingDown:
            name = "shutting_down"; error = nil; ready = false
        }
        return RuntimeSnapshot(
            phase: name,
            ready: ready,
            startedAt: startedAt,
            readyAt: readyAt,
            startupError: error,
            lastKeepWarmAt: lastKeepWarmAt,
            lastKeepWarmError: lastKeepWarmError,
            keepWarmFailures: keepWarmFailures)
    }
}

/// Whole-logical-request admission bound. Reservations stay live through response socket completion.
actor AdmissionGate {
    private let maxRequests: Int
    private let maxItems: Int
    private var requests = 0
    private var items = 0

    init(maxRequests: Int, maxItems: Int) {
        self.maxRequests = maxRequests
        self.maxItems = maxItems
    }

    func tryAcquire(items requestedItems: Int) -> Bool {
        guard requestedItems > 0,
              requests + 1 <= maxRequests,
              items + requestedItems <= maxItems else { return false }
        requests += 1
        items += requestedItems
        return true
    }

    func release(items releasedItems: Int) {
        requests = max(0, requests - 1)
        items = max(0, items - max(0, releasedItems))
    }

    func snapshot() -> (requests: Int, items: Int, maxRequests: Int, maxItems: Int) {
        (requests, items, maxRequests, maxItems)
    }
}

/// Single expensive accelerator permit across text/media/warmup/maintenance.
actor AcceleratorLane {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, any Error>
    }

    private var held = false
    private var waiters: [Waiter] = []

    func acquire() async throws {
        try Task.checkCancellation()
        if !held {
            held = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                guard !Task.isCancelled else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters.append(.init(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
        // Cancellation can race with permit transfer. If release() resumed us just before the
        // cancellation callback reached the actor, we own the permit and must pass it onward.
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    func tryAcquire() -> Bool {
        guard !held else { return false }
        held = true
        return true
    }

    func release() {
        if !waiters.isEmpty {
            let next = waiters.removeFirst()
            next.continuation.resume()
            return
        }
        held = false
    }

    private func cancel(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }
}

struct MetricsSnapshot: Sendable {
    var httpRequests: UInt64 = 0
    var embeddingRequests: UInt64 = 0
    var embeddingItems: UInt64 = 0
    var rejectedRequests: UInt64 = 0
    var failedRequests: UInt64 = 0
    var wavesByKind: [String: UInt64] = [:]
    var rowsByKind: [String: UInt64] = [:]
    var tokensByKind: [String: UInt64] = [:]
    var coalescedWaves: UInt64 = 0
    /// Text waves that failed as a batch and were re-run one row at a time (Jina), and the rows
    /// that still failed on their own.
    var isolationRetries: UInt64 = 0
    var isolationFailedRows: UInt64 = 0
    var longDocuments: UInt64 = 0
    var longTokens: UInt64 = 0
    var longSteps: UInt64 = 0
    var mediaItems: [String: UInt64] = [:]
    var keepWarmAttempts: UInt64 = 0
    var keepWarmFailures: UInt64 = 0
    var lastInferenceUnix: Double?
}

actor ServerMetrics {
    private var value = MetricsSnapshot()

    func recordHTTP() { value.httpRequests &+= 1 }
    func recordEmbeddingRequest(items: Int) {
        value.embeddingRequests &+= 1
        value.embeddingItems &+= UInt64(max(0, items))
    }
    func recordRejected() { value.rejectedRequests &+= 1 }
    func recordFailure() { value.failedRequests &+= 1 }
    func recordWave(kind: String, rows: Int, tokens: Int, requests: Int) {
        value.wavesByKind[kind, default: 0] &+= 1
        value.rowsByKind[kind, default: 0] &+= UInt64(max(0, rows))
        value.tokensByKind[kind, default: 0] &+= UInt64(max(0, tokens))
        if requests > 1 { value.coalescedWaves &+= 1 }
        value.lastInferenceUnix = Date().timeIntervalSince1970
    }
    func recordIsolation(failedRows: Int) {
        value.isolationRetries &+= 1
        value.isolationFailedRows &+= UInt64(max(0, failedRows))
    }
    func recordLong(tokens: Int, steps: Int) {
        value.longDocuments &+= 1
        value.longTokens &+= UInt64(max(0, tokens))
        value.longSteps &+= UInt64(max(0, steps))
        value.lastInferenceUnix = Date().timeIntervalSince1970
    }
    func recordMedia(kind: String) {
        value.mediaItems[kind, default: 0] &+= 1
        value.lastInferenceUnix = Date().timeIntervalSince1970
    }
    func recordInference() { value.lastInferenceUnix = Date().timeIntervalSince1970 }
    func recordKeepWarm(success: Bool) {
        value.keepWarmAttempts &+= 1
        if !success { value.keepWarmFailures &+= 1 }
    }
    func snapshot() -> MetricsSnapshot { value }
}
