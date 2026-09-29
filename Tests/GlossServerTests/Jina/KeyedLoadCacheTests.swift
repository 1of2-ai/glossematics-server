import Foundation
import Testing
@testable import gloss_server

/// The per-key once-only load cache behind every lazily loaded Core ML function. The regression it
/// guards: one global lock held across a multi-second `MLModel` load stalls every other bucket.

private struct LoadFailure: Error, Equatable { let attempt: Int }

/// A thread-safe counter for load invocations.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    @discardableResult func increment() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
    var current: Int { lock.lock(); defer { lock.unlock() }; return value }
}

/// `DispatchSemaphore.wait()` is unavailable directly in async code; a synchronous helper is the
/// deliberate escape hatch for these thread-level tests.
private func waitSynchronously(_ semaphore: DispatchSemaphore) { semaphore.wait() }

@Suite struct KeyedLoadCacheTests {
    @Test func concurrentCallersForOneKeyLoadItExactlyOnce() async throws {
        let cache = KeyedLoadCache<Int, String>()
        let loads = Counter()
        let results = try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<32 {
                group.addTask {
                    try cache.value(for: 7) {
                        loads.increment()
                        Thread.sleep(forTimeInterval: 0.05)   // wide race window
                        return "loaded"
                    }
                }
            }
            return try await group.reduce(into: [String]()) { $0.append($1) }
        }
        #expect(results.count == 32 && results.allSatisfy { $0 == "loaded" })
        #expect(loads.current == 1, "a key must be built once no matter how many callers race")
        // Later callers hit the cache without loading.
        #expect(try cache.value(for: 7) { loads.increment(); return "again" } == "loaded")
        #expect(loads.current == 1)
    }

    @Test func aSlowLoadDoesNotBlockOtherKeys() async throws {
        let cache = KeyedLoadCache<Int, String>()
        let slowStarted = DispatchSemaphore(value: 0)
        let releaseSlow = DispatchSemaphore(value: 0)

        let slow = Task.detached {
            try cache.value(for: 32_768) {
                slowStarted.signal()
                releaseSlow.wait()          // "bucket_32768" is still loading...
                return "slow"
            }
        }
        waitSynchronously(slowStarted)

        // ...while an unrelated key must load and return immediately.
        let started = ContinuousClock.now
        let fast = try cache.value(for: 32) { "fast" }
        let elapsed = ContinuousClock.now - started
        #expect(fast == "fast")
        #expect(elapsed < .seconds(2), "another key waited \(elapsed) behind an in-flight load")

        // A second caller for the slow key waits for the SAME load instead of starting another.
        let waiter = Task.detached { try cache.value(for: 32_768) { "duplicate" } }
        try await Task.sleep(for: .milliseconds(50))
        releaseSlow.signal()
        #expect(try await slow.value == "slow")
        #expect(try await waiter.value == "slow")
    }

    @Test func independentKeysLoadInParallel() async throws {
        let cache = KeyedLoadCache<Int, Int>()
        let arrived = Counter()
        let bothArrived = DispatchSemaphore(value: 0)
        // Each load waits until the OTHER has started: only possible if loads overlap.
        async let first = Task.detached {
            try cache.value(for: 1) {
                if arrived.increment() == 2 { bothArrived.signal(); bothArrived.signal() }
                guard bothArrived.wait(timeout: .now() + 5) == .success else { throw LoadFailure(attempt: -1) }
                return 1
            }
        }.value
        async let second = Task.detached {
            try cache.value(for: 2) {
                if arrived.increment() == 2 { bothArrived.signal(); bothArrived.signal() }
                guard bothArrived.wait(timeout: .now() + 5) == .success else { throw LoadFailure(attempt: -2) }
                return 2
            }
        }.value
        #expect(try await first == 1)
        #expect(try await second == 2, "loads of different keys must overlap, not serialize")
    }

    @Test func aFailedLoadIsNotCachedAndWaitersShareItsError() async throws {
        let cache = KeyedLoadCache<Int, String>()
        let attempts = Counter()
        let firstStarted = DispatchSemaphore(value: 0)
        let releaseFirst = DispatchSemaphore(value: 0)

        let loader = Task.detached {
            try cache.value(for: 5) {
                let attempt = attempts.increment()
                firstStarted.signal()
                releaseFirst.wait()
                throw LoadFailure(attempt: attempt)
            }
        }
        waitSynchronously(firstStarted)
        let waiter = Task.detached { try cache.value(for: 5) { "should not run" } }
        try await Task.sleep(for: .milliseconds(50))
        releaseFirst.signal()

        // The loader and the caller that was waiting on that attempt both see the failure.
        await #expect(throws: LoadFailure(attempt: 1)) { try await loader.value }
        await #expect(throws: LoadFailure(attempt: 1)) { try await waiter.value }
        #expect(attempts.current == 1)

        // The failure was not cached: the next caller retries and can succeed.
        #expect(try cache.value(for: 5) { attempts.increment(); return "recovered" } == "recovered")
        #expect(attempts.current == 2)
        #expect(cache.loadedValue(for: 5) == "recovered")
    }

    @Test func primeAndPeekNeverWaitOrOverwrite() throws {
        let cache = KeyedLoadCache<String, Int>()
        #expect(cache.loadedValue(for: "a") == nil)
        cache.prime(1, for: "a")
        cache.prime(2, for: "a")
        #expect(cache.loadedValue(for: "a") == 1, "priming must not replace an existing value")
        #expect(try cache.value(for: "a") { 3 } == 1)
        #expect(try cache.value(for: "b") { 4 } == 4)
        #expect(Set(cache.loadedKeys) == ["a", "b"])
    }
}
