import Foundation

/// A cache whose values are expensive to build (a Core ML function load can take seconds) and
/// must be built exactly once per key.
///
/// The naive shape — take one lock, look up, build, store, unlock — serializes every key behind
/// the slowest build: while `bucket_32768` loads, a request for `bucket_32` waits on the same lock
/// even though it is already resident or independent. This cache holds its lock only for the
/// dictionary bookkeeping and never across `load`:
///
/// * the first caller for a key becomes that key's loader and runs `load` with no lock held;
/// * concurrent callers for the SAME key block on that key's own condition until the loader
///   finishes and then share its result — the value is built once, never twice;
/// * callers for OTHER keys are never blocked by it, and independent keys load in parallel;
/// * a failed load is not cached: callers that were waiting on that attempt receive its error, and
///   the next caller retries from scratch (a transient failure must not poison the key forever).
///
/// `load` must not call back into the cache for the same key (that would wait on itself).
///
/// The cache is `@unchecked Sendable`: its own state is guarded by `lock` and each value is
/// published exactly once, after which it is only read. `Value` need not be `Sendable` because the
/// loaded objects here (`MLModel`, small immutable wrappers) are safe for concurrent prediction.
final class KeyedLoadCache<Key: Hashable, Value>: @unchecked Sendable {
    /// One in-flight load. Waiters block on `condition` until `result` is set.
    private final class Flight {
        private let condition = NSCondition()
        private var result: Result<Value, any Error>?

        func finish(_ outcome: Result<Value, any Error>) {
            condition.lock()
            result = outcome
            condition.broadcast()
            condition.unlock()
        }

        func wait() throws -> Value {
            condition.lock()
            defer { condition.unlock() }
            while true {
                if let result { return try result.get() }
                condition.wait()
            }
        }
    }

    private enum Slot {
        case loading(Flight)
        case loaded(Value)
    }

    private let lock = NSLock()
    private var slots: [Key: Slot] = [:]

    init() {}

    /// The value for `key`, loading it with `load` if nobody has yet. See the type documentation
    /// for the concurrency contract.
    func value(for key: Key, load: () throws -> Value) throws -> Value {
        lock.lock()
        switch slots[key] {
        case let .loaded(value)?:
            lock.unlock()
            return value
        case let .loading(flight)?:
            lock.unlock()
            return try flight.wait()
        case nil:
            let flight = Flight()
            slots[key] = .loading(flight)
            lock.unlock()

            let outcome = Result { try load() }
            lock.lock()
            switch outcome {
            case let .success(value): slots[key] = .loaded(value)
            case .failure: slots[key] = nil
            }
            lock.unlock()
            flight.finish(outcome)
            return try outcome.get()
        }
    }

    /// The value if it is already loaded; never waits for an in-flight load.
    func loadedValue(for key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        if case let .loaded(value)? = slots[key] { return value }
        return nil
    }

    /// Publish an already-built value (for example a model constructed eagerly by an initializer).
    /// An existing value or in-flight load for `key` is left untouched.
    func prime(_ value: Value, for key: Key) {
        lock.lock()
        defer { lock.unlock() }
        if slots[key] == nil { slots[key] = .loaded(value) }
    }

    /// Keys whose values are loaded right now.
    var loadedKeys: [Key] {
        lock.lock()
        defer { lock.unlock() }
        return slots.compactMap { key, slot in
            if case .loaded = slot { return key }
            return nil
        }
    }
}
