import CoreML
import Foundation
import OSLog

/// Loads Core ML functions in named sets under a Neural Engine program budget.
///
/// The Neural Engine holds a limited number of loaded programs for the whole machine (about
/// 124 were available on the qualification host, M4 Max / macOS 27, and every process's
/// models count against that pool). When a load finds no free program slot, Core ML either
/// fails the load or quietly runs the function on the CPU. This type keeps the server's
/// footprint bounded: the packed text stacks stay resident, while long-input functions,
/// attention buckets and media towers are loaded as sets when a request needs them and evicted
/// least-recently-used when the budget would be exceeded. In `ane` mode loads are verified
/// against this process's Core ML log for Neural Engine program-creation errors (synchronously
/// for the startup load of every set, in the background for later on-demand loads), so a
/// silent CPU fallback fails startup or is evicted and reloaded instead of lingering.
///
/// Not thread-safe: the owning backend actor serializes all use.
final class ProgramResidency: @unchecked Sendable {
    struct Function: Hashable, Sendable {
        let path: String
        let name: String?
    }

    enum Failure: Error, CustomStringConvertible {
        case budget(String, Int, Int)
        case neuralEngine(String)

        var description: String {
            switch self {
            case let .budget(set, needed, budget):
                "program set \(set) needs \(needed) Neural Engine programs; the budget is \(budget)"
            case let .neuralEngine(set):
                "the Neural Engine could not load \(set) (usually no free program slots: other processes may be "
                    + "holding Neural Engine models); refusing to run it on the CPU in ane mode"
            }
        }
    }

    private struct Resident {
        let models: [Function: MLModel]
        var lastUse: UInt64
        let pinned: Bool
    }

    let root: URL
    let mode: ComputeMode
    /// Maximum programs this process keeps loaded (Neural Engine mode); unlimited otherwise.
    let budget: Int
    /// When false (fixtures), Core ML's log is not inspected after loads.
    let verifyNeuralEngine: Bool
    private var sets: [String: Resident] = [:]
    private var clock: UInt64 = 0
    private(set) var loads = 0
    private(set) var evictions = 0
    /// Sets loaded since the last verification, with their load start times.
    private(set) var unverified: [(set: String, since: Date)] = []

    init(root: URL, mode: ComputeMode, budget: Int, verifyNeuralEngine: Bool) {
        self.root = root
        self.mode = mode
        self.budget = mode == .ane ? budget : Int.max
        self.verifyNeuralEngine = verifyNeuralEngine && mode == .ane
    }

    var residentPrograms: Int { sets.values.reduce(0) { $0 + $1.models.count } }
    var residentSets: [String] { sets.keys.sorted() }

    /// Ensure `functions` are loaded as set `name` and return them. `inUse` names sets that
    /// must not be evicted to make room (the caller's other active sets).
    func acquire(_ name: String, _ functions: [Function], pinned: Bool = false,
                 inUse: Set<String> = []) throws -> [Function: MLModel] {
        clock += 1
        if var resident = sets[name] {
            resident.lastUse = clock
            sets[name] = resident
            return resident.models
        }
        let needed = Set(functions).count
        let protected = inUse.union([name])
        guard needed + sets.filter({ $0.value.pinned || protected.contains($0.key) })
            .reduce(0, { $0 + $1.value.models.count }) <= budget else {
            throw Failure.budget(name, needed, budget)
        }
        while residentPrograms + needed > budget, evictOldest(except: protected) {}
        let models: [Function: MLModel]
        do {
            models = try load(name, functions)
        } catch Failure.neuralEngine {
            // Free every unprotected set and retry once; if nothing could be freed, the missing
            // slots are held elsewhere and the retry reports the failure.
            var freed = false
            while evictOldest(except: protected) { freed = true }
            guard freed else { throw Failure.neuralEngine(name) }
            models = try load(name, functions)
        }
        sets[name] = Resident(models: models, lastUse: clock, pinned: pinned)
        return models
    }

    func release(_ name: String) {
        if sets.removeValue(forKey: name) != nil { evictions += 1 }
    }

    private func evictOldest(except protected: Set<String>) -> Bool {
        guard let victim = sets.filter({ !$0.value.pinned && !protected.contains($0.key) })
            .min(by: { $0.value.lastUse < $1.value.lastUse })?.key else { return false }
        sets.removeValue(forKey: victim)
        evictions += 1
        return true
    }

    private func load(_ name: String, _ functions: [Function]) throws -> [Function: MLModel] {
        let started = Date()
        var models = [Function: MLModel]()
        for function in functions where models[function] == nil {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = mode.units
            if let fn = function.name { configuration.functionName = fn }
            do {
                models[function] = try MLModel(contentsOf: root.appendingPathComponent(function.path),
                                               configuration: configuration)
            } catch {
                // Measured: with no free program slot, a multifunction load fails with a
                // misleading "functionName must be nil" error.
                if verifyNeuralEngine, Self.neuralEngineErrors(since: started) { throw Failure.neuralEngine(name) }
                throw error
            }
        }
        loads += 1
        if verifyNeuralEngine { unverified.append((name, started)) }
        return models
    }

    /// Hand the pending verifications to the caller (which checks the log off the hot path).
    func takeUnverified() -> [(set: String, since: Date)] {
        defer { unverified = [] }
        return unverified
    }

    /// Whether Core ML logged a Neural Engine runtime (E5RT) error in this process since `date`.
    /// Reading the process log costs on the order of a second or more (it grows with the log),
    /// so callers batch it.
    /// Measured: when the Neural Engine has no free program slots, `aned` fails program creation
    /// and the client process logs `E5RT: ... (13)` from `com.apple.coreml`, after which a
    /// single-function model may load successfully but execute on the CPU.
    static func neuralEngineErrors(since date: Date) -> Bool {
        guard let store = try? OSLogStore(scope: .currentProcessIdentifier) else { return false }
        let position = store.position(date: date)
        let predicate = NSPredicate(format: "subsystem == %@ OR subsystem == %@", "com.apple.coreml", "com.apple.e5rt")
        guard let entries = try? store.getEntries(at: position, matching: predicate) else { return false }
        // The position is only a hint for this store (entries before it are still returned), so
        // filter by date explicitly: earlier errors (for example from placement audits) are not
        // load failures.
        for case let entry as OSLogEntryLog in entries
        where entry.date >= date && (entry.level == .error || entry.level == .fault) {
            if entry.subsystem == "com.apple.e5rt" || entry.composedMessage.contains("E5RT") { return true }
        }
        return false
    }
}
