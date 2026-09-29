import CoreML
import Foundation

/// Operation-level placement of every loaded function, from Core ML's compute plan.
///
/// In `ane` mode the daemon refuses to become ready unless every non-constant operation of
/// every function prefers the Neural Engine, lists it as supported, and has a finite cost.
/// `.cpuAndNeuralEngine` alone permits silent CPU fallback (for example after an OS update
/// changes the ANE compiler), so this check is what makes the ANE contract enforceable.
/// In `gpu` and `cpu` modes the plan is summarized for health reporting only.
struct PlacementReport: Sendable {
    struct Function: Sendable {
        let name: String
        let operations: Int
        let costByDevice: [String: Double]
        let strictANE: Bool
        let problem: String?
        /// The plan reported no device for any operation. Measured cause: a stale entry in
        /// Core ML's per-process compiled-model (E5) cache; a fresh compile places the same
        /// function on the Neural Engine.
        var indeterminate: Bool = false
    }

    let mode: ComputeMode
    let functions: [Function]

    var strictANE: Bool { functions.allSatisfy(\.strictANE) }

    /// Fraction of estimated cost placed on each device, across all functions.
    var costShare: [String: Double] {
        var totals = [String: Double]()
        for fn in functions { for (device, cost) in fn.costByDevice { totals[device, default: 0] += cost } }
        let sum = totals.values.reduce(0, +)
        guard sum > 0 else { return [:] }
        return totals.mapValues { $0 / sum }
    }

    /// Every failing function lacks placement information (as opposed to preferring the CPU/GPU).
    var failuresAreIndeterminate: Bool {
        let failing = functions.filter { !$0.strictANE }
        return !failing.isEmpty && failing.allSatisfy(\.indeterminate)
    }

    var firstProblem: String? {
        functions.first { !$0.strictANE }.map { "\($0.name): \($0.problem ?? "not placed on the Neural Engine")" }
    }
}

struct AuditError: Error, CustomStringConvertible {
    let function: String
    let underlying: String
    var description: String { "compute plan for \(function) could not be built: \(underlying)" }
}

enum PlacementAudit {
    static func device(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: "cpu"
        case .gpu: "gpu"
        case .neuralEngine: "ane"
        @unknown default: "unknown"
        }
    }

    static func audit(model url: URL, function: String, mode: ComputeMode) async throws -> PlacementReport.Function {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = mode.units
        configuration.functionName = function
        let plan: MLComputePlan
        do {
            plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        } catch {
            throw AuditError(function: "\(url.lastPathComponent):\(function)", underlying: String(describing: error))
        }
        guard case let .program(program) = plan.modelStructure, let body = program.functions[function] else {
            return .init(name: "\(url.lastPathComponent):\(function)", operations: 0, costByDevice: [:],
                         strictANE: false, problem: "compute plan has no ML Program function")
        }
        var operations = 0, unknown = 0, offANE = 0, unsupported = 0
        var costs = [String: Double]()
        var firstIssue: String?
        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                if op.operatorName == "const" || op.operatorName.hasSuffix(".const") { continue }
                // W8A16 weights are int8 constants expanded by `constexpr_*` decompression ops at
                // load time; like `const`, they have no execution device in the compute plan.
                if op.operatorName.contains("constexpr_"), plan.deviceUsage(for: op) == nil { continue }
                guard let usage = plan.deviceUsage(for: op) else {
                    unknown += 1
                    if firstIssue == nil { firstIssue = "\(op.operatorName) has no placement" }
                    continue
                }
                operations += 1
                let preferred = device(usage.preferred)
                if let weight = plan.estimatedCost(of: op)?.weight, weight.isFinite, weight >= 0 {
                    costs[preferred, default: 0] += weight
                } else {
                    unknown += 1
                    if firstIssue == nil { firstIssue = "\(op.operatorName) has no cost estimate" }
                }
                if preferred != "ane" {
                    offANE += 1
                    if firstIssue == nil { firstIssue = "\(op.operatorName) prefers \(preferred)" }
                }
                if !usage.supported.contains(where: { device($0) == "ane" }) {
                    unsupported += 1
                    if firstIssue == nil { firstIssue = "\(op.operatorName) is not supported on the Neural Engine" }
                }
                for nested in op.blocks { walk(nested) }
            }
        }
        walk(body.block)
        let total = costs.values.reduce(0, +)
        let strict = operations > 0 && total > 0 && unknown == 0 && offANE == 0 && unsupported == 0
        return .init(name: "\(url.lastPathComponent):\(function)", operations: operations,
                     costByDevice: costs, strictANE: strict, problem: strict ? nil : firstIssue ?? "no measurable cost",
                     indeterminate: operations == 0 && unknown > 0)
    }

    /// This process's Core ML compiled-model cache (`~/Library/Caches/<name>/com.apple.e5rt.e5bundlecache`).
    static var compiledModelCache: URL? {
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return nil }
        let owner = Bundle.main.bundleIdentifier ?? ProcessInfo.processInfo.processName
        return caches.appendingPathComponent(owner).appendingPathComponent("com.apple.e5rt.e5bundlecache")
    }

    /// Remove this process's own compiled-model cache so the next load recompiles from the
    /// bundle. Returns false when there was nothing to remove.
    static func purgeCompiledModelCache() -> Bool {
        guard let url = compiledModelCache, FileManager.default.fileExists(atPath: url.path) else { return false }
        return (try? FileManager.default.removeItem(at: url)) != nil
    }
}
