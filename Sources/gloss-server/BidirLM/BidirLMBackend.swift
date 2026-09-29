import Foundation

/// The resident BidirLM model: validated bundle, tokenizer, and Core ML functions loaded for one
/// operator-selected compute mode. Every Core ML call is made from this actor, and the daemon
/// additionally serializes callers on the accelerator lane.
actor BidirLMBackend {
    /// Lock-protected copy of the backend's reportable state, so `/health` and `/metrics`
    /// answer immediately even while the actor is busy (for example during a multi-minute
    /// first start that compiles every function).
    final class Status: @unchecked Sendable {
        struct Snapshot: Sendable {
            var placement: PlacementReport?
            var loadSeconds: Double?
            var programs = 0, loads = 0, evictions = 0, fallbacks = 0
            var sets: [String] = []
        }
        private let lock = NSLock()
        private var value = Snapshot()
        var snapshot: Snapshot { lock.lock(); defer { lock.unlock() }; return value }
        func update(_ body: (inout Snapshot) -> Void) { lock.lock(); body(&value); lock.unlock() }
    }

    nonisolated let status = Status()

    enum Failure: Error, CustomStringConvertible {
        case notLoaded
        case placement(String)
        case invalidEmbedding(String)

        var description: String {
            switch self {
            case .notLoaded: "model functions are not loaded"
            case let .placement(reason): "Neural Engine placement check failed: \(reason)"
            case let .invalidEmbedding(reason): "invalid embedding: \(reason)"
            }
        }
    }

    nonisolated let bundle: BidirLMBundle
    nonisolated let mode: ComputeMode
    nonisolated let tokenizer: BidirLMTokenizer
    /// Most programs this process keeps loaded on the Neural Engine (shared system-wide limit).
    nonisolated let programBudget: Int
    private var residency: ProgramResidency?
    private var text: (any TextModelEncoder)?
    private var media: BidirLMMediaEncoder?
    private(set) var placement: PlacementReport?
    private(set) var loadSeconds: Double?
    /// Times startup discarded this process's compiled-model cache after an indeterminate audit.
    private(set) var compiledCacheResets = 0
    /// On-demand loads that Core ML's log showed falling back from the Neural Engine.
    private(set) var neuralEngineFallbacks = 0
    private var verifying = false

    init(bundle: BidirLMBundle, mode: ComputeMode, tokenizer: BidirLMTokenizer, programBudget: Int = 64) {
        self.bundle = bundle
        self.mode = mode
        self.tokenizer = tokenizer
        self.programBudget = programBudget
    }

    nonisolated var spaceID: String { bundle.manifest.spaceID }
    nonisolated var supportsPackedText: Bool { bundle.manifest.streamed == nil }
    nonisolated var isFixture: Bool { bundle.manifest.isFixture }
    /// Media towers are served only where they are qualified. Measured: Core ML's CPU backend
    /// runs these FP16 programs with FP16 accumulation, and the vision tower drifts to image
    /// embedding cosine 0.96-0.98 against the FP32 source (the Neural Engine and GPU keep
    /// >= 0.9999), so `cpu` mode serves text only.
    nonisolated var mediaQualified: Bool { mode != .cpu }
    nonisolated var hasVision: Bool { bundle.manifest.vision != nil && mediaQualified }
    nonisolated var hasAudio: Bool { bundle.manifest.audio != nil && mediaQualified }

    /// Load every function for the selected mode, then check placement. In ANE mode a single
    /// function that does not place entirely on the Neural Engine fails startup; the fixture
    /// (no-op programs) is exempt because its trivial graphs are not ANE-shaped.
    ///
    /// If every failing function merely lacks placement information, the process's own
    /// compiled-model cache is discarded and the load is retried once: a stale cache entry
    /// was measured to hide the placement of a function that compiles cleanly for the ANE.
    func prepare(verifyPlacement: Bool) async throws {
        let started = Date()
        var attempt = 0
        while true {
            if verifyPlacement {
                var functions = [PlacementReport.Function]()
                do {
                    for item in BidirLMBundle.functions(bundle.manifest) {
                        functions.append(try await PlacementAudit.audit(
                            model: bundle.root.appendingPathComponent(item.path), function: item.function, mode: mode))
                    }
                } catch let error as AuditError where attempt == 0 && PlacementAudit.purgeCompiledModelCache() {
                    // Same measured failure mode as an indeterminate plan: a stale compiled-model
                    // cache entry. Rebuild the cache once before failing startup.
                    attempt += 1
                    compiledCacheResets += 1
                    FileHandle.standardError.write(Data(
                        "gloss-server: \(error); discarded the compiled-model cache and retrying\n".utf8))
                    continue
                }
                let report = PlacementReport(mode: mode, functions: functions)
                if mode == .ane, !isFixture, let problem = report.firstProblem {
                    if attempt == 0, report.failuresAreIndeterminate, PlacementAudit.purgeCompiledModelCache() {
                        attempt += 1
                        compiledCacheResets += 1
                        FileHandle.standardError.write(Data(
                            "gloss-server: placement unknown for \(problem); discarded the compiled-model cache and retrying\n".utf8))
                        continue
                    }
                    throw Failure.placement(problem)
                }
                placement = report
            }
            break
        }
        let audited = Date()
        if verifyPlacement {
            FileHandle.standardError.write(Data(String(format: "gloss-server: placement audit %.1fs\n",
                                                       audited.timeIntervalSince(started)).utf8))
        }
        // Load every program set once so it is compiled and cached before the first request, then
        // check this process's Core ML log for Neural Engine load failures. Measured causes: no
        // free program slot, and a first-compile race in which `aned` cannot find the program it
        // just compiled ("cachedModelFilePath does not exist"). The second cause clears on a
        // reload from the now-populated cache, so a failed pass is retried once from scratch.
        var loaded: (ProgramResidency, any TextModelEncoder, BidirLMMediaEncoder?)?
        for pass in 0..<2 where loaded == nil {
            let residency = ProgramResidency(root: bundle.root, mode: mode, budget: programBudget,
                                             verifyNeuralEngine: !isFixture)
            let encoder: any TextModelEncoder
            if let streamed = bundle.manifest.streamed {
                encoder = try StreamedTextEncoder(residency: residency, manifest: streamed)
            } else {
                encoder = try BidirLMTextEncoder(residency: residency, manifest: bundle.manifest)
            }
            let towers = hasVision || hasAudio ? try BidirLMMediaEncoder(residency: residency, manifest: bundle.manifest) : nil
            for (name, functions) in BidirLMBundle.onDemandSets(bundle.manifest)
                where mediaQualified || name.hasPrefix("text.") {
                _ = try residency.acquire(name, functions)
                // Keep the long-input set: reloading its 29 programs costs ~12 s on first use.
                // Media towers and attention buckets reload from the cache quickly.
                if name != BidirLMTextEncoder.longSet { residency.release(name) }
            }
            if let first = residency.takeUnverified().first, ProgramResidency.neuralEngineErrors(since: first.since) {
                if pass == 1 { throw ProgramResidency.Failure.neuralEngine("the startup load of every program set") }
                FileHandle.standardError.write(Data(
                    "gloss-server: a Neural Engine program load failed during startup; reloading every set once\n".utf8))
                continue
            }
            loaded = (residency, encoder, towers)
        }
        guard let (residency, encoder, towers) = loaded else { throw Failure.notLoaded }
        FileHandle.standardError.write(Data(String(format: "gloss-server: program load %.1fs\n",
                                                   Date().timeIntervalSince(audited)).utf8))
        defer { publish() }
        self.residency = residency
        text = encoder
        media = towers
        loadSeconds = Date().timeIntervalSince(started)
    }

    // MARK: - language model

    /// Begin a resumable long execution (more than 512 tokens).
    func startLong(_ sequence: LMSequence) throws -> any TextModelRun {
        guard let text else { throw Failure.notLoaded }
        defer { verifyLoadsInBackground() }
        return try text.startLong(sequence)
    }

    /// Check the log for sets loaded since the last check, off the accelerator path. A set whose
    /// load fell back from the Neural Engine is evicted so its next use reloads it (its results
    /// were still correct: the same program ran on the CPU).
    private func verifyLoadsInBackground() {
        publish()
        guard let residency, !verifying else { return }
        let pending = residency.takeUnverified()
        guard let since = pending.map(\.since).min() else { return }
        verifying = true
        Task.detached(priority: .utility) { [weak self] in
            let failed = ProgramResidency.neuralEngineErrors(since: since)
            await self?.finishVerification(sets: pending.map(\.set), failed: failed)
        }
    }

    private func publish() {
        let placement = placement, loadSeconds = loadSeconds, fallbacks = neuralEngineFallbacks
        let r = residency
        status.update {
            $0.placement = placement
            $0.loadSeconds = loadSeconds
            $0.fallbacks = fallbacks
            $0.programs = r?.residentPrograms ?? 0
            $0.loads = r?.loads ?? 0
            $0.evictions = r?.evictions ?? 0
            $0.sets = r?.residentSets ?? []
        }
    }

    private func finishVerification(sets: [String], failed: Bool) {
        defer { publish() }
        verifying = false
        guard failed, let residency else { return }
        neuralEngineFallbacks += 1
        for name in sets { residency.release(name) }
        FileHandle.standardError.write(Data(
            "gloss-server: Neural Engine program creation failed while loading \(sets.joined(separator: ", ")); evicted for reload\n".utf8))
    }

    func startLong(tokens: [Int32]) throws -> any TextModelRun {
        try startLong(LMSequence(ids: tokens))
    }

    /// Advance a long run by one chunk operation; returns the embedding when complete.
    func step(_ run: any TextModelRun) throws -> [Float]? {
        guard try run.step() else { return nil }
        let vector = try run.result()
        guard vector.count == BidirLMContract.dimension, vector.allSatisfy(\.isFinite) else {
            throw Failure.invalidEmbedding("long input produced a non-finite vector")
        }
        return vector
    }

    /// Execute one plan-compatible group: up to 64 sequences totalling at most 512 tokens, or
    /// one sequence of any length up to 32768 tokens. Sequences must already be templated.
    func embed(sequences: [LMSequence]) throws -> [[Float]] {
        guard let text else { throw Failure.notLoaded }
        defer { verifyLoadsInBackground() }
        let vectors = try text.encode(sequences: sequences)
        guard vectors.count == sequences.count else {
            throw Failure.invalidEmbedding("\(vectors.count) vectors for \(sequences.count) sequences")
        }
        for (index, v) in vectors.enumerated() {
            guard v.count == BidirLMContract.dimension, v.allSatisfy(\.isFinite) else {
                throw Failure.invalidEmbedding("row \(index) is not a finite 2048-d vector")
            }
        }
        return vectors
    }

    func embed(tokenRows: [[Int32]]) throws -> [[Float]] {
        try embed(sequences: tokenRows.map { LMSequence(ids: $0) })
    }

    // MARK: - media towers

    func startTower(_ input: MediaInput) throws -> any BidirLMMediaEncoder.Run {
        guard let media else { throw Failure.notLoaded }
        defer { verifyLoadsInBackground() }
        switch input {
        case let .image(image): return try media.startImage(image)
        case let .audio(clip): return try media.startAudio(clip)
        }
    }

    /// Advance a tower run by one step; returns its features when complete.
    func step(_ run: any BidirLMMediaEncoder.Run) throws -> BidirLMMediaEncoder.Features? {
        defer { verifyLoadsInBackground() }
        guard try run.step() else { return nil }
        guard let features = run.features, features.rows.allSatisfy(\.isFinite) else {
            throw Failure.invalidEmbedding("media tower produced non-finite features")
        }
        return features
    }
}

/// One media item's host-prepared tower input.
enum MediaInput: Sendable {
    case image(PreparedImage)
    case audio(PreparedAudio)

    var tokens: Int {
        switch self {
        case let .image(image): image.tokens
        case let .audio(clip): clip.tokens
        }
    }
}
