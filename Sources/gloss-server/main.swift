import Foundation

enum AccessLogMode: String, Sendable {
    case off
    case errors
    case all
}

struct Arguments: Sendable {
    var bundle: URL?
    var port: UInt16 = 11_435
    var compute: ComputeMode = .ane
    var modelName: String?
    var maxBatch = 2048
    var maxBodyMB = 64
    var maxTotalBodyMB = 256
    var maxQueueRequests = 128
    var maxQueueItems = 8192
    var maxRequestTokens = 131_072
    var aneProgramBudget = 64
    var batchWindowMS = 2.0
    var keepWarmSeconds = 60
    var maxConnections = 256
    var ioTimeoutSeconds = 30
    var shutdownGraceSeconds = 15
    var accessLog: AccessLogMode = .errors
    var checkConfig = false
    var allowDummy = false
    /// Default output size (Jina Matryoshka); BidirLM accepts only its native 2048.
    var dimensions: Int?
    /// Jina: what startup proves before `/ready` (BidirLM always audits placement).
    var startupVerification: JinaStartupVerification = .full
    /// Flags given on the command line, for family-specific validation after detection.
    var explicit: Set<String> = []

    static func parse(_ args: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < args.count {
            let flag = args[index]
            parsed.explicit.insert(flag)
            func nextValue() throws -> String {
                index += 1
                guard index < args.count else { throw ArgumentError("missing value for \(flag)") }
                return args[index]
            }
            switch flag {
            case "--bundle", "-b":
                parsed.bundle = URL(fileURLWithPath: try nextValue(), isDirectory: true)
            case "--port", "-p":
                guard let value = UInt16(try nextValue()), value > 0 else {
                    throw ArgumentError("--port expects 1...65535")
                }
                parsed.port = value
            case "--compute", "-c":
                let raw = try nextValue()
                guard let mode = ComputeMode(rawValue: raw) else {
                    throw ArgumentError("--compute must be ane|gpu|cpu")
                }
                parsed.compute = mode
            case "--dimensions", "-d":
                guard let value = Int(try nextValue()), value > 0 else {
                    throw ArgumentError("--dimensions expects a positive integer")
                }
                parsed.dimensions = value
            case "--startup-verification":
                let raw = try nextValue()
                guard let mode = JinaStartupVerification(rawValue: raw) else {
                    throw ArgumentError("--startup-verification must be full|basic")
                }
                parsed.startupVerification = mode
            case "--model-name", "-m":
                let value = try nextValue()
                guard !value.isEmpty, value.count <= 256,
                      value.rangeOfCharacter(from: .controlCharacters) == nil else {
                    throw ArgumentError("--model-name must be 1...256 printable characters")
                }
                parsed.modelName = value
            case "--max-batch": parsed.maxBatch = try boundedInt(try nextValue(), 1...2048, flag)
            case "--max-body-mb": parsed.maxBodyMB = try boundedInt(try nextValue(), 1...1024, flag)
            case "--max-total-body-mb": parsed.maxTotalBodyMB = try boundedInt(try nextValue(), 1...4096, flag)
            case "--max-queue-requests": parsed.maxQueueRequests = try boundedInt(try nextValue(), 1...4096, flag)
            case "--max-queue-items": parsed.maxQueueItems = try boundedInt(try nextValue(), 1...131_072, flag)
            case "--ane-program-budget": parsed.aneProgramBudget = try boundedInt(try nextValue(), 40...120, flag)
            case "--max-request-tokens":
                parsed.maxRequestTokens = try boundedInt(try nextValue(), BidirLMContract.maxTokens...4_194_304, flag)
            case "--batch-window-ms":
                guard let value = Double(try nextValue()), value.isFinite, (0...25).contains(value) else {
                    throw ArgumentError("--batch-window-ms must be 0...25")
                }
                parsed.batchWindowMS = value
            case "--keep-warm-seconds": parsed.keepWarmSeconds = try boundedInt(try nextValue(), 0...86_400, flag)
            case "--max-connections": parsed.maxConnections = try boundedInt(try nextValue(), 1...4096, flag)
            case "--idle-timeout-seconds": parsed.ioTimeoutSeconds = try boundedInt(try nextValue(), 1...3600, flag)
            case "--shutdown-grace-seconds": parsed.shutdownGraceSeconds = try boundedInt(try nextValue(), 0...300, flag)
            case "--access-log":
                guard let mode = AccessLogMode(rawValue: try nextValue()) else {
                    throw ArgumentError("--access-log must be off|errors|all")
                }
                parsed.accessLog = mode
            case "--check-config": parsed.checkConfig = true
            case "--allow-dummy": parsed.allowDummy = true
            case "--version": print(BuildInfo.version); exit(0)
            case "--help", "-h": print(helpText); exit(0)
            default: throw ArgumentError("unknown argument: \(flag)")
            }
            index += 1
        }
        guard parsed.maxTotalBodyMB >= parsed.maxBodyMB else {
            throw ArgumentError("--max-total-body-mb must be >= --max-body-mb")
        }
        guard parsed.maxQueueItems >= parsed.maxBatch else {
            throw ArgumentError("--max-queue-items must be >= --max-batch")
        }
        return parsed
    }

    private static func boundedInt(_ raw: String, _ range: ClosedRange<Int>, _ flag: String) throws -> Int {
        guard let value = Int(raw), range.contains(value) else {
            throw ArgumentError("\(flag) must be in \(range.lowerBound)...\(range.upperBound)")
        }
        return value
    }

    static let helpText = """
    gloss-server — local OpenAI-compatible embeddings microservice
    usage: gloss-server --bundle <dir> [options]
    The model family is detected from the bundle manifest: BidirLM Omni or
    jina-embeddings-v5-omni-small.
      --compute ane|gpu|cpu (ane)  BidirLM: operator-selected Core ML placement; never falls back
      --dimensions N  Jina: default Matryoshka size (1024; 32|64|128|256|512|1024). BidirLM: 2048 only
      --startup-verification full|basic (full)  Jina: full loads and runs every Core ML function
          once, cross-checked, before /ready (slower start; a fault keeps the server not ready);
          basic runs one warm text embedding
      --port N (11435)  --model-name ID
      --max-batch N (2048)  --max-queue-requests N (128)  --max-queue-items N (8192)
      --max-request-tokens N (131072)  --batch-window-ms N (2.0)  --keep-warm-seconds N (60)
      --ane-program-budget N (64; BidirLM: Neural Engine programs kept loaded, 40...120)
      --max-body-mb N (64)  --max-total-body-mb N (256)  --max-connections N (256)
      --idle-timeout-seconds N (30; bounds stalled reads/writes)
      --shutdown-grace-seconds N (15)  --access-log off|errors|all
      --check-config  --allow-dummy (real Core ML golden fixture)  --version  --help
    """
}

struct ArgumentError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func timestamped(_ message: String, error: Bool = false) {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let line = "\(formatter.string(from: Date())) \(message)\n"
    (error ? FileHandle.standardError : FileHandle.standardOutput).write(Data(line.utf8))
}

func elapsedMilliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1000
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000
}

func makeDocsContext(arguments: Arguments, bundle: BidirLMBundle, modalities: [String]) -> DocsPage.Context {
    DocsPage.Context(
        baseURL: "http://127.0.0.1:\(arguments.port)",
        modelID: arguments.modelName ?? bundle.manifest.modelID,
        dimensions: BidirLMContract.dimension,
        maxTokens: BidirLMContract.maxTokens,
        compute: arguments.compute.rawValue,
        modalities: modalities,
        maxBatch: arguments.maxBatch,
        maxRequestTokens: arguments.maxRequestTokens,
        maxBodyMB: arguments.maxBodyMB,
        maxTotalBodyMB: arguments.maxTotalBodyMB,
        maxQueueRequests: arguments.maxQueueRequests,
        maxQueueItems: arguments.maxQueueItems,
        batchWindowMS: arguments.batchWindowMS,
        keepWarmSeconds: arguments.keepWarmSeconds,
        maxConnections: arguments.maxConnections,
        ioTimeoutSeconds: arguments.ioTimeoutSeconds,
        shutdownGraceSeconds: arguments.shutdownGraceSeconds,
        accessLogMode: arguments.accessLog.rawValue,
        spaceID: bundle.manifest.spaceID,
        isDummy: bundle.manifest.isFixture)
}

/// Load every function for the selected compute mode, verify placement (ANE mode), then run
/// one real text inference before reporting readiness.
func performStartup(backend: BidirLMBackend, lane: AcceleratorLane, metrics: ServerMetrics) async throws {
    try await lane.acquire()
    do {
        try await backend.prepare(verifyPlacement: true)
        _ = try await backend.embed(tokenRows: [backend.tokenizer.templated("warm")])
        await lane.release()
        await metrics.recordInference()
    } catch {
        await lane.release()
        throw error
    }
}

func startKeepWarmLoop(
    intervalSeconds: Int,
    backend: BidirLMBackend,
    lane: AcceleratorLane,
    scheduler: TextScheduler,
    state: RuntimeState,
    metrics: ServerMetrics
) -> Task<Void, Never>? {
    guard intervalSeconds > 0 else { return nil }
    return Task.detached(priority: .background) {
        let nanos = UInt64(intervalSeconds) * 1_000_000_000
        let warm = backend.tokenizer.templated("warm")
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: nanos)
            guard !Task.isCancelled, await state.isReady() else { continue }
            let before = await scheduler.snapshot()
            guard before.shortRows == 0, before.longRows == 0, await lane.tryAcquire() else { continue }
            do {
                _ = try await backend.embed(tokenRows: [warm])
                await lane.release()
                await state.recordKeepWarmSuccess()
                await metrics.recordKeepWarm(success: true)
            } catch {
                await lane.release()
                await state.recordKeepWarmFailure(String(describing: error))
                await metrics.recordKeepWarm(success: false)
                timestamped("keep-warm failed: \(error)", error: true)
            }
        }
    }
}

final class ShutdownCoordinator: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false
    private let server: HTTPServer
    private let state: RuntimeState
    private let grace: Int

    init(server: HTTPServer, state: RuntimeState, grace: Int) {
        self.server = server
        self.state = state
        self.grace = grace
    }

    func request(reason: String, exitCode: Int32) {
        lock.lock()
        guard !requested else { lock.unlock(); return }
        requested = true
        lock.unlock()
        timestamped("shutdown: \(reason)", error: exitCode != 0)
        Task.detached { [server, state, grace] in
            await state.beginShutdown()
            await server.shutdown(graceSeconds: grace)
            exit(exitCode)
        }
    }
}

final class FatalRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var coordinator: ShutdownCoordinator?
    func install(_ coordinator: ShutdownCoordinator) { lock.withLock { self.coordinator = coordinator } }
    func fire(_ message: String) { lock.withLock { coordinator }?.request(reason: message, exitCode: 1) }
}

/// One model family's service behind the shared HTTP shell: its router, readiness work, and
/// keep-warm loop. Built after the family is detected from the bundle manifest.
struct ServingApp: Sendable {
    let modelID: String
    let isDummy: Bool
    let banner: String
    let state: RuntimeState
    let metrics: ServerMetrics
    let route: @Sendable (HTTPRequest) async -> HTTPResponse
    /// Loads and warms the model; returns the log line for readiness.
    let startup: @Sendable () async throws -> String
    let startKeepWarm: @Sendable () -> Task<Void, Never>?
}

final class BlockingResultBox<T>: @unchecked Sendable {
    let lock = NSLock()
    var result: Result<T, any Error>?
}

/// Run async bundle validation before the socket opens (top-level code is synchronous).
func blockingAsync<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = BlockingResultBox<T>()
    Task.detached {
        let result: Result<T, any Error>
        do { result = .success(try await operation()) } catch { result = .failure(error) }
        box.lock.withLock { box.result = result }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.lock.withLock({ box.result }) else {
        throw ArgumentError("async validation returned no result")
    }
    return try result.get()
}

func makeBidirLMApp(arguments: Arguments, bundleURL: URL) throws -> ServingApp {
    if arguments.explicit.contains("--startup-verification") {
        throw ArgumentError("--startup-verification applies to Jina bundles; BidirLM audits placement at every start")
    }
    if let dimensions = arguments.dimensions, dimensions != BidirLMContract.dimension {
        throw ArgumentError("--dimensions must be \(BidirLMContract.dimension) for BidirLM bundles (no Matryoshka truncation)")
    }
    // Integrity and contract validation happen before the tokenizer, Core ML, or the socket.
    let bundle = try BidirLMBundle.load(from: bundleURL, allowFixture: arguments.allowDummy)
    let isDummy = bundle.manifest.isFixture
    guard !isDummy || arguments.modelName == nil else {
        throw ArgumentError("--model-name is not allowed with a dummy fixture")
    }
    let tokenizer = try BidirLMTokenizer(
        folder: bundle.root.appendingPathComponent(bundle.manifest.tokenizer), manifest: bundle.manifest)
    let backend = BidirLMBackend(bundle: bundle, mode: arguments.compute, tokenizer: tokenizer,
                                 programBudget: arguments.aneProgramBudget)
    let state = RuntimeState()
    let metrics = ServerMetrics()
    let lane = AcceleratorLane()
    let media = try MediaPipeline(backend: backend, lane: lane, metrics: metrics)
    let docsContext = makeDocsContext(arguments: arguments, bundle: bundle, modalities: media.modalities)
    if let error = DocsPage.validationError(docsContext) {
        throw ArgumentError("integrated docs validation failed: \(error)")
    }

    if arguments.checkConfig {
        print("ok: \(bundle.manifest.modelID)\(isDummy ? " (Core ML dummy fixture)" : "")")
        print("family: \(ModelFamily.bidirlm.rawValue)")
        print("space: \(bundle.manifest.spaceID)")
        print("compute: \(arguments.compute.rawValue) (\(arguments.compute.coreMLName))")
        print("artifacts: \(bundle.artifactFingerprint)")
        exit(0)
    }

    let config = ServerConfig(
        bundleURL: bundleURL,
        port: arguments.port,
        compute: arguments.compute,
        modelName: arguments.modelName,
        maxBatch: arguments.maxBatch,
        maxBodyBytes: arguments.maxBodyMB * 1_048_576,
        maxTotalBodyBytes: arguments.maxTotalBodyMB * 1_048_576,
        maxQueueRequests: arguments.maxQueueRequests,
        maxQueueItems: arguments.maxQueueItems,
        maxRequestTokens: arguments.maxRequestTokens,
        batchWindowMilliseconds: arguments.batchWindowMS,
        keepWarmSeconds: arguments.keepWarmSeconds,
        maxConnections: arguments.maxConnections,
        ioTimeoutSeconds: arguments.ioTimeoutSeconds,
        shutdownGraceSeconds: arguments.shutdownGraceSeconds)

    let admission = AdmissionGate(maxRequests: config.maxQueueRequests, maxItems: config.maxQueueItems)
    let startedAt = Int(Date().timeIntervalSince1970)
    let scheduler = TextScheduler(backend: backend, lane: lane, metrics: metrics,
                                  windowMilliseconds: config.batchWindowMilliseconds)
    let service = EmbeddingsService(
        config: config, backend: backend, scheduler: scheduler, media: media, state: state,
        admission: admission, lane: lane, metrics: metrics, docsContext: docsContext, startedAt: startedAt)
    return ServingApp(
        modelID: bundle.manifest.modelID,
        isDummy: isDummy,
        banner: "gloss-server \(BuildInfo.version) model=\(bundle.manifest.modelID) family=\(ModelFamily.bidirlm.rawValue)\(isDummy ? " fixture=dummy-coreml" : "") compute=\(arguments.compute.rawValue) batch-window=\(arguments.batchWindowMS)ms",
        state: state,
        metrics: metrics,
        route: { await service.route($0) },
        startup: {
            try await performStartup(backend: backend, lane: lane, metrics: metrics)
            let placement = await backend.placement
            let share = placement?.costShare.sorted { $0.key < $1.key }
                .map { "\($0.key)=\(String(format: "%.3f", $0.value))" }.joined(separator: ",") ?? "n/a"
            let seconds = await backend.loadSeconds ?? 0
            return "ready: compute=\(arguments.compute.rawValue) load=\(String(format: "%.1f", seconds))s placement=\(share)"
        },
        startKeepWarm: {
            startKeepWarmLoop(intervalSeconds: config.keepWarmSeconds, backend: backend, lane: lane,
                              scheduler: scheduler, state: state, metrics: metrics)
        })
}

func makeJinaApp(arguments: Arguments, bundleURL: URL) throws -> ServingApp {
    for flag in ["--compute", "-c", "--ane-program-budget"] where arguments.explicit.contains(flag) {
        throw ArgumentError("\(flag) applies to BidirLM bundles; the Jina runtime places each function itself")
    }
    let bundle = try GlossModelBundle(url: bundleURL)
    // A golden fixture uses the same validated Core ML path as production; serving its constant
    // vectors needs an explicit opt-in.
    let isDummy = bundle.manifest.converter?.name == "dummy-noop"
    guard !isDummy || arguments.allowDummy else {
        throw ArgumentError("Core ML dummy fixture requires --allow-dummy")
    }
    guard !isDummy || arguments.modelName == nil else {
        throw ArgumentError("--model-name is not allowed with a dummy fixture")
    }
    let supported = bundle.capabilities.matryoshkaDimensions.sorted()
    let requested = arguments.dimensions ?? OmniSmall.Dimensions.d1024.rawValue
    guard let defaultDimensions = OmniSmall.Dimensions(rawValue: requested), supported.contains(requested) else {
        throw ArgumentError("--dimensions must be one of \(supported) for this bundle")
    }
    // Validate the pinned contract, compiled function shapes, and every declared checksum
    // before loading the tokenizer or opening a socket.
    let model = try blockingAsync { try await OmniSmall.load(from: bundleURL, dimensions: .d1024) }
    let modalities = JinaEmbeddingsService.modalities(bundle)
    let docsContext = DocsPage.Context(
        baseURL: "http://127.0.0.1:\(arguments.port)",
        modelID: arguments.modelName ?? bundle.manifest.modelID,
        dimensions: defaultDimensions.rawValue,
        maxTokens: JinaLimits.maximumTokensPerInput,
        compute: JinaLimits.computeDescription,
        modalities: modalities,
        maxBatch: arguments.maxBatch,
        maxRequestTokens: arguments.maxRequestTokens,
        maxBodyMB: arguments.maxBodyMB,
        maxTotalBodyMB: arguments.maxTotalBodyMB,
        maxQueueRequests: arguments.maxQueueRequests,
        maxQueueItems: arguments.maxQueueItems,
        batchWindowMS: arguments.batchWindowMS,
        keepWarmSeconds: arguments.keepWarmSeconds,
        maxConnections: arguments.maxConnections,
        ioTimeoutSeconds: arguments.ioTimeoutSeconds,
        shutdownGraceSeconds: arguments.shutdownGraceSeconds,
        accessLogMode: arguments.accessLog.rawValue,
        spaceID: model.space(for: defaultDimensions),
        isDummy: isDummy,
        family: .jinaOmniSmall,
        matryoshka: supported)
    if let error = DocsPage.validationError(docsContext) {
        throw ArgumentError("integrated docs validation failed: \(error)")
    }

    if arguments.checkConfig {
        print("ok: \(bundle.manifest.modelID)\(isDummy ? " (Core ML dummy fixture)" : "")")
        print("family: \(ModelFamily.jinaOmniSmall.rawValue)")
        print("space[\(defaultDimensions.rawValue)]: \(model.space(for: defaultDimensions))")
        print("dimensions: \(supported.map(String.init).joined(separator: ",")) (default \(defaultDimensions.rawValue))")
        print("modalities: \(modalities.joined(separator: ","))")
        print("startup-verification: \(arguments.startupVerification.rawValue)")
        exit(0)
    }

    let config = JinaServiceConfig(
        port: arguments.port,
        defaultDimensions: defaultDimensions,
        modelName: arguments.modelName,
        maxBatch: arguments.maxBatch,
        maxBodyBytes: arguments.maxBodyMB * 1_048_576,
        maxTotalBodyBytes: arguments.maxTotalBodyMB * 1_048_576,
        maxQueueRequests: arguments.maxQueueRequests,
        maxQueueItems: arguments.maxQueueItems,
        maxRequestTokens: arguments.maxRequestTokens,
        batchWindowMilliseconds: arguments.batchWindowMS,
        keepWarmSeconds: arguments.keepWarmSeconds,
        maxConnections: arguments.maxConnections,
        ioTimeoutSeconds: arguments.ioTimeoutSeconds,
        shutdownGraceSeconds: arguments.shutdownGraceSeconds)
    // Upload scratch directories a crashed or killed process left behind, before the listener
    // opens. Housekeeping only: it never throws and never blocks startup.
    let sweep = OmniSmall.sweepStaleTemporaryMedia()
    timestamped("temp media sweep: examined=\(sweep.examined) removed=\(sweep.removedDirectories) bytes=\(sweep.removedBytes)"
        + (sweep.failed > 0 ? " failed=\(sweep.failed)" : ""), error: sweep.failed > 0)

    let state = RuntimeState()
    let metrics = ServerMetrics()
    let lane = AcceleratorLane()
    let startupStatus = JinaStartupStatus(mode: arguments.startupVerification)
    let admission = AdmissionGate(maxRequests: config.maxQueueRequests, maxItems: config.maxQueueItems)
    let tokenizers = try JinaTokenizerPool(bundle: bundle)
    let store = JinaModelStore(model: model)
    let batcher = JinaTextBatcher(store: store, lane: lane, metrics: metrics,
                                  windowMilliseconds: config.batchWindowMilliseconds)
    let service = JinaEmbeddingsService(
        config: config, bundle: bundle, store: store, tokenizers: tokenizers, state: state,
        admission: admission, batcher: batcher, lane: lane, metrics: metrics, startup: startupStatus,
        docsContext: docsContext, startedAt: Int(Date().timeIntervalSince1970), isFixture: isDummy)

    @Sendable func warm() async throws { try await JinaStartup.warm(model: model, lane: lane, metrics: metrics) }
    return ServingApp(
        modelID: bundle.manifest.modelID,
        isDummy: isDummy,
        banner: "gloss-server \(BuildInfo.version) model=\(bundle.manifest.modelID) family=\(ModelFamily.jinaOmniSmall.rawValue)\(isDummy ? " fixture=dummy-coreml" : "") dimensions=\(defaultDimensions.rawValue) batch-window=\(arguments.batchWindowMS)ms startup-verification=\(arguments.startupVerification.rawValue) chip=\"\(HardwareIdentity.chip)\" macos=\(HardwareIdentity.macOS)",
        state: state,
        metrics: metrics,
        route: { await service.route($0) },
        startup: {
            try await JinaStartup.run(
                mode: arguments.startupVerification, model: model, lane: lane, metrics: metrics,
                status: startupStatus, modalities: modalities)
        },
        startKeepWarm: {
            guard config.keepWarmSeconds > 0 else { return nil }
            return Task.detached(priority: .background) {
                let nanos = UInt64(config.keepWarmSeconds) * 1_000_000_000
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: nanos)
                    guard !Task.isCancelled, await state.isReady() else { continue }
                    let before = await batcher.snapshot()
                    guard before.rows == 0, !before.executing, await lane.tryAcquire() else { continue }
                    await lane.release()
                    do {
                        try await warm()
                        await state.recordKeepWarmSuccess()
                        await metrics.recordKeepWarm(success: true)
                    } catch {
                        await state.recordKeepWarmFailure(String(describing: error))
                        await metrics.recordKeepWarm(success: false)
                        timestamped("keep-warm failed: \(error)", error: true)
                    }
                }
            }
        })
}

do {
    let arguments = try Arguments.parse(CommandLine.arguments)
    guard let bundleURL = arguments.bundle else { throw ArgumentError("--bundle is required") }
    guard FileManager.default.fileExists(atPath: bundleURL.path) else {
        throw ArgumentError("bundle not found: \(bundleURL.path)")
    }
    let app: ServingApp
    switch try ModelFamily.detect(bundle: bundleURL) {
    case .bidirlm: app = try makeBidirLMApp(arguments: arguments, bundleURL: bundleURL)
    case .jinaOmniSmall: app = try makeJinaApp(arguments: arguments, bundleURL: bundleURL)
    }
    let state = app.state
    let metrics = app.metrics
    let isDummy = app.isDummy

    timestamped(app.banner)
    let relay = FatalRelay()
    let server = try HTTPServer(
        port: arguments.port,
        maxBodyBytes: arguments.maxBodyMB * 1_048_576,
        maxTotalBodyBytes: arguments.maxTotalBodyMB * 1_048_576,
        maxConnections: arguments.maxConnections,
        ioTimeoutSeconds: arguments.ioTimeoutSeconds,
        handler: { request in
            let started = ContinuousClock.now
            var response = await app.route(request)
            if isDummy { response.extraHeaders.append(("X-Glossematics-Dummy", "true")) }
            await metrics.recordHTTP()
            let ms = elapsedMilliseconds(ContinuousClock.now - started)
            if arguments.accessLog == .all || (arguments.accessLog == .errors && response.status >= 400) {
                timestamped(String(
                    format: "http id=%@ %@ %@ -> %d %.1fms",
                    request.requestID, request.method, request.path, response.status, ms),
                    error: response.status >= 500)
            }
            return response
        },
        onFatal: { relay.fire($0) })
    let shutdown = ShutdownCoordinator(server: server, state: state, grace: arguments.shutdownGraceSeconds)
    relay.install(shutdown)

    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)
    let signalSources: [DispatchSourceSignal] = [SIGTERM, SIGINT].map { signalNumber in
        let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .main)
        source.setEventHandler { shutdown.request(reason: "signal \(signalNumber)", exitCode: 0) }
        source.resume()
        return source
    }

    try server.start()
    timestamped("listening http://127.0.0.1:\(arguments.port) docs=/docs ready=/ready")

    Task.detached(priority: .userInitiated) {
        do {
            let message = try await app.startup()
            await state.markReady()
            timestamped(message)
        } catch {
            await state.failStartup(String(describing: error))
            timestamped("startup FAILED: \(error)", error: true)
        }
    }
    let keepWarm = app.startKeepWarm()
    _ = keepWarm

    // A release build may otherwise destroy the local dispatch sources before the run loop
    // receives a signal, leaving SIGTERM/SIGINT ignored and the daemon unshuttable.
    withExtendedLifetime(signalSources) { RunLoop.main.run() }
} catch let error as ArgumentError {
    FileHandle.standardError.write(Data("error: \(error.description)\n\n\(Arguments.helpText)\n".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
