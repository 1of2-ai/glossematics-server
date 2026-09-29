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

    static func parse(_ args: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < args.count {
            let flag = args[index]
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
    gloss-server — local OpenAI-compatible BidirLM embeddings microservice
    usage: gloss-server --bundle <dir> [options]
      --compute ane|gpu|cpu (ane)  operator-selected Core ML placement; never falls back
      --port N (11435)  --model-name ID
      --max-batch N (2048)  --max-queue-requests N (128)  --max-queue-items N (8192)
      --max-request-tokens N (131072)  --batch-window-ms N (2.0)  --keep-warm-seconds N (60)
      --ane-program-budget N (64; Neural Engine programs this process keeps loaded, 40...120)
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

do {
    let arguments = try Arguments.parse(CommandLine.arguments)
    guard let bundleURL = arguments.bundle else { throw ArgumentError("--bundle is required") }
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

    timestamped("gloss-server \(BuildInfo.version) model=\(bundle.manifest.modelID)\(isDummy ? " fixture=dummy-coreml" : "") compute=\(arguments.compute.rawValue) batch-window=\(arguments.batchWindowMS)ms")
    let relay = FatalRelay()
    let server = try HTTPServer(
        port: config.port,
        maxBodyBytes: config.maxBodyBytes,
        maxTotalBodyBytes: config.maxTotalBodyBytes,
        maxConnections: config.maxConnections,
        ioTimeoutSeconds: config.ioTimeoutSeconds,
        handler: { request in
            let started = ContinuousClock.now
            var response = await service.route(request)
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
    let shutdown = ShutdownCoordinator(server: server, state: state, grace: config.shutdownGraceSeconds)
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
    timestamped("listening http://127.0.0.1:\(config.port) docs=/docs ready=/ready")

    Task.detached(priority: .userInitiated) {
        do {
            try await performStartup(backend: backend, lane: lane, metrics: metrics)
            await state.markReady()
            let placement = await backend.placement
            let share = placement?.costShare.sorted { $0.key < $1.key }
                .map { "\($0.key)=\(String(format: "%.3f", $0.value))" }.joined(separator: ",") ?? "n/a"
            let seconds = await backend.loadSeconds ?? 0
            timestamped("ready: compute=\(arguments.compute.rawValue) load=\(String(format: "%.1f", seconds))s placement=\(share)")
        } catch {
            await state.failStartup(String(describing: error))
            timestamped("startup FAILED: \(error)", error: true)
        }
    }
    let keepWarm = startKeepWarmLoop(
        intervalSeconds: config.keepWarmSeconds, backend: backend, lane: lane,
        scheduler: scheduler, state: state, metrics: metrics)
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
