import Foundation
import Glossematics

// gloss-server — OpenAI-compatible /v1/embeddings daemon over a local model bundle.
//
// usage: gloss-server --bundle <dir> [--port N] [--dimensions N] [--model-name S]
//                     [--max-batch N] [--max-body-mb N]
//
// Binds 127.0.0.1 only (localhost daemon by design). Routes:
//   POST /v1/embeddings   OpenAI embeddings contract (+ task/role + image_path/audio_path
//                         local extensions)
//   GET  /v1/models       served model list
//   GET  /v1/models/{id}  served model object
//   GET  /health          readiness + loaded Matryoshka dimensions

struct Arguments {
    var bundle: URL?
    var port: UInt16 = 11_435
    var dimensions = 1024
    var modelName: String?
    var maxBatch = 2048
    var maxBodyMB = 64

    static func parse(_ arguments: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < arguments.count {
            let flag = arguments[index]
            func value(_ name: String) throws -> String {
                index += 1
                guard index < arguments.count else {
                    throw ArgumentError("missing value for \(name)")
                }
                return arguments[index]
            }
            switch flag {
            case "--bundle", "-b":
                parsed.bundle = URL(fileURLWithPath: try value(flag), isDirectory: true)
            case "--port", "-p":
                guard let port = UInt16(try value(flag)) else {
                    throw ArgumentError("--port expects a port number in 1...65535")
                }
                parsed.port = port
            case "--dimensions", "-d":
                guard let raw = Int(try value(flag)),
                      OmniSmall.Dimensions(rawValue: raw) != nil else {
                    throw ArgumentError("--dimensions must be one of 32, 64, 128, 256, 512, 1024")
                }
                parsed.dimensions = raw
            case "--model-name", "-m":
                parsed.modelName = try value(flag)
            case "--max-batch":
                guard let raw = Int(try value(flag)), raw > 0, raw <= 2048 else {
                    throw ArgumentError("--max-batch must be in 1...2048")
                }
                parsed.maxBatch = raw
            case "--max-body-mb":
                guard let raw = Int(try value(flag)), raw > 0, raw <= 1024 else {
                    throw ArgumentError("--max-body-mb must be in 1...1024")
                }
                parsed.maxBodyMB = raw
            case "--help", "-h":
                print(Self.helpText)
                exit(0)
            default:
                throw ArgumentError("unknown argument: \(flag)")
            }
            index += 1
        }
        return parsed
    }

    static let helpText = """
        gloss-server — OpenAI-compatible /v1/embeddings server for a local Glossematics bundle

        usage:
          gloss-server --bundle <bundle-dir> [options]

        options:
          --bundle, -b <dir>       native-v1 distribution bundle directory (required)
          --port, -p <port>        TCP port on 127.0.0.1 (default 11435)
          --dimensions, -d <n>     default embedding dimensions: 32|64|128|256|512|1024
                                   (default 1024; requests may still ask for any Matryoshka size)
          --model-name, -m <id>    model id surfaced over the API (default: bundle modelID)
          --max-batch <n>          maximum input items per request (default 2048)
          --max-body-mb <n>        maximum request body size in MiB (default 64)
        """
}

struct ArgumentError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func timestamped(_ message: String) {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    // Direct handle write: print() buffers on non-tty stdout, which hides daemon logs under
    // launchd redirection until the buffer fills.
    FileHandle.standardOutput.write(Data("\(formatter.string(from: Date())) \(message)\n".utf8))
}

do {
    let config = try Arguments.parse(CommandLine.arguments)
    guard let bundleURL = config.bundle else {
        FileHandle.standardError.write(Data("error: --bundle is required\n\n\(Arguments.helpText)\n".utf8))
        exit(2)
    }
    guard FileManager.default.fileExists(atPath: bundleURL.path) else {
        FileHandle.standardError.write(Data("error: bundle not found: \(bundleURL.path)\n".utf8))
        exit(2)
    }

    let bundle: GlossModelBundle
    do {
        bundle = try GlossModelBundle(url: bundleURL)
    } catch {
        FileHandle.standardError.write(Data("error: invalid bundle at \(bundleURL.path): \(error)\n".utf8))
        exit(1)
    }

    guard let defaultDimensions = OmniSmall.Dimensions(rawValue: config.dimensions) else {
        FileHandle.standardError.write(Data("error: unsupported default dimensions\n".utf8))
        exit(2)
    }

    let service = EmbeddingsService(
        config: ServerConfig(
            bundleURL: bundleURL,
            port: config.port,
            defaultDimensions: defaultDimensions,
            modelName: config.modelName,
            maxBatch: config.maxBatch,
            maxBodyBytes: config.maxBodyMB * 1_048_576),
        bundle: bundle,
        store: ModelStore(bundleURL: bundleURL, defaultDimensions: defaultDimensions),
        counter: TokenCounter(bundle: bundle),
        startedAt: Int(Date().timeIntervalSince1970))

    timestamped("bundle: \(bundle.manifest.modelID) space=\(bundle.manifest.spaceID ?? "?")")
    timestamped(
        "capabilities: dim=\(bundle.capabilities.embeddingDimension) "
        + "matryoshka=\(bundle.capabilities.matryoshkaDimensions) "
        + "text=\(bundle.capabilities.supportsText) image=\(bundle.capabilities.supportsImage) "
        + "audio=\(bundle.capabilities.supportsAudio)")

    // Warm the default dimension in the background so the listener is up immediately.
    let warmDimensions = defaultDimensions
    let warmStore = service.store
    Task.detached(priority: .userInitiated) {
        do {
            _ = try await warmStore.model(for: warmDimensions)
            timestamped("warm: default dimension \(warmDimensions.rawValue) loaded")
        } catch {
            timestamped("warm: default dimension \(warmDimensions.rawValue) FAILED: \(error)")
        }
    }

    if !DocsPage.templateAvailable {
        timestamped("warning: docs.html is missing from the resource bundle; /docs will return 500")
    }

    let server = try HTTPServer(port: config.port, maxBodyBytes: service.config.maxBodyBytes) { request in
        let started = ContinuousClock.now
        let response = await service.route(request)
        let elapsed = ContinuousClock.now - started
        let milliseconds = Double(elapsed.components.attoseconds) / 1e18 * 1_000
        timestamped(String(
            format: "%@ %@ -> %d (%.1f ms, %d bytes)", request.method, request.path,
            response.status, milliseconds, response.body.count))
        return response
    }
    server.start()
    timestamped("listening: http://127.0.0.1:\(config.port)/v1/embeddings (docs: /docs)")

    // Park until launchd (or an interactive Ctrl-C) asks us to stop. The main run loop
    // services the signal sources; dispatchMain() is not usable from async top-level code.
    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)
    let shutdownSources: [DispatchSourceSignal] = [SIGTERM, SIGINT].map { signalNumber in
        let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .main)
        source.setEventHandler {
            timestamped("signal \(signalNumber): shutting down")
            exit(0)
        }
        source.resume()
        return source
    }
    _ = shutdownSources // retained for the process lifetime
    RunLoop.main.run()
} catch let error as ArgumentError {
    FileHandle.standardError.write(Data("error: \(error.description)\n\n\(Arguments.helpText)\n".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
