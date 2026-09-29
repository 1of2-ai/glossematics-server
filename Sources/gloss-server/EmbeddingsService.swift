import Foundation

struct ServerConfig: Sendable {
    var bundleURL: URL
    var port: UInt16
    var compute: ComputeMode
    var modelName: String?
    var maxBatch: Int
    var maxBodyBytes: Int
    var maxTotalBodyBytes: Int
    var maxQueueRequests: Int
    var maxQueueItems: Int
    var maxRequestTokens: Int
    var batchWindowMilliseconds: Double
    var keepWarmSeconds: Int
    var maxConnections: Int
    var ioTimeoutSeconds: Int
    var shutdownGraceSeconds: Int
}

/// OpenAI-compatible boundary limits. Text inputs may use the model's full 32K context.
enum EmbeddingsLimits {
    static let maximumItems = 2_048
    static let maximumTokensPerInput = BidirLMContract.maxTokens
}

struct EmbeddingsService: Sendable {
    let config: ServerConfig
    let backend: BidirLMBackend
    let scheduler: TextScheduler
    let media: MediaPipeline
    let state: RuntimeState
    let admission: AdmissionGate
    let lane: AcceleratorLane
    let metrics: ServerMetrics
    let docsContext: DocsPage.Context
    let startedAt: Int

    var servedModelName: String { config.modelName ?? backend.bundle.manifest.modelID }
    var tokenizer: BidirLMTokenizer { backend.tokenizer }

    func route(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/docs"), ("GET", "/docs/"): return DocsPage.response(docsContext)
        case ("GET", "/"): return .init(status: 302, contentType: "text/plain; charset=utf-8", extraHeaders: [("Location", "/docs")], body: Data())
        case ("GET", "/live"): return live()
        case ("GET", "/health"), ("GET", "/ready"): return await health()
        case ("GET", "/metrics"): return await prometheusMetrics()
        case ("GET", "/v1/models"): return ModelListResponse(data: [modelObject()]).asResponse()
        case ("GET", _) where request.path.hasPrefix("/v1/models/"):
            return model(pathComponent: percentDecode(String(request.path.dropFirst("/v1/models/".count))))
        case ("POST", "/v1/embeddings"): return await embeddings(request)
        case (_, "/docs"), (_, "/docs/"), (_, "/"), (_, "/live"), (_, "/health"), (_, "/ready"), (_, "/metrics"), (_, "/v1/models"):
            return .error(405, .invalidRequest("method not allowed; use GET"), extraHeaders: [("Allow", "GET")])
        case (_, _) where request.path.hasPrefix("/v1/models/"):
            return .error(405, .invalidRequest("method not allowed; use GET"), extraHeaders: [("Allow", "GET")])
        case (_, "/v1/embeddings"):
            return .error(405, .invalidRequest("method not allowed; use POST"), extraHeaders: [("Allow", "POST")])
        default: return .error(404, .invalidRequest("unknown route: \(request.method) \(request.path)", code: "not_found"))
        }
    }

    private func live() -> HTTPResponse {
        .json(200, LiveResponse(status: "ok", version: BuildInfo.version, model: servedModelName, port: Int(config.port)))
    }

    private func health() async -> HTTPResponse {
        let runtime = await state.snapshot()
        let queue = await admission.snapshot()
        let batching = await scheduler.snapshot()
        let status = backend.status.snapshot
        let placement = status.placement
        let payload = HealthResponse(
            status: runtime.phase,
            ready: runtime.ready,
            version: BuildInfo.version,
            model: servedModelName,
            dimensions: BidirLMContract.dimension,
            space: backend.spaceID,
            compute: config.compute.rawValue,
            core_ml_compute_units: config.compute.coreMLName,
            placement_verified: placement != nil,
            placement_strict_ane: placement.map(\.strictANE),
            placement_cost_share: placement?.costShare,
            load_seconds: status.loadSeconds,
            max_tokens: BidirLMContract.maxTokens,
            modalities: media.modalities,
            port: Int(config.port),
            uptime_seconds: max(0, Int(Date().timeIntervalSince(runtime.startedAt))),
            ready_at: runtime.readyAt.map { Int($0.timeIntervalSince1970) },
            queue_requests: queue.requests,
            queue_items: queue.items,
            scheduled_requests: batching.requests,
            pending_short_rows: batching.shortRows,
            pending_long_documents: batching.longRows,
            active_long_tokens: batching.activeLong?.tokens,
            active_long_progress: batching.activeLong?.progress,
            active_media_kind: batching.activeMedia?.kind,
            active_media_progress: batching.activeMedia?.progress,
            programs_resident: status.programs,
            program_budget: backend.programBudget,
            program_sets_resident: status.sets,
            program_set_loads: status.loads,
            program_set_evictions: status.evictions,
            keep_warm_seconds: config.keepWarmSeconds,
            last_keep_warm_at: runtime.lastKeepWarmAt.map { Int($0.timeIntervalSince1970) },
            last_keep_warm_error: runtime.lastKeepWarmError,
            keep_warm_failures: runtime.keepWarmFailures,
            fixture: backend.isFixture,
            error: runtime.startupError)
        return .json(runtime.ready ? 200 : 503, payload)
    }

    private func prometheusMetrics() async -> HTTPResponse {
        let runtime = await state.snapshot()
        let counters = await metrics.snapshot()
        let queue = await admission.snapshot()
        let batching = await scheduler.snapshot()
        var lines: [String] = []
        func gauge(_ help: String, _ name: String, _ value: String) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) gauge", "\(name) \(value)"]
        }
        func counter(_ help: String, _ name: String, _ value: UInt64) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) counter", "\(name) \(value)"]
        }
        func labeled(_ help: String, _ name: String, _ label: String, _ values: [String: UInt64], _ keys: [String]) {
            lines += ["# HELP \(name) \(help)", "# TYPE \(name) counter"]
            for key in keys { lines.append("\(name){\(label)=\"\(key)\"} \(values[key, default: 0])") }
        }
        gauge("Whether the service is ready.", "gloss_ready", runtime.ready ? "1" : "0")
        lines += ["# HELP gloss_compute_mode Operator-selected Core ML compute mode.", "# TYPE gloss_compute_mode gauge"]
        for mode in ComputeMode.allCases {
            lines.append("gloss_compute_mode{mode=\"\(mode.rawValue)\"} \(mode == config.compute ? 1 : 0)")
        }
        gauge("Process uptime seconds.", "gloss_uptime_seconds", String(format: "%.3f", Date().timeIntervalSince(runtime.startedAt)))
        gauge("Admitted logical requests.", "gloss_queue_requests", String(queue.requests))
        gauge("Admitted input items.", "gloss_queue_items", String(queue.items))
        gauge("Short rows waiting to be packed.", "gloss_pending_short_rows", String(batching.shortRows))
        gauge("Long documents queued or running.", "gloss_pending_long_documents", String(batching.longRows))
        gauge("Progress of the running long document (0-1).", "gloss_active_long_progress",
              String(format: "%.4f", batching.activeLong?.progress ?? 0))
        let residency = backend.status.snapshot
        gauge("Core ML programs this process keeps loaded.", "gloss_programs_resident", String(residency.programs))
        gauge("Neural Engine program budget.", "gloss_program_budget", String(backend.programBudget))
        counter("On-demand program set loads.", "gloss_program_set_loads_total", UInt64(residency.loads))
        counter("Program sets evicted to stay within the budget.", "gloss_program_set_evictions_total", UInt64(residency.evictions))
        counter("On-demand loads that fell back from the Neural Engine (then evicted for reload).",
                "gloss_neural_engine_fallbacks_total", UInt64(residency.fallbacks))
        gauge("Progress of the running media tower item (0-1).", "gloss_active_media_progress",
              String(format: "%.4f", batching.activeMedia?.progress ?? 0))
        counter("HTTP requests handled.", "gloss_http_requests_total", counters.httpRequests)
        counter("Embedding requests accepted.", "gloss_embedding_requests_total", counters.embeddingRequests)
        counter("Embedding input items accepted.", "gloss_embedding_items_total", counters.embeddingItems)
        counter("Embedding requests rejected by admission.", "gloss_rejected_requests_total", counters.rejectedRequests)
        counter("Embedding requests failed after admission.", "gloss_failed_requests_total", counters.failedRequests)
        let kinds = ["fused64", "pack512"]
        labeled("Packed text executions by kind.", "gloss_text_waves_total", "kind", counters.wavesByKind, kinds)
        labeled("Text rows executed in packed waves.", "gloss_text_rows_total", "kind", counters.rowsByKind, kinds)
        labeled("Templated text tokens executed in packed waves.", "gloss_text_tokens_total", "kind", counters.tokensByKind, kinds)
        counter("Packed waves that combined several requests.", "gloss_coalesced_waves_total", counters.coalescedWaves)
        counter("Long documents completed.", "gloss_long_documents_total", counters.longDocuments)
        counter("Tokens in completed long documents.", "gloss_long_tokens_total", counters.longTokens)
        counter("Accelerator turns used by long documents.", "gloss_long_steps_total", counters.longSteps)
        labeled("Media items embedded.", "gloss_media_items_total", "kind", counters.mediaItems, ["image", "audio", "message"])
        gauge("Configured admission request limit.", "gloss_admission_max_requests", String(queue.maxRequests))
        gauge("Configured admission item limit.", "gloss_admission_max_items", String(queue.maxItems))
        gauge("Dynamic batching collection window milliseconds.", "gloss_batch_window_milliseconds", String(format: "%.3f", config.batchWindowMilliseconds))
        counter("Keep-warm attempts.", "gloss_keep_warm_attempts_total", counters.keepWarmAttempts)
        counter("Keep-warm failures.", "gloss_keep_warm_failures_total", counters.keepWarmFailures)
        if let last = counters.lastInferenceUnix {
            gauge("Unix timestamp of last inference.", "gloss_last_inference_unixtime", String(format: "%.3f", last))
        }
        lines.append("")
        return .init(status: 200, contentType: "text/plain; version=0.0.4; charset=utf-8",
                     extraHeaders: [("Cache-Control", "no-store")], body: Data(lines.joined(separator: "\n").utf8))
    }

    private func model(pathComponent: String) -> HTTPResponse {
        pathComponent == servedModelName ? modelObject().asResponse() : .error(404, .modelNotFound(pathComponent))
    }

    private func modelObject() -> ModelObject { .init(id: servedModelName, created: startedAt, owned_by: "glossematics") }

    // MARK: - embeddings

    private func embeddings(_ request: HTTPRequest) async -> HTTPResponse {
        guard request.contentTypeIsJSON else {
            return .error(415, .invalidRequest("Content-Type must be application/json", code: "unsupported_media_type"))
        }
        let body: EmbeddingsRequestBody
        do {
            body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: request.body)
        } catch {
            return .error(400, .invalidRequest(decodingMessage(error), param: "input"))
        }
        guard let requested = body.model else {
            return .error(400, .invalidRequest("you must provide a model parameter", param: "model"))
        }
        guard requested == servedModelName else { return .error(404, .modelNotFound(requested)) }
        guard !body.items.isEmpty else {
            return .error(400, .invalidRequest("input must not be an empty array", param: "input"))
        }
        let maximumItems = min(config.maxBatch, EmbeddingsLimits.maximumItems)
        guard body.items.count <= maximumItems else {
            return .error(400, .invalidRequest("input exceeds the maximum of \(maximumItems) items per request", param: "input"))
        }
        let base64: Bool
        switch body.encodingFormat?.lowercased() {
        case nil, "float": base64 = false
        case "base64": base64 = true
        default:
            return .error(400, .invalidRequest("encoding_format must be \"float\" or \"base64\"", param: "encoding_format"))
        }
        if let dimensions = body.dimensions, dimensions != BidirLMContract.dimension {
            return .error(400, .invalidRequest(
                "dimensions must be \(BidirLMContract.dimension) for this model (it has no Matryoshka truncation)",
                param: "dimensions"))
        }
        guard await state.isReady() else {
            return .error(503, .serviceUnavailable("model is not ready"), extraHeaders: [("Retry-After", "5")])
        }
        let itemCount = body.items.count
        guard await admission.tryAcquire(items: itemCount) else {
            await metrics.recordRejected()
            return .error(503, .serviceUnavailable("embedding queue is full; retry shortly"), extraHeaders: [("Retry-After", "1")])
        }
        var response = await embeddingsAdmitted(body: body, base64: base64)
        response.onComplete = { await admission.release(items: itemCount) }
        return response
    }

    private func embeddingsAdmitted(body: EmbeddingsRequestBody, base64: Bool) async -> HTTPResponse {
        await metrics.recordEmbeddingRequest(items: body.items.count)
        var prepared = [ScheduledInput]()
        prepared.reserveCapacity(body.items.count)
        var promptTokens = 0
        do {
            for (index, item) in body.items.enumerated() {
                try Task.checkCancellation()
                let next: ScheduledInput
                switch item {
                case let .text(text):
                    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        throw APIError.invalidRequest("input \(index) must not be an empty string", param: "input")
                    }
                    next = .sequence(LMSequence(ids: tokenizer.templated(text)))
                case let .tokenIDs(ids):
                    guard !ids.isEmpty else {
                        throw APIError.invalidRequest("input \(index): token-ID inputs must not be empty", param: "input")
                    }
                    guard ids.allSatisfy({ Int($0) < BidirLMContract.vocabulary }) else {
                        throw APIError.invalidRequest(
                            "input \(index): token IDs must be below the vocabulary size \(BidirLMContract.vocabulary)",
                            param: "input")
                    }
                    next = .sequence(LMSequence(ids: tokenizer.templated(tokens: ids)))
                case .video:
                    throw APIError.invalidRequest(
                        "input \(index): video input is not supported by this server; send frames as images",
                        param: "input", code: "unsupported_modality")
                case .image, .audio, .message:
                    next = try await media.prepare(item, index: index)
                }
                guard next.tokens <= EmbeddingsLimits.maximumTokensPerInput else {
                    throw APIError.invalidRequest(
                        "input \(index) has \(next.tokens) tokens including the chat template and media placeholders; maximum is \(EmbeddingsLimits.maximumTokensPerInput)",
                        param: "input")
                }
                promptTokens += next.tokens
                guard promptTokens <= config.maxRequestTokens else {
                    throw APIError.invalidRequest(
                        "total input tokens exceed the per-request maximum of \(config.maxRequestTokens)", param: "input")
                }
                prepared.append(next)
            }
        } catch let error as APIError {
            await metrics.recordFailure()
            return .error(400, error)
        } catch let error as MediaPipeline.Failure {
            await metrics.recordFailure()
            return error.isClientError
                ? .error(400, .invalidRequest(error.description, param: "input", code: error.code))
                : .error(500, .apiError(error.description))
        } catch is CancellationError {
            await metrics.recordFailure()
            return .error(503, .serviceUnavailable("request was cancelled"))
        } catch {
            await metrics.recordFailure()
            return .error(400, .invalidRequest(String(describing: error), param: "input"))
        }

        do {
            let vectors = try await scheduler.submit(inputs: prepared)
            let data = try vectors.enumerated().map { offset, vector -> EmbeddingObject in
                guard vector.count == BidirLMContract.dimension else {
                    throw MediaPipeline.Failure.internalError("input \(offset) produced no embedding")
                }
                return EmbeddingObject(index: offset, embedding: base64 ? .base64(Self.base64LittleEndian(vector)) : .floats(vector))
            }
            return .json(
                200,
                EmbeddingsResponse(data: data, model: servedModelName,
                                   usage: .init(prompt_tokens: promptTokens, total_tokens: promptTokens)),
                extraHeaders: [
                    ("X-Glossematics-Space", backend.spaceID),
                    ("X-Glossematics-Dimensions", String(BidirLMContract.dimension)),
                    ("X-Glossematics-Compute", config.compute.rawValue),
                ])
        } catch is CancellationError {
            await metrics.recordFailure()
            return .error(503, .serviceUnavailable("request was cancelled"))
        } catch let error as MediaPipeline.Failure {
            await metrics.recordFailure()
            return error.isClientError
                ? .error(400, .invalidRequest(error.description, param: "input", code: error.code))
                : .error(500, .apiError(error.description))
        } catch let error as BidirLMMediaEncoder.Failure {
            await metrics.recordFailure()
            if case .invalidInput = error {
                return .error(400, .invalidRequest(error.description, param: "input", code: "invalid_media"))
            }
            return .error(500, .apiError(error.description))
        } catch {
            await metrics.recordFailure()
            return .error(500, .apiError(String(describing: error)))
        }
    }

    private func percentDecode(_ s: String) -> String { s.removingPercentEncoding ?? s }

    private func decodingMessage(_ error: any Error) -> String {
        if let e = error as? DecodingError {
            switch e {
            case let .valueNotFound(_, c): return c.debugDescription
            case let .dataCorrupted(c): return c.debugDescription
            case let .keyNotFound(k, _): return "missing required parameter: '\(k.stringValue)'"
            case let .typeMismatch(_, c): return c.debugDescription
            @unknown default: return "malformed JSON body"
            }
        }
        return "malformed JSON body: \(String(describing: error))"
    }

    static func base64LittleEndian(_ values: [Float]) -> String {
        guard !values.isEmpty else { return "" }
        return values.withUnsafeBytes { Data($0).base64EncodedString() }
    }
}

extension HTTPRequest {
    var contentTypeIsJSON: Bool {
        guard let raw = headers["content-type"] else { return false }
        return raw.split(separator: ";", maxSplits: 1).first?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "application/json"
    }
}

extension ModelObject { func asResponse() -> HTTPResponse { .json(200, self) } }
extension ModelListResponse { func asResponse() -> HTTPResponse { .json(200, self) } }
