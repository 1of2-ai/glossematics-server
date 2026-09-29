import AVFoundation
import Foundation

// MARK: - Bounded CPU executor

/// Cooperative cancellation for CPU work that runs on a dispatch thread instead of a Swift task.
/// A dispatch thread has no task context, so `Task.isCancelled` cannot be read there; the waiting
/// task's cancellation handler flips this flag and the work polls it between coarse steps (per
/// video frame, per decode stage).
final class PreparationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    /// Throws `CancellationError` once cancellation has been requested.
    func check() throws {
        if isCancelled { throw CancellationError() }
    }
}

/// Runs CPU-heavy synchronous work on a bounded pool of dispatch threads and bridges the result
/// back to Swift concurrency.
///
/// Why not just run it in a task: image decode, WAV decode and resample, mel, and above all MP4
/// decode with per-pixel Y'CbCr -> RGB and antialiased bicubic at source resolution take
/// milliseconds to many seconds of pure CPU. Run on the cooperative pool it would block threads the
/// HTTP server, admission control, and every awaiting request need; a few concurrent videos could
/// stall the whole daemon. This executor:
///
/// * runs the work on dedicated threads of a private `OperationQueue`, never the cooperative pool;
/// * bounds concurrency to `maxConcurrentJobs` (default: half the cores), so a burst of uploads
///   cannot saturate every core and starve the Core ML runtime and the rest of the system —
///   excess jobs wait in FIFO order;
/// * honours task cancellation: a job still waiting in the queue resumes its caller with
///   `CancellationError` immediately and never runs; a job already running is signalled through
///   ``PreparationCancellation`` and stops at its next poll.
final class MediaPreparationExecutor: @unchecked Sendable {
    /// The process-wide executor the production backend uses.
    static let shared = MediaPreparationExecutor(maxConcurrentJobs: defaultConcurrency)

    /// About half the cores, and never fewer than two so one long video cannot serialize behind
    /// itself on small machines.
    static var defaultConcurrency: Int { max(2, ProcessInfo.processInfo.activeProcessorCount / 2) }

    let maxConcurrentJobs: Int
    private let queue: OperationQueue

    init(maxConcurrentJobs: Int, name: String = "gloss-media-preparation") {
        self.maxConcurrentJobs = max(1, maxConcurrentJobs)
        queue = OperationQueue()
        queue.name = name
        queue.maxConcurrentOperationCount = self.maxConcurrentJobs
        queue.qualityOfService = .userInitiated
    }

    /// One submitted job. It owns its continuation exactly once: either the queue thread resumes it
    /// with the work's outcome, or a cancellation that arrives while the job is still queued does.
    private final class Job<T: Sendable>: @unchecked Sendable {
        private enum Phase { case queued, running, finished }
        private let lock = NSLock()
        private var phase = Phase.queued
        private var continuation: CheckedContinuation<T, any Error>?
        let cancellation = PreparationCancellation()

        /// Attach the continuation. A cancellation that raced ahead of this call resumes it at once.
        func install(_ continuation: CheckedContinuation<T, any Error>) {
            lock.lock()
            if cancellation.isCancelled {
                phase = .finished
                lock.unlock()
                continuation.resume(throwing: CancellationError())
                return
            }
            self.continuation = continuation
            lock.unlock()
        }

        /// Called on a queue thread when the job's turn comes.
        func runIfQueued(_ work: @Sendable (PreparationCancellation) throws -> T) {
            lock.lock()
            guard phase == .queued, let continuation else {
                lock.unlock()
                return
            }
            phase = .running
            self.continuation = nil
            lock.unlock()
            do {
                continuation.resume(returning: try work(cancellation))
            } catch {
                continuation.resume(throwing: error)
            }
        }

        /// Called from the awaiting task's cancellation handler.
        func requestCancellation() {
            cancellation.cancel()
            lock.lock()
            guard phase == .queued, let continuation else {
                lock.unlock()
                return
            }
            phase = .finished
            self.continuation = nil
            lock.unlock()
            continuation.resume(throwing: CancellationError())
        }
    }

    /// Run `work` on the executor and return its result. Throws whatever `work` throws, or
    /// `CancellationError` when the calling task is cancelled before the work starts or the work
    /// observes ``PreparationCancellation``.
    func run<T: Sendable>(
        _ work: @escaping @Sendable (PreparationCancellation) throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        let job = Job<T>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, any Error>) in
                job.install(continuation)
                queue.addOperation { job.runIfQueued(work) }
            }
        } onCancel: {
            job.requestCancellation()
        }
    }
}

// MARK: - Prepared media (the value that crosses from the CPU phase to the accelerator phase)

/// A media input whose CPU work — decode, resize, patchify, mel, position tables, prompt ids — is
/// done. It carries exactly the arrays the Core ML functions consume, so running it costs only
/// predictions. Produced by ``OmniSmall/prepareMedia(_:role:)`` off the accelerator lane and consumed
/// by ``OmniSmall/embedPreparedMedia(_:dimensions:)`` on it.
///
/// The retrieval role is baked into the prompt ids at preparation time, which is why the inference
/// entry point takes no role. The value is immutable and `Sendable`; it holds up to a few tens of
/// megabytes for a large image or long video, so callers should consume it promptly rather than
/// queue many.
@_spi(Server)
public struct OmniSmallPreparedMedia: Sendable {
    public enum Kind: String, Sendable {
        case image, audio, video
    }

    public let kind: Kind
    public let role: OmniSmallRole
    /// CPU time spent preparing, in milliseconds (decode through prompt ids; excludes any time the
    /// job waited for a free executor thread).
    public let prepareMilliseconds: Double
    /// Time the job waited for a free executor thread, in milliseconds. Sustained nonzero values
    /// mean media uploads are arriving faster than the bounded executor can prepare them.
    public let queuedMilliseconds: Double

    let payload: Payload

    enum Payload: Sendable {
        case image(PreparedVisionInputs)
        case video(PreparedVisionInputs)
        case audio(PreparedAudioInputs)
        /// For backends with no separate CPU phase (test doubles): the original input, executed as
        /// one step by `embedMedia`.
        case passthrough(OmniSmall.Input)
    }

    init(kind: Kind, role: OmniSmallRole, prepareMilliseconds: Double, queuedMilliseconds: Double,
         payload: Payload) {
        self.kind = kind
        self.role = role
        self.prepareMilliseconds = prepareMilliseconds
        self.queuedMilliseconds = queuedMilliseconds
        self.payload = payload
    }
}

/// The retrieval side a vector is for. It selects the "Query: " / "Document: " conditioning that
/// is baked into every modality's prompt ids; a query and a document embedding of the same input
/// differ, and only a query scores against a document.
@_spi(Server)
public enum OmniSmallRole: Sendable {
    case query
    case document

    var prompt: GlossTextEmbedder.Prompt {
        switch self {
        case .query: .query
        case .document: .document
        }
    }
}

// MARK: - Host-side components (no Core ML)

/// The CPU-side building blocks for media preparation — vision position tables, the mel frontend,
/// and the three preparers — built lazily and exactly once from the validated manifest.
///
/// These are pure values with no Core ML state, so the backend actor can hand them to the bounded
/// executor without the non-`Sendable` embedders ever crossing an isolation boundary. The backend
/// builds its embedders from the SAME position table and mel frontend, so the preparers and the
/// embedders cannot diverge.
final class OmniSmallMediaHost: Sendable {
    private let bundle: GlossModelBundle
    /// Where uploaded audio and video bytes are staged for AVFoundation. The system temporary
    /// directory in production; tests point it at a private directory to observe cleanup.
    let temporaryDirectory: URL
    private let positionsCache = KeyedLoadCache<Int, VisionPositions>()
    private let melCache = KeyedLoadCache<Int, GlossMelFrontend>()
    private let imageCache = KeyedLoadCache<Int, ImageInputPreparer>()
    private let videoCache = KeyedLoadCache<Int, VideoInputPreparer>()
    private let audioCache = KeyedLoadCache<Int, AudioInputPreparer>()

    init(bundle: GlossModelBundle, temporaryDirectory: URL = FileManager.default.temporaryDirectory) {
        self.bundle = bundle
        self.temporaryDirectory = temporaryDirectory
    }

    /// Vision position tables (`meta.json`, `pos_embed_table.f32`, `rope_inv_freq.f32`), shared by
    /// the image and video pipelines.
    func visionPositions() throws -> VisionPositions {
        try positionsCache.value(for: 0) {
            guard let image = bundle.manifest.image else {
                throw OmniSmallBackendError.failure("bundle manifest is missing the image pipeline")
            }
            let resources = bundle.resolve(image.resources)
            return try VisionPositions(
                metaURL: resources.appendingPathComponent("meta.json"),
                posTableURL: resources.appendingPathComponent("pos_embed_table.f32"),
                invFreqURL: resources.appendingPathComponent("rope_inv_freq.f32"))
        }
    }

    func melFrontend() throws -> GlossMelFrontend {
        try melCache.value(for: 0) { try GlossMelFrontend() }
    }

    /// The image preprocessor the manifest pins.
    func imagePreprocessor() throws -> GlossImagePreprocessor {
        guard let image = bundle.manifest.image else {
            throw OmniSmallBackendError.failure("bundle manifest is missing the image pipeline")
        }
        return GlossImagePreprocessor(
            minPixels: image.preprocess.minPixels,
            maxPixels: image.preprocess.maxPixels)
    }

    func imagePreparer() throws -> ImageInputPreparer {
        try imageCache.value(for: 0) {
            let manifest = bundle.manifest
            guard let image = manifest.image, let tokens = manifest.tokens.image,
                  let maxPatches = image.patchBuckets.max() else {
                throw OmniSmallBackendError.failure("bundle manifest is missing the image pipeline")
            }
            return ImageInputPreparer(
                preprocessor: try imagePreprocessor(),
                positions: try visionPositions(),
                tokens: tokens.mediaTokens,
                maxPatches: maxPatches)
        }
    }

    func videoPreparer() throws -> VideoInputPreparer {
        try videoCache.value(for: 0) {
            let manifest = bundle.manifest
            guard let video = manifest.video, let tokens = manifest.tokens.video,
                  let maxPatches = video.patchBuckets.max() else {
                throw OmniSmallBackendError.failure("bundle manifest is missing the video pipeline")
            }
            // The video processor's pixel budget is planned by `GlossVideoFile.plan`; the
            // preprocessor only supplies patch, merge, and temporal geometry, so its default is the
            // one the video pipeline has always used.
            return VideoInputPreparer(
                preprocessor: GlossImagePreprocessor(),
                positions: try visionPositions(),
                tokens: tokens.mediaTokens,
                maxPatches: maxPatches)
        }
    }

    func audioPreparer() throws -> AudioInputPreparer {
        try audioCache.value(for: 0) {
            guard let tokens = bundle.manifest.tokens.audio else {
                throw OmniSmallBackendError.failure("bundle manifest is missing the audio pipeline")
            }
            return AudioInputPreparer(mel: try melFrontend(), tokens: tokens.mediaTokens)
        }
    }
}

// MARK: - The CPU phase itself

/// Media preparation and the error taxonomy shared by both phases.
///
/// Input problems (undecodable image, bad WAV, oversized or undecodable video, too-short audio)
/// surface as `OmniSmallBackendError.invalidInput` (HTTP 400); anything else — including a
/// temporary-file write failure — is an `OmniSmallBackendError.failure` (HTTP 500). The mapping is
/// deliberately the one the single-phase path always had.
enum OmniSmallMediaPreparation {
    /// Prepare one media input synchronously. Runs on an executor thread; it must not touch any
    /// actor state and must poll `cancellation` between coarse steps.
    static func prepare(
        _ input: OmniSmall.Input,
        role: OmniSmallRole,
        host: OmniSmallMediaHost,
        cancellation: PreparationCancellation
    ) throws -> (kind: OmniSmallPreparedMedia.Kind, payload: OmniSmallPreparedMedia.Payload) {
        try cancellation.check()
        let prompt = role.prompt
        let result: (OmniSmallPreparedMedia.Kind, OmniSmallPreparedMedia.Payload)
        switch input {
        case .text:
            throw OmniSmallBackendError.failure("text input reached the media backend")
        case let .image(url):
            result = (.image, .image(try host.imagePreparer().prepare(imageURL: url, prompt: prompt)))
        case let .imageData(data):
            result = (.image, .image(try host.imagePreparer().prepare(imageData: data, prompt: prompt)))
        case let .audio(url):
            result = (.audio, .audio(try prepareAudio(url, prompt: prompt, host: host, cancellation: cancellation)))
        case let .audioData(data):
            result = (.audio, .audio(try TemporaryMediaFiles.withFile(
                data, extension: "wav", in: host.temporaryDirectory) {
                try prepareAudio($0, prompt: prompt, host: host, cancellation: cancellation)
            }))
        case let .video(url):
            result = (.video, .video(try host.videoPreparer().prepare(
                videoURL: url, prompt: prompt, checkCancellation: cancellation.check)))
        case let .videoData(data):
            result = (.video, .video(try TemporaryMediaFiles.withFile(
                data, extension: "mp4", in: host.temporaryDirectory) {
                try host.videoPreparer().prepare(
                    videoURL: $0, prompt: prompt, checkCancellation: cancellation.check)
            }))
        }
        try cancellation.check()
        return result
    }

    private static func prepareAudio(
        _ url: URL,
        prompt: GlossTextEmbedder.Prompt,
        host: OmniSmallMediaHost,
        cancellation: PreparationCancellation
    ) throws -> PreparedAudioInputs {
        let audio: [Float]
        do {
            audio = try decodeBoundedAudio(url)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OmniSmallBackendError {
            throw error
        } catch {
            throw OmniSmallBackendError.invalidInput(
                "audio file could not be decoded: \(error)")
        }
        guard !audio.isEmpty else {
            throw OmniSmallBackendError.invalidInput("audio file contains no samples")
        }
        try OmniSmallInputLimits.validateDecodedAudio(sampleCount: audio.count)
        guard audio.allSatisfy(\.isFinite) else {
            throw OmniSmallBackendError.invalidInput("audio contains non-finite samples")
        }
        try cancellation.check()
        return try host.audioPreparer().prepare(audio, prompt: prompt)
    }

    /// Reject clearly over-limit files from container metadata before allocating a decoded buffer.
    /// The exact decoded sample count is checked again by the caller after resampling.
    private static func decodeBoundedAudio(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let rate = file.processingFormat.sampleRate
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: file.length,
            sampleRate: rate,
            channelCount: file.processingFormat.channelCount)
        return try GlossAudioFile.decode16kMono(url)
    }

    /// Map any error from either phase to the backend error taxonomy. Cancellation is not an
    /// error to translate and is rethrown as itself.
    static func backendError(from error: any Error) -> any Error {
        switch error {
        case is CancellationError:
            return CancellationError()
        case let error as GlossImagePreprocessor.ImageError:
            return OmniSmallBackendError.invalidInput("image input could not be decoded: \(error)")
        case let error as GlossMelFrontend.MelError:
            return OmniSmallBackendError.invalidInput("audio input is invalid: \(error)")
        case let error as GlossVideoFile.DecodeError:
            return OmniSmallBackendError.invalidInput("video input could not be decoded: \(error)")
        case let error as VideoFrameDecoder.Failure:
            return OmniSmallBackendError.invalidInput("video input could not be decoded: \(error)")
        case let error as VideoCoreMLEncoderMasked.EncoderError:
            return OmniSmallBackendError.invalidInput("video input is invalid: \(error)")
        case let error as OmniSmallBackendError:
            return error
        default:
            return OmniSmallBackendError.failure(String(describing: error))
        }
    }
}
