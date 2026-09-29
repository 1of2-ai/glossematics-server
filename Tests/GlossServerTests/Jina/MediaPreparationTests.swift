import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
@_spi(Server) @testable import gloss_server

/// The split media path: CPU preparation on a bounded executor, then Core ML inference.
///
/// * The executor's contract (bounded, off the cooperative pool, cancellable).
/// * Preparation produces EXACTLY the arrays the original single-phase code fed Core ML — proven
///   against an independent recomputation from the primitives, on the golden fixture always and
///   with real-bundle vectors when `GLOSS_JINA_BUNDLE` is set.
/// * Error mapping, temporary-file cleanup, and cancellation.

private let goldenFixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle")

private let repositoryFixtures = goldenFixture.deletingLastPathComponent()
private let testSpace = "glossematics:omni-small:sha256:" + String(repeating: "a", count: 64)
private let testArtifact = "sha256:" + String(repeating: "c", count: 64)

// MARK: - Procedural inputs (shared with the parity tests)

/// A deterministic 640 x 480 PNG: not factor-aligned, so smart-resize and resampling both run.
private func proceduralPNG() throws -> Data {
    let width = 640, height = 480
    let rgb = OmniSmallVerificationInputs.rgbImage(height: height, width: width)
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    for pixel in 0..<(width * height) {
        rgba[pixel * 4] = rgb[pixel * 3]
        rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
        rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
    }
    let provider = try #require(CGDataProvider(data: Data(rgba) as CFData))
    let image = try #require(CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let data = NSMutableData()
    let destination = try #require(CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    #expect(CGImageDestinationFinalize(destination))
    return data as Data
}

/// A 16 kHz mono float WAV of `count` procedural samples.
private func proceduralWAV(sampleCount count: Int) throws -> Data {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-prep-test-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    let format = try #require(AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false))
    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
        buffer.frameLength = AVAudioFrameCount(count)
        let samples = OmniSmallVerificationInputs.audio(sampleCount: count)
        for index in 0..<count { buffer.floatChannelData?[0][index] = samples[index] }
        try file.write(from: buffer)
    }
    return try Data(contentsOf: url)
}

private func privateTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-prep-scratch-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func entries(in directory: URL) -> [String] {
    ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []).sorted()
}

// MARK: - Executor

private final class Gauge: @unchecked Sendable {
    private let lock = NSLock()
    private var running = 0
    private(set) var maximum = 0
    private(set) var started = 0
    func enter() { lock.lock(); running += 1; started += 1; maximum = max(maximum, running); lock.unlock() }
    func leave() { lock.lock(); running -= 1; lock.unlock() }
    var startedCount: Int { lock.lock(); defer { lock.unlock() }; return started }
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = false
    func set() { lock.lock(); stored = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return stored }
}

private struct PreparationFailure: Error, Equatable { let code: Int }

@Suite struct MediaPreparationExecutorTests {
    @Test func defaultConcurrencyIsAboutHalfTheCoresAndAtLeastTwo() {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        #expect(MediaPreparationExecutor.defaultConcurrency == max(2, cores / 2))
        #expect(MediaPreparationExecutor.shared.maxConcurrentJobs == MediaPreparationExecutor.defaultConcurrency)
    }

    @Test func concurrencyIsBoundedAndEveryJobRuns() async throws {
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 3, name: "test-bounded")
        let gauge = Gauge()
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for job in 0..<24 {
                group.addTask {
                    try await executor.run { _ in
                        gauge.enter()
                        defer { gauge.leave() }
                        Thread.sleep(forTimeInterval: 0.03)
                        return job
                    }
                }
            }
            return try await group.reduce(into: [Int]()) { $0.append($1) }
        }
        #expect(results.sorted() == Array(0..<24))
        #expect(gauge.maximum <= 3, "more than 3 jobs ran at once: \(gauge.maximum)")
        #expect(gauge.maximum >= 2, "the executor never overlapped jobs at all")
    }

    @Test func workRunsOffTheCooperativePool() async throws {
        // Block as many jobs as there are cores (twice over). Were they running on the Swift
        // cooperative pool (one thread per core) they would exhaust it and an unrelated task
        // could not run; on dedicated dispatch threads the pool stays free.
        let blockers = ProcessInfo.processInfo.activeProcessorCount * 2
        let executor = MediaPreparationExecutor(maxConcurrentJobs: blockers, name: "test-off-pool")
        let gate = DispatchSemaphore(value: 0)
        let allStarted = Gauge()
        let released = Flag()
        // Safety net that does not depend on the cooperative pool: never hang the suite.
        DispatchQueue.global().asyncAfter(deadline: .now() + 20) {
            released.set()
            for _ in 0..<blockers { gate.signal() }
        }

        let jobs = (0..<blockers).map { _ in
            Task { try await executor.run { _ in allStarted.enter(); gate.wait(); return () } }
        }
        while allStarted.startedCount < blockers, !released.isSet {
            try await Task.sleep(for: .milliseconds(10))
        }
        let ranWhileBlocked = Flag()
        let probe = await Task.detached { () -> Bool in
            if !released.isSet { ranWhileBlocked.set() }
            return true
        }.value
        #expect(probe)
        #expect(ranWhileBlocked.isSet, "the cooperative pool was starved by \(blockers) blocked jobs")
        for _ in 0..<blockers { gate.signal() }
        for job in jobs { try await job.value }
    }

    @Test func aQueuedJobIsCancelledImmediatelyAndNeverRuns() async throws {
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 1, name: "test-queued-cancel")
        let gate = DispatchSemaphore(value: 0)
        let blockerStarted = Flag()
        let blocker = Task { try await executor.run { _ in blockerStarted.set(); gate.wait(); return "blocker" } }
        while !blockerStarted.isSet { try await Task.sleep(for: .milliseconds(5)) }

        let ran = Flag()
        let queued = Task { try await executor.run { _ in ran.set(); return "queued" } }
        try await Task.sleep(for: .milliseconds(50))
        queued.cancel()
        // It resumes with CancellationError while the blocker STILL holds the only slot.
        do {
            _ = try await queued.value
            Issue.record("a cancelled queued job must not produce a value")
        } catch is CancellationError {
            // Expected.
        }
        gate.signal()
        #expect(try await blocker.value == "blocker")
        // The cancelled job's queue entry must not have run, and the executor is still healthy.
        #expect(try await executor.run { _ in "after" } == "after")
        #expect(!ran.isSet, "a job cancelled while queued ran anyway")
    }

    @Test func aRunningJobStopsAtItsNextCancellationCheck() async throws {
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 2, name: "test-running-cancel")
        let started = Flag()
        let iterations = Gauge()
        let job = Task {
            try await executor.run { cancellation in
                started.set()
                for _ in 0..<10_000 {
                    try cancellation.check()
                    iterations.enter(); iterations.leave()
                    Thread.sleep(forTimeInterval: 0.005)
                }
                return "finished"
            }
        }
        while !started.isSet { try await Task.sleep(for: .milliseconds(5)) }
        try await Task.sleep(for: .milliseconds(30))
        job.cancel()
        do {
            _ = try await job.value
            Issue.record("a cancelled running job must throw")
        } catch is CancellationError {
            // Expected: stopped long before 10,000 iterations (50 s).
        }
        #expect(iterations.startedCount < 5_000)
    }

    @Test func aCallerThatIsAlreadyCancelledNeverStartsWork() async throws {
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 2, name: "test-precancelled")
        let ran = Flag()
        let task = Task {
            while !Task.isCancelled { await Task.yield() }
            return try await executor.run { _ in ran.set(); return 1 }
        }
        task.cancel()
        do {
            _ = try await task.value
            Issue.record("must throw")
        } catch is CancellationError {
            // Expected.
        }
        #expect(!ran.isSet)
    }

    @Test func errorsPropagateUnchanged() async throws {
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 2, name: "test-errors")
        await #expect(throws: PreparationFailure(code: 7)) {
            try await executor.run { _ -> Int in throw PreparationFailure(code: 7) }
        }
    }
}

// MARK: - Preparation equals the original single-phase arrays

@Suite struct MediaPreparationParity {
    private func host() throws -> OmniSmallMediaHost {
        OmniSmallMediaHost(bundle: try GlossModelBundle(url: goldenFixture))
    }

    /// The image path as the pre-split code wrote it, recomputed from the primitives.
    @Test func imagePreparationMatchesTheOriginalComputation() throws {
        let host = try host()
        let preparer = try host.imagePreparer()
        let png = try proceduralPNG()
        for prompt in [GlossTextEmbedder.Prompt.query, .document] {
            let prepared = try preparer.prepare(imageData: png, prompt: prompt)

            let preprocessor = try host.imagePreprocessor()
            let positions = try host.visionPositions()
            let tokens = try #require(GlossModelBundle(url: goldenFixture).manifest.tokens.image).mediaTokens
            let cg = try GlossImagePreprocessor.loadCGImage(png)
            let (h, w) = preprocessor.smartResize(h: cg.height, w: cg.width, maxPixelsOverride: 5_120 * 256)
            let rgb = try preprocessor.resizedRGB(cg, w: w, h: h)
            let (pixels, gh, gw) = try preprocessor.pixelValues(rgb: rgb, h: h, w: w)
            let (posEmbeds, cos, sin, merged) = positions.compute(gh: gh, gw: gw)

            #expect(prepared.patches == gh * gw)
            #expect(prepared.pixelValues == pixels)
            #expect(prepared.posEmbeds == posEmbeds)
            #expect(prepared.ropeCos == cos && prepared.ropeSin == sin)
            #expect(prepared.mergedTokens == merged)
            #expect(prepared.temporalGroups == 1 && prepared.groupPatches == gh * gw)
            #expect(prepared.tokenIDs == tokens.ids(prompt: prompt, count: merged))
            #expect(prepared.scatterOffset == tokens.resolvedPrefix(for: prompt).count)
        }
    }

    @Test func audioPreparationMatchesTheOriginalComputation() throws {
        let host = try host()
        let preparer = try host.audioPreparer()
        let mel = try host.melFrontend()
        let tokens = try #require(GlossModelBundle(url: goldenFixture).manifest.tokens.audio).mediaTokens
        // Whole chunks, a partial boundary chunk, and the 480-sample minimum.
        for count in [480, 3_200, 30_400, 48_321, 96_000] {
            let samples = OmniSmallVerificationInputs.audio(sampleCount: count)
            let prepared = try preparer.prepare(samples, prompt: .document)

            let exactFrames = min((samples.count + mel.hop - 1) / mel.hop, 3_000)
            let bucket = AudioMasks.bucket(forFrames: exactFrames)
            let masks = AudioMasks(exactFrames: exactFrames, bucketFrames: bucket)
            var packed = try mel.packedMel(samples, frames: bucket)
            if exactFrames < bucket {
                for m in 0..<mel.nMels { for t in exactFrames..<bucket { packed[m * bucket + t] = 0 } }
            }
            #expect(prepared.packedMel == packed, "\(count) samples")
            #expect(prepared.masks.bucketFrames == bucket)
            #expect(prepared.masks.convMask == masks.convMask)
            #expect(prepared.masks.attnBias == masks.attnBias)
            #expect(prepared.tokenCount == masks.realTokens)
            #expect(prepared.tokenIDs == tokens.ids(prompt: .document, count: masks.realTokens))
            #expect(prepared.scatterOffset == tokens.resolvedPrefix(for: .document).count)
        }
    }

    @Test func videoPreparationMatchesTheOriginalComputation() throws {
        let host = try host()
        let preparer = try host.videoPreparer()
        let tokens = try #require(GlossModelBundle(url: goldenFixture).manifest.tokens.video).mediaTokens
        for name in ["golden-video.mp4", "golden-long-video.mp4"] {
            let url = repositoryFixtures.appendingPathComponent(name)
            let prepared = try preparer.prepare(videoURL: url, prompt: .query)

            let preprocessor = GlossImagePreprocessor()
            let positions = try host.visionPositions()
            let input = try GlossVideoFile.extractFrames(url, maxPatches: 2_048, preprocessor: preprocessor)
            let (pixels, t, gh, gw) = try preprocessor.videoPixelValues(frames: input.frames, h: input.h, w: input.w)
            let (posEmbeds, cos, sin, merged) = positions.computeVideo(t: t, gh: gh, gw: gw)

            #expect(prepared.patches == t * gh * gw, "\(name)")
            #expect(prepared.pixelValues == pixels, "\(name)")
            #expect(prepared.posEmbeds == posEmbeds, "\(name)")
            #expect(prepared.ropeCos == cos && prepared.ropeSin == sin, "\(name)")
            #expect(prepared.temporalGroups == t && prepared.groupPatches == gh * gw, "\(name)")
            #expect(prepared.mergedTokens == merged, "\(name)")
            #expect(prepared.tokenIDs == tokens.ids(prompt: .query, count: merged), "\(name)")
        }
    }

    @Test func splitPathThroughTheModelMatchesTheDirectEntryPoints() async throws {
        let model = try await OmniSmall.load(from: goldenFixture)
        let png = try proceduralPNG()
        let wav = try proceduralWAV(sampleCount: 16_000)
        let mp4 = try Data(contentsOf: repositoryFixtures.appendingPathComponent("golden-video.mp4"))
        for input in [OmniSmall.Input.imageData(png), .audioData(wav), .videoData(mp4)] {
            let prepared = try await model.prepareMedia(input, role: .document)
            let viaSplit = try await model.embedPreparedMedia(prepared, dimensions: .d256)
            let direct = try await model.embedDocument(input, dimensions: .d256).values
            #expect(viaSplit == direct)
            #expect(prepared.prepareMilliseconds > 0)
            #expect(prepared.queuedMilliseconds >= 0)
        }
        let image = try await model.prepareMedia(.imageData(png), role: .query)
        #expect(image.kind == .image && image.role == .query)
    }

    /// Real bundle: the split path, the direct entry points, and the ORIGINAL single-phase
    /// computation (re-run here from the primitives against separately loaded Core ML models) must
    /// return bit-identical vectors, because the same arrays reach the same functions.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["GLOSS_JINA_BUNDLE"] != nil,
                   "Set GLOSS_JINA_BUNDLE to the production bundle"))
    func realBundleSplitPathIsBitIdenticalToTheOriginal() async throws {
        let bundleURL = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GLOSS_JINA_BUNDLE"]))
        let bundle = try GlossModelBundle(url: bundleURL)
        let manifest = bundle.manifest
        let model = try await OmniSmall.load(from: bundleURL)

        let image = try #require(manifest.image), video = try #require(manifest.video)
        let audio = try #require(manifest.audio), decoder = try #require(manifest.decoder)
        let imageTokens = try #require(manifest.tokens.image).mediaTokens
        let videoTokens = try #require(manifest.tokens.video).mediaTokens
        let audioTokens = try #require(manifest.tokens.audio).mediaTokens

        // Independent pipelines: their own MLModel instances, their own decoder.
        let imagePipeline = try GlossImageEmbedderMasked(
            visionModelURL: bundle.resolve(image.encoder), embedModelURL: bundle.resolve(decoder.embed),
            decoderModelURL: bundle.resolve(decoder.model), resourcesDir: bundle.resolve(image.resources),
            tokens: imageTokens, patchBuckets: image.patchBuckets, padTokenID: manifest.tokens.padID,
            preprocessor: GlossImagePreprocessor(minPixels: image.preprocess.minPixels,
                                                 maxPixels: image.preprocess.maxPixels),
            sequenceBuckets: decoder.sequenceBuckets)
        let videoPipeline = try GlossVideoEmbedderMasked(
            visionModelURL: bundle.resolve(video.encoder), embedModelURL: bundle.resolve(decoder.embed),
            decoderModelURL: bundle.resolve(decoder.model), resourcesDir: bundle.resolve(image.resources),
            tokens: videoTokens, patchBuckets: video.patchBuckets, padTokenID: manifest.tokens.padID,
            sequenceBuckets: decoder.sequenceBuckets)
        let audioPipeline = try GlossAudioEmbedderMasked(
            audioModelURL: bundle.resolve(audio.encoder), embedModelURL: bundle.resolve(decoder.embed),
            decoderModelURL: bundle.resolve(decoder.model), tokens: audioTokens,
            padTokenID: manifest.tokens.padID, sequenceBuckets: decoder.sequenceBuckets)

        // --- image: the original body of `embed(imageData:)`, from primitives ---
        let png = try proceduralPNG()
        let cg = try GlossImagePreprocessor.loadCGImage(png)
        let (h, w) = imagePipeline.preprocessor.smartResize(
            h: cg.height, w: cg.width, maxPixelsOverride: imagePipeline.maxPatches * 256)
        let rgb = try imagePipeline.preprocessor.resizedRGB(cg, w: w, h: h)
        let (pixels, gh, gw) = try imagePipeline.preprocessor.pixelValues(rgb: rgb, h: h, w: w)
        let (posEmbeds, cos, sin, merged) = imagePipeline.positions.compute(gh: gh, gw: gw)
        let imageFull = try imagePipeline.encoder.encode(
            pixelValues: pixels, pixelDim: imagePipeline.preprocessor.featuresPerPatch, posEmbeds: posEmbeds,
            hidden: imagePipeline.positions.hidden, cos: cos, sin: sin,
            ropeDim: imagePipeline.positions.ropeDim, patches: gh * gw)
        // The daemon always asks for a Matryoshka size (1024 included), which re-normalizes.
        let originalImage = matryoshka(try imagePipeline.decoder.decode(
            tokenIds: imageTokens.ids(prompt: .document, count: merged),
            features: Array(imageFull[0 ..< (merged * imagePipeline.featureDim)]),
            scatterOffset: imageTokens.resolvedPrefix(for: .document).count), dim: 1_024)

        let preparedImage = try await model.prepareMedia(.imageData(png), role: .document)
        let splitImage = try await model.embedPreparedMedia(preparedImage, dimensions: .d1024)
        let directImage = try await model.embedDocument(.imageData(png)).values
        #expect(splitImage == originalImage, "image: split path vs original computation")
        #expect(directImage == originalImage, "image: direct entry point vs original computation")
        let lowLevelImage = try imagePipeline.embed(imageData: png, dim: 1_024, prompt: .document)
        #expect(lowLevelImage == originalImage, "image: refactored embedder vs original computation")

        // --- audio ---
        let wav = try proceduralWAV(sampleCount: 16_000 * 5)
        let samples = try withScratchFile(wav, "wav") { try GlossAudioFile.decode16kMono($0) }
        let mel = audioPipeline.mel
        let exactFrames = min((samples.count + mel.hop - 1) / mel.hop, 3_000)
        let bucket = AudioMasks.bucket(forFrames: exactFrames)
        let masks = AudioMasks(exactFrames: exactFrames, bucketFrames: bucket)
        var packed = try mel.packedMel(samples, frames: bucket)
        if exactFrames < bucket {
            for m in 0..<mel.nMels { for t in exactFrames..<bucket { packed[m * bucket + t] = 0 } }
        }
        let audioFull = try audioPipeline.encoder.encode(packedMel: packed, nMels: mel.nMels, masks: masks)
        let originalAudio = matryoshka(try audioPipeline.decoder.decode(
            tokenIds: audioTokens.ids(prompt: .document, count: masks.realTokens),
            features: Array(audioFull[0 ..< (masks.realTokens * audioPipeline.featureDim)]),
            scatterOffset: audioTokens.resolvedPrefix(for: .document).count), dim: 1_024)

        let preparedAudio = try await model.prepareMedia(.audioData(wav), role: .document)
        let splitAudio = try await model.embedPreparedMedia(preparedAudio, dimensions: .d1024)
        let directAudio = try await model.embedDocument(.audioData(wav)).values
        #expect(splitAudio == originalAudio, "audio: split path vs original computation")
        #expect(directAudio == originalAudio, "audio: direct entry point vs original computation")

        // --- video (the golden MP4) ---
        let mp4URL = repositoryFixtures.appendingPathComponent("golden-video.mp4")
        let mp4 = try Data(contentsOf: mp4URL)
        let extracted = try GlossVideoFile.extractFrames(
            mp4URL, maxPatches: videoPipeline.encoder.patchBuckets.last ?? 0,
            preprocessor: videoPipeline.preprocessor)
        let (videoPixels, t, vgh, vgw) = try videoPipeline.preprocessor.videoPixelValues(
            frames: extracted.frames, h: extracted.h, w: extracted.w)
        let (vPos, vCos, vSin, vMerged) = videoPipeline.positions.computeVideo(t: t, gh: vgh, gw: vgw)
        let videoFull = try videoPipeline.encoder.encode(
            pixelValues: videoPixels, pixelDim: videoPipeline.preprocessor.featuresPerPatch, posEmbeds: vPos,
            hidden: videoPipeline.positions.hidden, cos: vCos, sin: vSin,
            ropeDim: videoPipeline.positions.ropeDim, frames: t, framePatches: vgh * vgw)
        let originalVideo = matryoshka(try videoPipeline.decoder.decode(
            tokenIds: videoTokens.ids(prompt: .document, count: vMerged),
            features: Array(videoFull[0 ..< (vMerged * videoPipeline.featureDim)]),
            scatterOffset: videoTokens.resolvedPrefix(for: .document).count), dim: 1_024)

        let preparedVideo = try await model.prepareMedia(.videoData(mp4), role: .document)
        let splitVideo = try await model.embedPreparedMedia(preparedVideo, dimensions: .d1024)
        let directVideo = try await model.embedDocument(.videoData(mp4)).values
        #expect(splitVideo == originalVideo, "video: split path vs original computation")
        #expect(directVideo == originalVideo, "video: direct entry point vs original computation")

        // Cosine 1.0 (in addition to exact equality) is what the parity contract promises.
        for (name, split, original) in [("image", splitImage, originalImage),
                                        ("audio", splitAudio, originalAudio),
                                        ("video", splitVideo, originalVideo)] {
            let similarity = cosine(split, original)
            print("parity \(name): exact=\(split == original) cosine=\(similarity)")
            #expect(abs(similarity - 1) < 1e-6, "\(name)")
        }
    }
}

private func withScratchFile<T>(_ data: Data, _ ext: String, _ body: (URL) throws -> T) throws -> T {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gloss-prep-file-\(UUID().uuidString).\(ext)")
    try data.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    return try body(url)
}

// MARK: - Error mapping, temp files, cancellation

@Suite struct MediaPreparationErrors {
    private func model(
        executor: MediaPreparationExecutor = MediaPreparationExecutor(maxConcurrentJobs: 2, name: "test-errors"),
        temporaryDirectory: URL
    ) throws -> OmniSmall {
        let bundle = try GlossModelBundle(url: goldenFixture)
        return OmniSmall(
            backend: OmniSmallProductionBackend(
                bundle: bundle, executor: executor, temporaryDirectory: temporaryDirectory),
            dimensions: .d1024, space: testSpace, artifactFingerprint: testArtifact)
    }

    @Test func inputProblemsAreInvalidInputAndTemporaryFilesAreRemoved() async throws {
        let scratch = try privateTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let model = try model(temporaryDirectory: scratch)
        let garbage = Data("this is not media".utf8)

        // Each undecodable upload is the caller's fault (400), not a server error, in the split
        // path and through the legacy entry point — including the recent VideoFrameDecoder.Failure
        // -> 400 mapping for video.
        for input in [OmniSmall.Input.imageData(garbage), .audioData(garbage), .videoData(garbage)] {
            do {
                _ = try await model.prepareMedia(input, role: .document)
                Issue.record("undecodable media must not prepare: \(input)")
            } catch let error as OmniSmallError {
                guard case .invalidInput = error else {
                    Issue.record("expected invalidInput for \(input), got \(error)")
                    continue
                }
            }
            do {
                _ = try await model.embedDocument(input)
                Issue.record("undecodable media must not embed: \(input)")
            } catch let error as OmniSmallError {
                guard case .invalidInput = error else {
                    Issue.record("legacy path: expected invalidInput for \(input), got \(error)")
                    continue
                }
            }
        }
        // audioData/videoData were staged in the private directory; nothing may remain.
        #expect(entries(in: scratch).isEmpty, "temporary media survived: \(entries(in: scratch))")

        do {
            _ = try await model.prepareMedia(.text("not media"), role: .query)
            Issue.record("text is not media")
        } catch let error as OmniSmallError {
            guard case .invalidInput = error else {
                Issue.record("unexpected \(error)")
                return
            }
        }
    }

    @Test func successfulPreparationAlsoRemovesItsStagedFile() async throws {
        let scratch = try privateTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let model = try model(temporaryDirectory: scratch)
        let mp4 = try Data(contentsOf: repositoryFixtures.appendingPathComponent("golden-video.mp4"))
        let wav = try proceduralWAV(sampleCount: 8_000)
        _ = try await model.prepareMedia(.videoData(mp4), role: .document)
        _ = try await model.prepareMedia(.audioData(wav), role: .query)
        #expect(entries(in: scratch).isEmpty)
    }

    @Test func aCancelledQueuedPreparationNeverWritesItsFile() async throws {
        let scratch = try privateTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let executor = MediaPreparationExecutor(maxConcurrentJobs: 1, name: "test-prep-cancel")
        let model = try model(executor: executor, temporaryDirectory: scratch)
        let mp4 = try Data(contentsOf: repositoryFixtures.appendingPathComponent("golden-video.mp4"))

        let gate = DispatchSemaphore(value: 0)
        let blockerStarted = Flag()
        let blocker = Task { try await executor.run { _ in blockerStarted.set(); gate.wait(); return () } }
        while !blockerStarted.isSet { try await Task.sleep(for: .milliseconds(5)) }

        let queued = Task { try await model.prepareMedia(.videoData(mp4), role: .document) }
        try await Task.sleep(for: .milliseconds(50))
        queued.cancel()
        do {
            _ = try await queued.value
            Issue.record("a cancelled preparation must throw")
        } catch is CancellationError {
            // Expected.
        }
        gate.signal()
        try await blocker.value
        #expect(entries(in: scratch).isEmpty)
    }

    @Test func manyConcurrentPreparationsMatchSequentialOnes() async throws {
        let scratch = try privateTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let model = try model(temporaryDirectory: scratch)
        let png = try proceduralPNG()
        let mp4 = try Data(contentsOf: repositoryFixtures.appendingPathComponent("golden-video.mp4"))
        let wav = try proceduralWAV(sampleCount: 24_000)
        let inputs: [OmniSmall.Input] = [.imageData(png), .videoData(mp4), .audioData(wav)]

        var expected = [[Float]]()
        for input in inputs {
            expected.append(try await model.embedPreparedMedia(
                try await model.prepareMedia(input, role: .document), dimensions: .d1024))
        }
        let results = try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            for index in 0..<18 {
                let input = inputs[index % inputs.count]
                group.addTask {
                    let prepared = try await model.prepareMedia(input, role: .document)
                    return (index, try await model.embedPreparedMedia(prepared, dimensions: .d1024))
                }
            }
            return try await group.reduce(into: [(Int, [Float])]()) { $0.append($1) }
        }
        #expect(results.count == 18)
        for (index, vector) in results { #expect(vector == expected[index % inputs.count]) }
        #expect(entries(in: scratch).isEmpty)
    }

    @Test func aBackendWithoutASplitPhaseStillWorksThroughThePassthrough() async throws {
        let model = OmniSmall(
            backend: EchoBackend(), dimensions: .d32, space: testSpace, artifactFingerprint: testArtifact)
        let prepared = try await model.prepareMedia(.imageData(Data([9])), role: .query)
        #expect(prepared.kind == .image && prepared.role == .query)
        let vector = try await model.embedPreparedMedia(prepared, dimensions: .d32)
        #expect(vector.count == 32 && vector[9] == 1)
        let direct = try await model.embedQuery(.imageData(Data([9])))
        #expect(direct.values == vector)
    }
}

/// A backend whose `embedMedia` is the whole story: it exercises the protocol's default
/// prepare/infer implementations.
private actor EchoBackend: OmniSmallBackend {
    func prepareTexts(_ texts: [String], role: OmniSmallRole) async throws -> [ValidatedText] { [] }
    func embedTexts(_ rows: [ValidatedText], dimensions: OmniSmall.Dimensions) async throws -> [[Float]] { [] }
    func embedMedia(_ input: OmniSmall.Input, role: OmniSmallRole, dimensions: OmniSmall.Dimensions) async throws -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions.rawValue)
        if case let .imageData(data) = input { vector[Int(data.first ?? 0) % dimensions.rawValue] = 1 }
        return vector
    }
}
