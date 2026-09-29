import CoreML
import Darwin
import Foundation

// MARK: - Report types

/// The outcome of loading and running ONE Core ML function during startup verification.
@_spi(Server)
public struct OmniSmallFunctionCheck: Sendable, Codable, Equatable {
    /// The model package: `text`, `image`, `video`, `audio`, `embed`, or `decoder`.
    public let model: String
    /// The function inside it: `bucket_32`, `bucket_32_b64`, `f1024`, ...
    public let function: String
    /// Requested Core ML compute units (`cpu+ane`, `cpu+gpu`, ...) when known. This is the
    /// placement the runtime asks for; Core ML may still schedule individual ops elsewhere.
    public let computeUnits: String?
    /// Time to obtain the loaded function, in milliseconds. Near zero when it was already
    /// resident (a warm start, or a function loaded earlier in the same run).
    public let loadMilliseconds: Double
    /// Time for this function's own first prediction, in milliseconds. It includes any lazy
    /// first-run compilation Core ML defers past load, so it is a cold number, not a steady-state
    /// latency. Reference predictions used only for comparison are not included.
    public let runMilliseconds: Double
    /// The function this one was compared against (`nil` for the anchor of its group).
    public let referenceFunction: String?
    /// The lowest cosine similarity observed against the reference (`nil` for an anchor).
    public let minimumCosine: Double?
    /// The consistency threshold `minimumCosine` had to reach.
    public let cosineThreshold: Double?
    public let passed: Bool
    /// Why it failed, when it did.
    public let failure: String?

    /// `model.function`, e.g. `text.bucket_32_b64`.
    public var name: String { "\(model).\(function)" }

    private enum CodingKeys: String, CodingKey {
        case model, function, computeUnits, loadMilliseconds, runMilliseconds
        case referenceFunction, minimumCosine, cosineThreshold, passed, failure
    }
}

/// Everything startup verification measured, for `/health` and the startup log.
@_spi(Server)
public struct OmniSmallVerificationReport: Sendable, Codable, Equatable {
    /// One entry per function the bundle declares, in execution order.
    public let functions: [OmniSmallFunctionCheck]
    /// Wall-clock time of the whole verification, in milliseconds (including reference
    /// predictions and input construction, which the per-function numbers exclude).
    public let wallMilliseconds: Double
    /// Sum of the per-function load times.
    public let loadMilliseconds: Double
    /// Sum of the per-function run times.
    public let runMilliseconds: Double
    /// Process resident memory before and after, in bytes (0 when the OS query failed).
    public let residentBytesBefore: UInt64
    public let residentBytesAfter: UInt64
    /// Process physical footprint (resident plus compressed and wired accelerator memory) before
    /// and after, in bytes. This is the number that grows when Core ML maps model weights.
    public let footprintBytesBefore: UInt64
    public let footprintBytesAfter: UInt64

    public var passedCount: Int { functions.filter(\.passed).count }
    public var failedCount: Int { functions.count - passedCount }
    public var allPassed: Bool { failedCount == 0 }
    /// The lowest consistency cosine across every compared function, if any were compared.
    public var minimumCosine: Double? { functions.compactMap(\.minimumCosine).min() }
    /// Growth in resident memory over the run (negative if the process shrank).
    public var residentDeltaBytes: Int64 { Int64(residentBytesAfter) - Int64(residentBytesBefore) }
    /// Growth in physical footprint over the run.
    public var footprintDeltaBytes: Int64 { Int64(footprintBytesAfter) - Int64(footprintBytesBefore) }

    public init(
        functions: [OmniSmallFunctionCheck],
        wallMilliseconds: Double,
        residentBytesBefore: UInt64 = 0,
        residentBytesAfter: UInt64 = 0,
        footprintBytesBefore: UInt64 = 0,
        footprintBytesAfter: UInt64 = 0
    ) {
        self.functions = functions
        self.wallMilliseconds = wallMilliseconds
        self.loadMilliseconds = functions.reduce(0) { $0 + $1.loadMilliseconds }
        self.runMilliseconds = functions.reduce(0) { $0 + $1.runMilliseconds }
        self.residentBytesBefore = residentBytesBefore
        self.residentBytesAfter = residentBytesAfter
        self.footprintBytesBefore = footprintBytesBefore
        self.footprintBytesAfter = footprintBytesAfter
    }

    /// Codable keys for the stored properties only; the computed summaries are derived.
    private enum CodingKeys: String, CodingKey {
        case functions, wallMilliseconds, loadMilliseconds, runMilliseconds
        case residentBytesBefore, residentBytesAfter, footprintBytesBefore, footprintBytesAfter
    }
}

/// A verification run found a function that would fault or give wrong numbers.
@_spi(Server)
public struct OmniSmallVerificationError: Error, Sendable, CustomStringConvertible {
    public enum Reason: Sendable, Equatable, CustomStringConvertible {
        /// Core ML could not load the function.
        case loadFailed(String)
        /// The prediction (or building its input) threw.
        case runFailed(String)
        /// The output was non-finite, mis-shaped, or an embedding was not unit norm.
        case invalidOutput(String)
        /// The output was finite and well-formed but disagreed with the reference function.
        case inconsistent(cosine: Double, threshold: Double, reference: String)
        /// Verification could not start (a model or tokenizer would not open).
        case setupFailed(String)

        public var description: String {
            switch self {
            case let .loadFailed(reason): "could not load: \(reason)"
            case let .runFailed(reason): "prediction failed: \(reason)"
            case let .invalidOutput(reason): "invalid output: \(reason)"
            case let .inconsistent(cosine, threshold, reference):
                "disagrees with \(reference) (cosine \(cosine), required \(threshold))"
            case let .setupFailed(reason): "setup failed: \(reason)"
            }
        }
    }

    /// The first function that failed, as `model.function` (`setup` for a startup failure).
    public let function: String
    public let reason: Reason
    /// Every function checked, including all failures — the run always completes so an operator
    /// sees the full picture, not just the first fault.
    public let report: OmniSmallVerificationReport

    public var description: String {
        let failed = report.failedCount
        let extra = failed > 1 ? " (\(failed) of \(report.functions.count) functions failed)" : ""
        return "OmniSmall function verification failed at \(function): \(reason)\(extra)"
    }
}

// MARK: - Process memory

/// The process's memory as the OS accounts it.
struct ProcessMemory: Sendable, Equatable {
    var residentBytes: UInt64
    var footprintBytes: UInt64

    /// Current values; zeros if the Mach query fails.
    static func current() -> ProcessMemory {
        var basic = mach_task_basic_info()
        var basicCount = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let basicStatus = withUnsafeMutablePointer(to: &basic) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(basicCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &basicCount)
            }
        }
        var vm = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmStatus = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }
        return ProcessMemory(
            residentBytes: basicStatus == KERN_SUCCESS ? UInt64(basic.resident_size) : 0,
            footprintBytes: vmStatus == KERN_SUCCESS ? UInt64(vm.phys_footprint) : 0)
    }
}

// MARK: - Deterministic procedural inputs

/// Inputs for verification, all generated in code so no fixture files ship. Each is deterministic
/// (integer arithmetic and a fixed LCG; no clock, no randomness) and non-trivial (structure at
/// several scales), so a function that ignores its input or returns garbage cannot hide behind a
/// blank image or silence.
enum OmniSmallVerificationInputs {
    /// The fixed passage embedded with the document prompt as the single-row reference. It must
    /// stay short enough for the smallest text bucket (32 tokens including conditioning).
    static let referencePassage = "Verification passage: retrieval maps text, images, audio, and video into one space."

    private static let vocabulary = [
        "harbor", "lantern", "orchard", "compass", "granite", "meadow", "violin", "glacier",
        "ledger", "saffron", "tundra", "beacon", "quartz", "willow", "anchor", "cobalt",
        "thistle", "ember", "atlas", "juniper", "marble", "raven", "summit", "canvas",
        "bramble", "cipher", "dune", "falcon", "gable", "hollow", "iris", "jasper",
        "kettle", "lagoon", "mosaic", "nectar", "opal", "prairie", "quiver", "ridge",
    ]

    /// A long pseudo-prose string; tokenized once, then sliced into batch rows of varied length.
    static func poolText(words: Int = 1_400) -> String {
        (0..<words).map { vocabulary[($0 * 7 + $0 / 3) % vocabulary.count] }.joined(separator: " ")
    }

    /// Packed RGB, row-major `height * width * 3`: a two-axis gradient, a checker, a disc, and
    /// diagonal stripes, so every patch differs from its neighbours.
    static func rgbImage(height: Int, width: Int) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: height * width * 3)
        let centerX = width / 2, centerY = height / 2, radius = min(width, height) / 4
        for y in 0..<height {
            for x in 0..<width {
                var red = x * 255 / max(1, width - 1)
                var green = y * 255 / max(1, height - 1)
                var blue = ((x / 24 + y / 24) % 2 == 0) ? 200 : 60
                let dx = x - centerX, dy = y - centerY
                if dx * dx + dy * dy < radius * radius { (red, green, blue) = (250, 220, 40) }
                if (x + y) % 64 < 6 { (red, green, blue) = (20, 20, 20) }
                let offset = (y * width + x) * 3
                rgb[offset] = UInt8(red)
                rgb[offset + 1] = UInt8(green)
                rgb[offset + 2] = UInt8(blue)
            }
        }
        return rgb
    }

    /// `count` frames of `rgbImage`-style content that change over time (a drifting gradient and a
    /// moving block), so consecutive frames differ.
    static func videoFrames(count: Int, height: Int, width: Int) -> [[UInt8]] {
        (0..<count).map { frame in
            var rgb = [UInt8](repeating: 0, count: height * width * 3)
            let blockX = (frame * 23) % max(1, width - 40), blockY = (frame * 31) % max(1, height - 40)
            for y in 0..<height {
                for x in 0..<width {
                    var red = (x * 255 / max(1, width - 1) + frame * 40) % 256
                    var green = y * 255 / max(1, height - 1)
                    var blue = (x + y + frame * 17) % 256
                    if x >= blockX, x < blockX + 40, y >= blockY, y < blockY + 40 {
                        (red, green, blue) = (240, 30 + frame * 20, 30)
                    }
                    let offset = (y * width + x) * 3
                    rgb[offset] = UInt8(red)
                    rgb[offset + 1] = UInt8(green)
                    rgb[offset + 2] = UInt8(blue)
                }
            }
            return rgb
        }
    }

    /// 16 kHz mono: a rising tone, a pulsed upper partial, and a little LCG noise.
    static func audio(sampleCount: Int) -> [Float] {
        var state: UInt32 = 0x1234_5678
        return (0..<sampleCount).map { index in
            state = state &* 1_664_525 &+ 1_013_904_223
            let noise = Float(state >> 8) / Float(1 << 24) - 0.5
            let t = Float(index) / 16_000
            let envelope = 0.5 + 0.5 * sinf(2 * .pi * 3 * t)
            return 0.35 * sinf(2 * .pi * (200 + 400 * t) * t)
                + 0.20 * envelope * sinf(2 * .pi * 1_250 * t)
                + 0.05 * noise
        }
    }
}

// MARK: - Thresholds

/// Consistency thresholds for the self-consistency checks, each set with an explicit margin over
/// what a healthy function was MEASURED to do.
///
/// The checks compare a function against a sibling that must compute the same thing (the same
/// tokens through a different bucket, the same rows through the batch function, the same image
/// through a different padded bucket). They need no golden vectors, so they run on any chip:
/// numerics differ between chips and between the ANE (fp16) and the GPU, but a healthy sibling
/// comparison stays within ~5e-5 of 1, while a "finite but wrong" function (a bad kernel,
/// mis-masked padding, a chip-specific fault) lands orders of magnitude further away.
///
/// Observed on the production `JinaV5OmniSmall.w8a16.bundle`, Apple M4 Max, macOS 27, all 41
/// functions (lowest cosine per group; the anchor of each group is the reference):
///
/// | group (reference)                       | observed min cosine | worst deviation | threshold | allowed / observed |
/// |-----------------------------------------|---------------------|-----------------|-----------|--------------------|
/// | text single bucket (`bucket_32`)        | 0.9999849 (GPU 256+); 1.0 (ANE 64, 128) | 1.5e-5 | 0.999   | 66x  |
/// | text batch rung rows (single-row twin)  | 0.9999896 (b64@32)  | 1.0e-5          | 0.999     | 96x                |
/// | image encoder bucket (`f1024`)          | 0.9999529           | 4.7e-5          | 0.999     | 21x                |
/// | video encoder bucket (`f256`)           | 0.9999484 (f1024)   | 5.2e-5          | 0.998     | 38x                |
/// | audio encoder bucket (`f200`)           | 0.9999789 (f1600)   | 2.1e-5          | 0.999     | 48x                |
/// | embed rows (`f128`, exact lookup)       | 0.9999999999999999  | 1.1e-16         | 0.99999   | 1e11x              |
/// | decoder bucket (`f128`)                 | 0.9999774 (GPU 512+); 1.0 (ANE 256) | 2.3e-5 | 0.999 | 43x |
///
/// The rule: the allowed deviation `1 - threshold` is at least 20x the worst deviation seen. The
/// margin absorbs a different chip generation (older ANEs accumulate fp16 differently) without
/// hiding real faults. If a healthy chip ever fails a threshold, widen it from measured data, not
/// by guess: run `verifyAllFunctionsOnTheRealBundle` and read the printed cosines.
enum OmniSmallVerificationThresholds {
    static let textBucket = 0.999
    static let textBatchRow = 0.999
    static let imageBucket = 0.999
    static let videoBucket = 0.998
    static let audioBucket = 0.999
    static let embedRows = 0.99999
    static let decoderBucket = 0.999
}

// MARK: - Verifier

/// Loads and runs every function the validated bundle declares, checking each output and
/// cross-checking functions that must agree. Runs synchronously; the backend calls it on its actor
/// so the functions it loads stay in the same caches serving uses (verification leaves them warm).
final class OmniSmallFunctionVerifier {
    /// Fault injection for tests: called with `model.function` and each output vector/row right
    /// after Core ML returns, so a test can corrupt one function and prove the checks catch it.
    typealias FaultInjector = @Sendable (_ function: String, _ values: inout [Float]) -> Void

    /// A check failed; carries the reason. Internal control flow only.
    private struct Failure: Error {
        let reason: OmniSmallVerificationError.Reason
    }

    private let text: GlossTextEmbedder
    private let image: GlossImageEmbedderMasked
    private let video: GlossVideoEmbedderMasked
    private let audio: GlossAudioEmbedderMasked
    private let decoder: GeneralMediaDecoder
    private let validateEmbedding: @Sendable ([Float]) throws -> Void
    private let progress: (@Sendable (OmniSmallFunctionCheck) -> Void)?
    private let faults: FaultInjector?

    private var checks: [OmniSmallFunctionCheck] = []
    private var firstFailure: (function: String, reason: OmniSmallVerificationError.Reason)?

    /// Real merged features an encoder produced, kept so the decoder group can decode a media
    /// sequence with realistic magnitudes. Video is preferred, then audio, then image.
    private struct FeatureSource {
        var values: [Float]
        var count: Int
        var tokens: MediaTokens
    }
    private var imageFeatures: FeatureSource?
    private var videoFeatures: FeatureSource?
    private var audioFeatures: FeatureSource?

    init(
        text: GlossTextEmbedder,
        image: GlossImageEmbedderMasked,
        video: GlossVideoEmbedderMasked,
        audio: GlossAudioEmbedderMasked,
        decoder: GeneralMediaDecoder,
        validateEmbedding: @escaping @Sendable ([Float]) throws -> Void,
        progress: (@Sendable (OmniSmallFunctionCheck) -> Void)?,
        faults: FaultInjector? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
        self.decoder = decoder
        self.validateEmbedding = validateEmbedding
        self.progress = progress
        self.faults = faults
    }

    /// Verify everything. Throws `CancellationError` if cancelled; throws
    /// ``OmniSmallVerificationError`` after the full run when any function failed.
    func run() throws -> OmniSmallVerificationReport {
        let started = ContinuousClock.now
        let before = ProcessMemory.current()

        try verifyText()
        try verifyImage()
        try verifyVideo()
        try verifyAudio()
        try verifyDecoder()

        let after = ProcessMemory.current()
        let report = OmniSmallVerificationReport(
            functions: checks,
            wallMilliseconds: verificationMilliseconds(ContinuousClock.now - started),
            residentBytesBefore: before.residentBytes,
            residentBytesAfter: after.residentBytes,
            footprintBytesBefore: before.footprintBytes,
            footprintBytesAfter: after.footprintBytes)
        if let failure = firstFailure {
            throw OmniSmallVerificationError(
                function: failure.function, reason: failure.reason, report: report)
        }
        return report
    }

    // MARK: Text

    private func verifyText() throws {
        let buckets = text.availableBuckets.sorted()
        guard let anchorBucket = buckets.first else { return }
        let referenceIDs = text.tokenIDs(for: Self.referencePassageText, prompt: .document)
        var anchor: [Float]?

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "text", function: "bucket_\(bucket)",
                units: text.units(forBucket: bucket),
                reference: isAnchor ? nil : "bucket_\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.textBucket
            ) { timing in
                guard referenceIDs.count <= anchorBucket else {
                    throw Failure(reason: .invalidOutput(
                        "verification passage is \(referenceIDs.count) tokens; smallest bucket is \(anchorBucket)"))
                }
                let encoder = try timing.load { try self.text.singleRowEncoder(bucket: bucket) }
                var vector = try timing.run { try encoder.encode(tokenIds: referenceIDs) }
                self.faults?("text.bucket_\(bucket)", &vector)
                try self.requireEmbedding(vector)
                if isAnchor {
                    anchor = vector
                    return nil
                }
                guard let anchor else { throw Self.missingReference("bucket_\(anchorBucket)") }
                return try Self.consistency(vector, anchor)
            }
        }

        for pair in text.batchGeometry {
            let function = "bucket_\(pair.bucket)_b\(pair.size)"
            try check(
                model: "text", function: function,
                units: text.units(forBucket: pair.bucket),
                reference: "bucket_\(pair.bucket)",
                threshold: OmniSmallVerificationThresholds.textBatchRow
            ) { timing in
                let rows = self.batchRows(size: pair.size, bucket: pair.bucket)
                let batch = try timing.load { try self.text.batchRowEncoder(bucket: pair.bucket) }
                var outputs = try timing.run { try batch.encodeBatch(tokenIds: rows) }
                guard outputs.count == pair.size else {
                    throw Failure(reason: .invalidOutput(
                        "batch returned \(outputs.count) rows for \(pair.size) inputs"))
                }
                // The reference is the single-row function of the same bucket, fed the same rows.
                let single = try self.text.singleRowEncoder(bucket: pair.bucket)
                var worst = Double.infinity
                for index in outputs.indices {
                    self.faults?("text.\(function)", &outputs[index])
                    try self.requireEmbedding(outputs[index])
                    let expected = try single.encode(tokenIds: rows[index])
                    worst = Self.lowest(worst, cosine(outputs[index], expected))
                }
                return worst
            }
        }
    }

    private static let referencePassageText = OmniSmallVerificationInputs.referencePassage

    /// `size` deterministic rows of real tokenized text for a batch function of `bucket`, with
    /// lengths spread from a few tokens up to exactly `bucket` (so the boundary row has no padding
    /// at all). Each row is the document conditioning followed by a slice of a tokenized
    /// pseudo-prose pool at a different offset.
    private func batchRows(size: Int, bucket: Int) -> [[Int32]] {
        let prefix = text.tokenIDs(for: "", prompt: .document)
        let pool = text.tokenIDs(for: OmniSmallVerificationInputs.poolText(), prompt: .none)
        let body = Array(pool.isEmpty ? [text.padTokenID] : pool)
        return (0..<size).map { index in
            let length = index == size - 1
                ? bucket
                : min(bucket, max(prefix.count + 1, 4 + (index * 37 + 11) % max(1, bucket - 3)))
            let offset = (index * 53) % body.count
            var row = Array(prefix.prefix(length))
            var cursor = offset
            while row.count < length {
                row.append(body[cursor % body.count])
                cursor += 1
            }
            return row
        }
    }

    // MARK: Image

    private func verifyImage() throws {
        let buckets = image.encoder.patchBuckets.sorted()
        guard let anchorBucket = buckets.first else { return }
        // 480 x 480 = 30 x 30 = 900 patches: inside the smallest bucket with padding to spare.
        let height = 480, width = 480
        let prepared = Result {
            try self.image.preparer.prepare(
                rgb: OmniSmallVerificationInputs.rgbImage(height: height, width: width),
                h: height, w: width, prompt: .document)
        }
        var anchor: [Float]?

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "image", function: "f\(bucket)",
                units: image.encoder.computeUnits,
                reference: isAnchor ? nil : "f\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.imageBucket
            ) { timing in
                let input = try Self.input(prepared)
                guard input.patches <= anchorBucket else {
                    throw Failure(reason: .invalidOutput(
                        "verification image is \(input.patches) patches; smallest bucket is \(anchorBucket)"))
                }
                _ = try timing.load { try self.image.encoder.model(bucket) }
                var full = try timing.run {
                    try self.image.encoder.encode(
                        pixelValues: input.pixelValues, pixelDim: self.image.preprocessor.featuresPerPatch,
                        posEmbeds: input.posEmbeds, hidden: self.image.positions.hidden,
                        cos: input.ropeCos, sin: input.ropeSin, ropeDim: self.image.positions.ropeDim,
                        patches: input.patches, bucket: bucket)
                }
                self.faults?("image.f\(bucket)", &full)
                try Self.requireFeatures(full, rows: bucket / 4, width: self.image.featureDim)
                let real = Array(full[0 ..< (input.mergedTokens * self.image.featureDim)])
                if isAnchor {
                    anchor = real
                    self.imageFeatures = FeatureSource(values: real, count: input.mergedTokens, tokens: self.image.tokens)
                    return nil
                }
                guard let anchor else { throw Self.missingReference("f\(anchorBucket)") }
                return try Self.consistency(real, anchor)
            }
        }
    }

    // MARK: Video

    private func verifyVideo() throws {
        let buckets = video.encoder.patchBuckets.sorted()
        guard let anchorBucket = buckets.first else { return }
        // 4 frames of 224 x 128 = 2 temporal groups of 14 x 8 = 112 patches: 224 patches total.
        let height = 224, width = 128
        let prepared = Result {
            try self.video.preparer.prepare(
                frames: OmniSmallVerificationInputs.videoFrames(count: 4, height: height, width: width),
                h: height, w: width, prompt: .document)
        }
        var anchor: [Float]?

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "video", function: "f\(bucket)",
                units: video.encoder.computeUnits,
                reference: isAnchor ? nil : "f\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.videoBucket
            ) { timing in
                let input = try Self.input(prepared)
                guard input.patches <= anchorBucket else {
                    throw Failure(reason: .invalidOutput(
                        "verification clip is \(input.patches) patches; smallest bucket is \(anchorBucket)"))
                }
                _ = try timing.load { try self.video.encoder.model(bucket) }
                var full = try timing.run {
                    try self.video.encoder.encode(
                        pixelValues: input.pixelValues, pixelDim: self.video.preprocessor.featuresPerPatch,
                        posEmbeds: input.posEmbeds, hidden: self.video.positions.hidden,
                        cos: input.ropeCos, sin: input.ropeSin, ropeDim: self.video.positions.ropeDim,
                        frames: input.temporalGroups, framePatches: input.groupPatches, bucket: bucket)
                }
                self.faults?("video.f\(bucket)", &full)
                try Self.requireFeatures(full, rows: bucket / 4, width: self.video.featureDim)
                let real = Array(full[0 ..< (input.mergedTokens * self.video.featureDim)])
                if isAnchor {
                    anchor = real
                    self.videoFeatures = FeatureSource(values: real, count: input.mergedTokens, tokens: self.video.tokens)
                    return nil
                }
                guard let anchor else { throw Self.missingReference("f\(anchorBucket)") }
                return try Self.consistency(real, anchor)
            }
        }
    }

    // MARK: Audio

    private func verifyAudio() throws {
        let buckets = AudioCoreMLEncoderMasked.frameBuckets.sorted()
        guard let anchorBucket = buckets.first else { return }
        // 1.9 s = 190 mel frames: inside the smallest bucket, with a partial boundary chunk.
        let samples = OmniSmallVerificationInputs.audio(sampleCount: 30_400)
        var anchor: [Float]?

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "audio", function: "f\(bucket)",
                units: audio.encoder.computeUnits,
                reference: isAnchor ? nil : "f\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.audioBucket
            ) { timing in
                let input: PreparedAudioInputs
                do {
                    input = try self.audio.preparer.prepare(samples, prompt: .document, bucketFrames: bucket)
                } catch {
                    throw Failure(reason: .runFailed("cannot build verification input: \(error)"))
                }
                _ = try timing.load { try self.audio.encoder.model(bucket) }
                var full = try timing.run {
                    try self.audio.encoder.encode(
                        packedMel: input.packedMel, nMels: self.audio.mel.nMels, masks: input.masks)
                }
                self.faults?("audio.f\(bucket)", &full)
                try Self.requireFeatures(full, rows: (bucket / 200) * 50, width: self.audio.featureDim)
                let real = Array(full[0 ..< (input.tokenCount * self.audio.featureDim)])
                if isAnchor {
                    anchor = real
                    self.audioFeatures = FeatureSource(values: real, count: input.tokenCount, tokens: self.audio.tokens)
                    return nil
                }
                guard let anchor else { throw Self.missingReference("f\(anchorBucket)") }
                return try Self.consistency(real, anchor)
            }
        }
    }

    // MARK: Embed and decoder

    private func verifyDecoder() throws {
        let buckets = decoder.buckets.sorted()
        guard let anchorBucket = buckets.first else { return }
        let feature = videoFeatures ?? audioFeatures ?? imageFeatures
        // Use at most 100 real features so the sequence fits the smallest bucket
        // (prefix 7 + 100 + suffix 3 = 110 <= 128).
        let sequence: (ids: [Int32], features: [Float], offset: Int)? = feature.map { media in
            let count = min(media.count, 100)
            return (
                media.tokens.ids(prompt: .document, count: count),
                Array(media.values.prefix(count * decoder.featDim)),
                media.tokens.resolvedPrefix(for: .document).count)
        }
        var embedAnchor: [Float]?
        var decoderAnchor: [Float]?
        var embedded: [Int: [Float]] = [:]

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "embed", function: "f\(bucket)",
                units: decoder.units(forS: bucket),
                reference: isAnchor ? nil : "f\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.embedRows
            ) { timing in
                guard let sequence else {
                    throw Failure(reason: .runFailed("no encoder output was available to build a media sequence"))
                }
                guard sequence.ids.count <= anchorBucket else {
                    throw Failure(reason: .invalidOutput(
                        "verification sequence is \(sequence.ids.count) tokens; smallest bucket is \(anchorBucket)"))
                }
                _ = try timing.load { try self.decoder.embedModel(bucket) }
                var full = try timing.run { try self.decoder.embedTokens(sequence.ids, bucket: bucket) }
                self.faults?("embed.f\(bucket)", &full)
                try Self.requireFeatures(full, rows: bucket, width: self.decoder.featDim)
                embedded[bucket] = full
                let real = Array(full[0 ..< (sequence.ids.count * self.decoder.featDim)])
                if isAnchor {
                    embedAnchor = real
                    return nil
                }
                guard let embedAnchor else { throw Self.missingReference("f\(anchorBucket)") }
                return try Self.rowConsistency(real, embedAnchor, width: self.decoder.featDim)
            }
        }

        for bucket in buckets {
            let isAnchor = bucket == anchorBucket
            try check(
                model: "decoder", function: "f\(bucket)",
                units: decoder.units(forS: bucket),
                reference: isAnchor ? nil : "f\(anchorBucket)",
                threshold: isAnchor ? nil : OmniSmallVerificationThresholds.decoderBucket
            ) { timing in
                guard let sequence else {
                    throw Failure(reason: .runFailed("no encoder output was available to build a media sequence"))
                }
                guard let embeddedRows = embedded[bucket] else {
                    throw Failure(reason: .runFailed("embed.f\(bucket) produced no output to decode"))
                }
                _ = try timing.load { try self.decoder.decoderModel(bucket) }
                var vector = try timing.run {
                    try self.decoder.decodeEmbedded(
                        embeddedRows, tokenCount: sequence.ids.count, features: sequence.features,
                        scatterOffset: sequence.offset, bucket: bucket)
                }
                self.faults?("decoder.f\(bucket)", &vector)
                try self.requireEmbedding(vector)
                if isAnchor {
                    decoderAnchor = vector
                    return nil
                }
                guard let decoderAnchor else { throw Self.missingReference("f\(anchorBucket)") }
                return try Self.consistency(vector, decoderAnchor)
            }
        }
    }

    // MARK: Recording

    /// Accumulates load and run time for one function and converts thrown errors to typed reasons.
    struct Timing {
        var loadMilliseconds = 0.0
        var runMilliseconds = 0.0

        mutating func load<T>(_ body: () throws -> T) throws -> T {
            let started = ContinuousClock.now
            defer { loadMilliseconds += verificationMilliseconds(ContinuousClock.now - started) }
            do {
                return try body()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw Failure(reason: .loadFailed(String(describing: error)))
            }
        }

        mutating func run<T>(_ body: () throws -> T) throws -> T {
            let started = ContinuousClock.now
            defer { runMilliseconds += verificationMilliseconds(ContinuousClock.now - started) }
            do {
                return try body()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw Failure(reason: .runFailed(String(describing: error)))
            }
        }
    }

    /// Run one function's check and record the outcome; never throws except for cancellation, so
    /// one bad function cannot hide the rest.
    private func check(
        model: String,
        function: String,
        units: MLComputeUnits?,
        reference: String?,
        threshold: Double?,
        _ body: (inout Timing) throws -> Double?
    ) throws {
        try Task.checkCancellation()
        var timing = Timing()
        var cosineValue: Double?
        var reason: OmniSmallVerificationError.Reason?
        do {
            cosineValue = try body(&timing)
            if let cosineValue, let threshold, let reference {
                // Written so that NaN (a zero vector) fails.
                if !(cosineValue >= threshold) {
                    reason = .inconsistent(cosine: cosineValue, threshold: threshold, reference: reference)
                }
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch let failure as Failure {
            reason = failure.reason
        } catch {
            reason = .runFailed(String(describing: error))
        }
        let entry = OmniSmallFunctionCheck(
            model: model,
            function: function,
            computeUnits: units.map(Self.describe),
            loadMilliseconds: timing.loadMilliseconds,
            runMilliseconds: timing.runMilliseconds,
            referenceFunction: reference,
            minimumCosine: cosineValue,
            cosineThreshold: threshold,
            passed: reason == nil,
            failure: reason.map { String(describing: $0) })
        checks.append(entry)
        if let reason, firstFailure == nil {
            firstFailure = (entry.name, reason)
        }
        progress?(entry)
    }

    // MARK: Output checks

    /// An embedding must have the native width, be finite, and be unit norm — the same validator
    /// every served vector goes through.
    private func requireEmbedding(_ values: [Float]) throws {
        do {
            try validateEmbedding(values)
        } catch {
            throw Failure(reason: .invalidOutput(String(describing: error)))
        }
    }

    /// An encoder feature matrix must have exactly `rows * width` values, all finite.
    private static func requireFeatures(_ values: [Float], rows: Int, width: Int) throws {
        guard values.count == rows * width else {
            throw Failure(reason: .invalidOutput(
                "expected \(rows) × \(width) = \(rows * width) values, found \(values.count)"))
        }
        guard values.allSatisfy(\.isFinite) else {
            throw Failure(reason: .invalidOutput("output contains non-finite values"))
        }
    }

    /// Cosine between a function's output and its reference's; the lengths must match.
    private static func consistency(_ output: [Float], _ reference: [Float]) throws -> Double {
        guard output.count == reference.count else {
            throw Failure(reason: .invalidOutput(
                "output has \(output.count) values but its reference has \(reference.count)"))
        }
        return cosine(output, reference)
    }

    /// The smaller of two cosines, where NaN (an undefined cosine — a zero vector) is contagious.
    /// `min` alone would drop it: every comparison with NaN is false, so `min(x, .nan)` returns `x`
    /// and a zero row would silently pass.
    private static func lowest(_ a: Double, _ b: Double) -> Double {
        if a.isNaN || b.isNaN { return .nan }
        return min(a, b)
    }

    /// The lowest per-row cosine of two row-major matrices; the lengths must match.
    private static func rowConsistency(_ output: [Float], _ reference: [Float], width: Int) throws -> Double {
        guard output.count == reference.count else {
            throw Failure(reason: .invalidOutput(
                "output has \(output.count) values but its reference has \(reference.count)"))
        }
        return minimumRowCosine(output, reference, width: width)
    }

    private static func missingReference(_ name: String) -> Failure {
        Failure(reason: .invalidOutput("reference function \(name) produced no usable output to compare against"))
    }

    private static func input(_ prepared: Result<PreparedVisionInputs, any Error>) throws -> PreparedVisionInputs {
        do {
            return try prepared.get()
        } catch {
            throw Failure(reason: .runFailed("cannot build verification input: \(error)"))
        }
    }

    /// The lowest per-row cosine between two row-major matrices of `width` columns.
    static func minimumRowCosine(_ a: [Float], _ b: [Float], width: Int) -> Double {
        guard width > 0, !a.isEmpty, a.count == b.count, a.count.isMultiple(of: width) else { return .nan }
        var worst = Double.infinity
        var start = 0
        while start < a.count {
            let end = start + width
            worst = lowest(worst, cosine(Array(a[start..<end]), Array(b[start..<end])))
            start = end
        }
        return worst
    }

    private static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: "cpu"
        case .cpuAndGPU: "cpu+gpu"
        case .cpuAndNeuralEngine: "cpu+ane"
        case .all: "all"
        @unknown default: "unknown"
        }
    }
}

/// Milliseconds in a `Duration`.
private func verificationMilliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1_000
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000
}
