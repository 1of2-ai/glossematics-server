import AVFoundation
import CoreML
import Foundation

/// Extract bounded video frames for the native video ViT. The serving path follows the source
/// processor's 2 fps / minimum-four / uniform-frame-index sampling, capped by the converted
/// tower's frame and patch budgets. AVFoundation timestamps and CoreGraphics resizing are not
/// bit-identical to the reference decoder and bicubic resize.
internal enum GlossVideoFile {
    public enum DecodeError: Error, CustomStringConvertible {
        case noVideoTrack
        case oddFrameCount(Int)
        case invalidDuration
        case invalidFrameRate
        case oversizedFrame
        case frameFailed(Int)

        public var description: String {
            switch self {
            case .noVideoTrack: "file has no readable video track"
            case let .oddFrameCount(count): "frame count \(count) must be even and between 2 and 32"
            case .invalidDuration: "video duration must be greater than zero and no more than 60 seconds"
            case .invalidFrameRate: "video frame rate is invalid"
            case .oversizedFrame: "source frames must be no larger than 4096 × 4096 or 16 megapixels"
            case let .frameFailed(index): "video frame \(index) could not be decoded"
            }
        }
    }

    static let maximumDurationSeconds = 60.0
    /// Per-frame source pixel cap, shared with the decoder (see `VideoFrameDecoder.maximumSourceEdge`).
    static let maximumSourcePixels = VideoFrameDecoder.maximumSourcePixels
    static let maximumSampledFrames = 32
    static let samplingFPS = 2.0
    static let referenceMinimumPixels = 262_144
    static let referenceMaximumPixels = 12_845_056

    struct FramePlan {
        let frameIndices: [Int]
        let height: Int
        let width: Int
        let sourceFPS: Double
        let patches: Int
    }

    /// Source `sample_frames` uses `int(total_frames / source_fps * 2)` and rounded linspace
    /// indices over the decoded frames. The converted ViT holds at most 2048 patches, so this
    /// serving profile limits temporal samples to 32 and spends the remaining patch budget on
    /// aspect-preserving pixels (the processor's `longest_edge` budget over t·h·w).
    static func plan(totalFrames: Int, sourceFPS: Double, height: Int, width: Int,
                     maxPatches: Int, preprocessor: GlossImagePreprocessor) throws -> FramePlan {
        guard sourceFPS.isFinite, sourceFPS > 0, sourceFPS <= 1_000 else {
            throw DecodeError.invalidFrameRate
        }
        let duration = Double(totalFrames) / sourceFPS
        guard totalFrames > 0, duration <= maximumDurationSeconds else {
            throw DecodeError.invalidDuration
        }
        let sourcePixels = height.multipliedReportingOverflow(by: width)
        guard height > 0, width > 0,
              height <= VideoFrameDecoder.maximumSourceEdge, width <= VideoFrameDecoder.maximumSourceEdge,
              !sourcePixels.overflow, sourcePixels.partialValue <= maximumSourcePixels else {
            throw DecodeError.oversizedFrame
        }
        let requested = Int(Double(totalFrames) / sourceFPS * samplingFPS)
        let sampled = min(max(requested, 4), maximumSampledFrames, totalFrames)
        let indices: [Int]
        if sampled == 1 {
            indices = [0]
        } else {
            indices = (0..<sampled).map {
                Int((Double($0) * Double(totalFrames - 1) / Double(sampled - 1)).rounded(.toNearestOrEven))
            }
        }
        // Qwen's temporal patchification repeats the last frame for an odd sample count.
        let paddedIndices = sampled.isMultiple(of: preprocessor.temporal) ? indices : indices + [indices.last!]
        // The processor's budget is over t·h·w pixels; the converted tower caps it at
        // maxPatches·patch²·temporal. For odd sample counts the processor's own rounding can still
        // exceed the tower, so the budget shrinks until the padded grid fits (a forced deviation).
        let factor = preprocessor.factor
        let towerPixels = maxPatches * preprocessor.patch * preprocessor.patch * preprocessor.temporal
        var budget = min(referenceMaximumPixels, towerPixels)
        var resizedH = 0, resizedW = 0, patches = Int.max
        while patches > maxPatches, budget >= factor * factor {
            (resizedH, resizedW) = try videoSmartResize(
                frames: sampled, height: height, width: width, factor: factor,
                temporal: preprocessor.temporal, minPixels: referenceMinimumPixels, maxPixels: budget)
            patches = (paddedIndices.count / preprocessor.temporal)
                * (resizedH / preprocessor.patch) * (resizedW / preprocessor.patch)
            if patches > maxPatches { budget = budget * maxPatches / patches - 1 }
        }
        guard patches > 0, patches <= maxPatches else {
            throw VideoCoreMLEncoderMasked.EncoderError.invalidInput("video exceeds the native patch grid")
        }
        return FramePlan(frameIndices: paddedIndices, height: resizedH, width: resizedW,
                         sourceFPS: sourceFPS, patches: patches)
    }

    /// `Qwen3VLVideoProcessor`'s `smart_resize` exactly: Python `round` (half to even) for the
    /// factor grid and the temporal count, the pixel bounds applied to `t_bar·h·w`, and the rescale
    /// computed from the unpadded frame count.
    static func videoSmartResize(frames n: Int, height: Int, width: Int, factor: Int, temporal: Int,
                                 minPixels: Int, maxPixels: Int) throws -> (Int, Int) {
        var h = Double(height), w = Double(width)
        let f = Double(factor)
        if h < f || w < f {
            let scale = max(f / h, f / w)
            h = Double(Int(h * scale)); w = Double(Int(w * scale))
        }
        guard max(h, w) / min(h, w) <= 200 else {
            throw DecodeError.oversizedFrame
        }
        var hBar = (h / f).rounded(.toNearestOrEven) * f
        var wBar = (w / f).rounded(.toNearestOrEven) * f
        let tBar = (Double(n) / Double(temporal)).rounded(.toNearestOrEven) * Double(temporal)
        if tBar * hBar * wBar > Double(maxPixels) {
            let beta = (Double(n) * h * w / Double(maxPixels)).squareRoot()
            hBar = max(f, (h / beta / f).rounded(.down) * f)
            wBar = max(f, (w / beta / f).rounded(.down) * f)
        } else if tBar * hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / (Double(n) * h * w)).squareRoot()
            hBar = (h * beta / f).rounded(.up) * f
            wBar = (w * beta / f).rounded(.up) * f
        }
        return (Int(hBar), Int(wBar))
    }

    /// Decode, sample, and resize a video file the way the source processor sees it; see
    /// `VideoFrameDecoder` for the decode, color, and resize contract.
    ///
    /// `VideoFrameDecoder.open` enforces the per-frame source cap (and rejects a zero, negative, or
    /// non-finite declared size) before this decodes anything. `checkCancellation` is polled through
    /// the frame scan and decode so cancelled work stops early.
    static func extractFrames(_ url: URL, maxPatches: Int,
                              preprocessor: GlossImagePreprocessor,
                              checkCancellation: (() throws -> Void)? = nil) throws -> (frames: [[UInt8]], h: Int, w: Int) {
        let source = try VideoFrameDecoder.open(url)
        let total = try VideoFrameDecoder.frameCount(source, checkCancellation: checkCancellation)
        let fps = VideoFrameDecoder.averageFrameRate(source, frames: total)
        guard fps.isFinite, fps > 0 else { throw DecodeError.invalidFrameRate }
        let framePlan = try plan(totalFrames: total, sourceFPS: fps,
                                 height: source.height, width: source.width,
                                 maxPatches: maxPatches, preprocessor: preprocessor)
        let frames = try VideoFrameDecoder.frames(source, indices: framePlan.frameIndices,
                                                  width: framePlan.width, height: framePlan.height,
                                                  checkCancellation: checkCancellation)
        return (frames, framePlan.height, framePlan.width)
    }

    /// Explicit fixed-profile extraction retained for direct model/parity tests.
    public static func extractSquareFrames(_ url: URL, count: Int, size: Int,
                                           preprocessor: GlossImagePreprocessor) throws -> [[UInt8]] {
        guard count > 0, count <= 32, count % 2 == 0 else { throw DecodeError.oddFrameCount(count) }
        guard size >= 32, size <= 512, size.isMultiple(of: preprocessor.factor) else {
            throw GlossImagePreprocessor.ImageError.invalidGeometry("video frame size must be a 32-aligned value from 32 to 512")
        }
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw DecodeError.noVideoTrack
        }
        let source = track.naturalSize
        guard source.width.isFinite, source.height.isFinite,
              source.width > 0, source.height > 0,
              source.width <= 4_096, source.height <= 4_096,
              source.width * source.height <= Double(maximumSourcePixels) else {
            throw DecodeError.oversizedFrame
        }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: size, height: size)
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        let dur = asset.duration   // sync (deprecated, like copyCGImage below) — keeps this helper sync
        let seconds = dur.seconds
        guard seconds.isFinite, seconds > 0, seconds <= maximumDurationSeconds else {
            throw DecodeError.invalidDuration
        }
        let frameStep = track.minFrameDuration
        let frameSeconds = frameStep.seconds
        var frames = [[UInt8]]()
        frames.reserveCapacity(count)
        for i in 0..<count {
            let midpoint = seconds * (Double(i) + 0.5) / Double(count)
            let t: CMTime
            if frameSeconds.isFinite, frameSeconds > 0,
               frameSeconds <= 1,
               midpoint / frameSeconds < Double(Int32.max) {
                let frameIndex = Int32((midpoint / frameSeconds).rounded(.down))
                t = CMTimeMultiply(frameStep, multiplier: frameIndex)
            } else {
                t = CMTime(seconds: midpoint, preferredTimescale: 600)
            }
            do {
                let cg = try gen.copyCGImage(at: t, actualTime: nil)
                frames.append(try preprocessor.resizedRGB(cg, w: size, h: size))
            } catch { throw DecodeError.frameFailed(i) }
        }
        return frames
    }
}

/// Host-side position computation for the runtime-position ViT — the Swift port of Qwen3VL's
/// `get_vision_position_ids` + `get_vision_bilinear_indices_and_weights` + vision RoPE. For a patch
/// grid (gh, gw) it produces, in the merger's 2×2-block order, the bilinear-interpolated `posEmbeds`
/// (N·hidden) and the rotary `cos`/`sin` (N·ropeDim), N = gh·gw. These feed the converted masked ViT
/// (which also takes pixel_values + an attn_bias masking padding patches).
internal struct VisionPositions {
    public let numGridPerSide: Int   // 48 (sqrt of the learned pos_embed table rows)
    public let hidden: Int           // 1024
    public let mergeSize: Int        // 2
    public let patchSize: Int        // 16
    let posTable: [Float]            // (numGridPerSide^2 * hidden) row-major
    let invFreq: [Float]             // (rope_inv_freq_len,) e.g. 16
    /// rope dim = 2 positional axes × invFreq × 2 (the cat([rot,rot]) doubling) = 64.
    public var ropeDim: Int { invFreq.count * 4 }

    public struct Meta: Decodable {
        public let num_grid_per_side: Int; public let hidden: Int; public let spatial_merge_size: Int
        public let patch_size: Int; public let rope_theta: Double
        public let pos_table_rows: Int; public let rope_inv_freq_len: Int
    }

    public enum PosError: Error { case badTable }

    public init(metaURL: URL, posTableURL: URL, invFreqURL: URL) throws {
        let m = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        guard m.num_grid_per_side == 48, m.hidden == 1_024,
              m.spatial_merge_size == 2, m.patch_size == 16,
              m.rope_theta == 10_000,
              m.pos_table_rows == 2_304, m.rope_inv_freq_len == 16 else {
            throw PosError.badTable
        }
        numGridPerSide = m.num_grid_per_side; hidden = m.hidden
        mergeSize = m.spatial_merge_size; patchSize = m.patch_size
        posTable = try Self.loadF32(posTableURL, count: 2_304 * 1_024)
        invFreq = try Self.loadF32(invFreqURL, count: 16)
    }

    static func loadF32(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == count * MemoryLayout<Float>.size else { throw PosError.badTable }
        var values = [Float](repeating: 0, count: count)
        _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        guard values.allSatisfy(\.isFinite) else { throw PosError.badTable }
        return values
    }

    /// linspace(0, side-1, n) matching torch.linspace.
    private func linspace(_ n: Int) -> [Float] {
        if n <= 1 { return [0] }
        let step = Float(numGridPerSide - 1) / Float(n - 1)
        return (0..<n).map { Float($0) * step }
    }

    /// Returns merge-ordered (posEmbeds, cos, sin) and the merged-token count for grid (gh,gw).
    public func compute(gh: Int, gw: Int) -> (posEmbeds: [Float], cos: [Float], sin: [Float], merged: Int) {
        let m = mergeSize, side = numGridPerSide, N = gh * gw, D = ropeDim, K = invFreq.count
        let hGrid = linspace(gh), wGrid = linspace(gw)
        var posEmbeds = [Float](repeating: 0, count: N * hidden)
        var cosA = [Float](repeating: 0, count: N * D)
        var sinA = [Float](repeating: 0, count: N * D)
        posTable.withUnsafeBufferPointer { tbl in
            var t = 0
            for bi in 0..<(gh / m) {
                for bj in 0..<(gw / m) {
                    for pi in 0..<m {
                        for pj in 0..<m {
                            let hp = bi * m + pi, wp = bj * m + pj
                            // bilinear interpolation of the pos_embed table at (hGrid[hp], wGrid[wp])
                            let hg = hGrid[hp], wg = wGrid[wp]
                            let hf = Int(hg), wf = Int(wg)
                            let hc = Swift.min(hf + 1, side - 1), wc = Swift.min(wf + 1, side - 1)
                            let hfr = hg - Float(hf), wfr = wg - Float(wf)
                            let w0 = (1 - hfr) * (1 - wfr), w1 = (1 - hfr) * wfr
                            let w2 = hfr * (1 - wfr), w3 = hfr * wfr
                            let c0 = (hf * side + wf) * hidden, c1 = (hf * side + wc) * hidden
                            let c2 = (hc * side + wf) * hidden, c3 = (hc * side + wc) * hidden
                            let base = t * hidden
                            for d in 0..<hidden {
                                posEmbeds[base + d] = w0 * tbl[c0 + d] + w1 * tbl[c1 + d]
                                    + w2 * tbl[c2 + d] + w3 * tbl[c3 + d]
                            }
                            // RoPE: rot = [hp·invF (K), wp·invF (K)]; emb = [rot, rot] (2K each axis)
                            let rb = t * D, half = 2 * K
                            for k in 0..<K {
                                let fh = Float(hp) * invFreq[k], fw = Float(wp) * invFreq[k]
                                let ch = cosf(fh), cw = cosf(fw), sh = sinf(fh), sw = sinf(fw)
                                cosA[rb + k] = ch; cosA[rb + K + k] = cw
                                cosA[rb + half + k] = ch; cosA[rb + half + K + k] = cw
                                sinA[rb + k] = sh; sinA[rb + K + k] = sw
                                sinA[rb + half + k] = sh; sinA[rb + half + K + k] = sw
                            }
                            t += 1
                        }
                    }
                }
            }
        }
        return (posEmbeds, cosA, sinA, (gh / m) * (gw / m))
    }

    /// VIDEO: `t` frames share the same spatial positions (no temporal RoPE — verified), so tile the
    /// single-frame block `t` times. Returns the tiled (posEmbeds, cos, sin) and merged = t·(gh/2)(gw/2).
    public func computeVideo(t: Int, gh: Int, gw: Int) -> (posEmbeds: [Float], cos: [Float], sin: [Float], merged: Int) {
        let (pe, cv, sv, merged1) = compute(gh: gh, gw: gw)
        func tile(_ a: [Float]) -> [Float] { var o = [Float](); o.reserveCapacity(a.count * t); for _ in 0..<t { o.append(contentsOf: a) }; return o }
        return (tile(pe), tile(cv), tile(sv), merged1 * t)
    }
}

/// Runtime-position masked ViT (`vision_tower_masked_multifunc`): per N_max patch-bucket f{N} taking
/// pixel_values (N,1536) + pos_embeds (N,hidden) + rope cos/sin (N,ropeDim) + attn_bias (1,1,1,N).
/// The host pads the real L patches to N and masks the padding; output is the full-layout merged
/// features (N/4, 1024), truncated by the caller to the real merged count. Runs on GPU (fp16 accum).
internal final class VisionCoreMLEncoderMasked {
    /// Conversion-pipeline bucket convention; bundles override via `manifest.image.patchBuckets`.
    public static let defaultPatchBuckets = [1024, 1600, 2304, 3072, 4032]
    public static let neg: Float = -1e4
    public let patchBuckets: [Int]
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    /// Per-bucket functions, each loaded once on first use without blocking other buckets.
    private let models = KeyedLoadCache<Int, MLModel>()

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU,
                patchBuckets: [Int] = VisionCoreMLEncoderMasked.defaultPatchBuckets) throws {
        guard !patchBuckets.isEmpty,
              patchBuckets.allSatisfy({ $0 > 0 && $0 <= 5_120 && $0.isMultiple(of: 4) }) else {
            throw EncoderError.invalidInput("image patch buckets must be positive native multiples of four")
        }
        self.computeUnits = computeUnits
        self.patchBuckets = patchBuckets.sorted()
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput, invalidInput(String) }

    public func bucket(forPatches L: Int) -> Int { patchBuckets.first { $0 >= L } ?? patchBuckets.last! }

    /// The `f{N}` function for a patch bucket, loaded on first use.
    func model(_ N: Int) throws -> MLModel {
        try models.value(for: N) {
            let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(N)"
            return try MLModel(contentsOf: compiledURL, configuration: cfg)
        }
    }

    /// `pixelValues` = (L·pixelDim), `posEmbeds` = (L·hidden), `cos`/`sin` = (L·ropeDim). Returns the
    /// full-layout merged features (N/4 · 1024); caller keeps the first `merged` tokens.
    /// `forcedBucket` runs the same patches through a specific declared bucket instead of the
    /// smallest that fits (startup verification compares every bucket on one image).
    public func encode(pixelValues: [Float], pixelDim: Int, posEmbeds: [Float], hidden: Int,
                       cos: [Float], sin: [Float], ropeDim: Int, patches L: Int,
                       bucket forcedBucket: Int? = nil) throws -> [Float] {
        let N = forcedBucket ?? bucket(forPatches: L)
        guard L > 0, L <= N, forcedBucket == nil || patchBuckets.contains(N) else {
            throw EncoderError.invalidInput("image exceeds the largest patch bucket")
        }
        guard pixelDim == 1_536, hidden == 1_024, ropeDim == 64,
              pixelValues.count == L * pixelDim,
              posEmbeds.count == L * hidden,
              cos.count == L * ropeDim,
              sin.count == L * ropeDim else {
            throw EncoderError.invalidInput("image tensors do not match the native patch geometry")
        }
        let pv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: pixelDim)], dataType: .float32)
        let pe = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: hidden)], dataType: .float32)
        let cv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let sv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        // Vision attention is full -> a (1,1,1,N) key-padding mask (N floats, not N²) suffices.
        let ab = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: N)], dataType: .float32)
        // MLMultiArray is not zero-initialized; clear padding before filling real patch rows.
        try CoreMLArrayReader.fillFloat32(pv, with: pixelValues, label: "image pixels")
        try CoreMLArrayReader.fillFloat32(pe, with: posEmbeds, label: "image positions")
        try CoreMLArrayReader.fillFloat32(cv, with: cos, label: "image rope cosine")
        try CoreMLArrayReader.fillFloat32(sv, with: sin, label: "image rope sine")
        let abp = ab.dataPointer.bindMemory(to: Float.self, capacity: N)
        for j in 0..<N { abp[j] = j < L ? 0.0 : Self.neg }   // mask padding keys
        let out = try model(N).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": pv, "pos_embeds": pe, "rope_cos": cv, "rope_sin": sv, "attn_bias": ab,
        ]))
        guard let f = out.featureValue(for: "vision_features")?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try CoreMLArrayReader.float32(
            f, shape: [N / 4, hidden], label: "image encoder features")
    }
}

/// Encoder-ready tensors for one image or video: everything the vision tower and the shared decoder
/// consume, computed on the CPU with no Core ML involvement. Splitting this out of the embedders is
/// what lets the daemon do image decode, video decode, resize, patchify, and position math on a
/// bounded CPU executor while the single accelerator lane stays free for inference. Because
/// `embed(...)` is literally `infer(prepare(...))`, the split path and the direct path feed Core ML
/// the same arrays by construction.
internal struct PreparedVisionInputs: Sendable {
    /// Patch rows fed to the ViT: `gh*gw` for an image, `t*gh*gw` for a video.
    let patches: Int
    /// (patches × 1536) processor-normalized patch pixels in the merger's 2×2-block order.
    let pixelValues: [Float]
    /// (patches × hidden) bilinear-interpolated learned positions.
    let posEmbeds: [Float]
    /// (patches × ropeDim) rotary tables.
    let ropeCos: [Float]
    let ropeSin: [Float]
    /// Temporal groups and patches per group: `(1, patches)` for an image; `(t, gh*gw)` for a video,
    /// whose attention is block-diagonal per group.
    let temporalGroups: Int
    let groupPatches: Int
    /// Real merged tokens (`patches / 4`) the decoder receives.
    let mergedTokens: Int
    /// `prefix + placeholder × mergedTokens + suffix` for the requested retrieval role.
    let tokenIDs: [Int32]
    /// Where the merged features are scattered into `tokenIDs` (the role's prefix length).
    let scatterOffset: Int
}

/// Host-side image preparation: bounded decode, smart-resize, patchify, positions, and prompt ids.
/// A pure `Sendable` value with no Core ML state, so it can run on any thread.
internal struct ImageInputPreparer: Sendable {
    let preprocessor: GlossImagePreprocessor
    let positions: VisionPositions
    let tokens: MediaTokens
    /// Largest converted patch bucket; larger images are downscaled to fit it.
    let maxPatches: Int

    /// Image file -> encoder inputs for ANY resolution: smart-resize to the model grid (capped to
    /// the largest converted bucket so big images downscale gracefully instead of being unsupported),
    /// patchify. Non-factor-aligned native sizes carry the documented CoreGraphics resample caveat.
    func prepare(imageURL: URL, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        try prepare(cgImage: try GlossImagePreprocessor.loadCGImage(imageURL), prompt: prompt)
    }

    /// In-memory image data -> encoder inputs (no temporary file).
    func prepare(imageData: Data, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        try prepare(cgImage: try GlossImagePreprocessor.loadCGImage(imageData), prompt: prompt)
    }

    func prepare(cgImage: CGImage, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        let (hbar, wbar) = preprocessor.smartResize(
            h: cgImage.height,
            w: cgImage.width,
            maxPixelsOverride: maxPatches * 256)
        let rgb = try preprocessor.resizedRGB(cgImage, w: wbar, h: hbar)
        return try prepare(rgb: rgb, h: hbar, w: wbar, prompt: prompt)
    }

    /// Raw RGB (h*w*3, h/w factor-aligned) -> encoder inputs. Exact (no resample).
    /// Requires (h/16)*(w/16) ≤ maxPatches.
    func prepare(rgb: [UInt8], h: Int, w: Int, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        let (pv, gh, gw) = try preprocessor.pixelValues(rgb: rgb, h: h, w: w)
        guard gh * gw <= maxPatches else {
            throw VisionCoreMLEncoderMasked.EncoderError.invalidInput("image exceeds the largest patch bucket")
        }
        return try prepare(pixelValues: pv, gh: gh, gw: gw, prompt: prompt)
    }

    /// `pixelValues` = (gh·gw · pixelDim) row-major (processor output).
    func prepare(pixelValues: [Float], gh: Int, gw: Int, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        let product = gh.multipliedReportingOverflow(by: gw)
        guard gh > 0, gw > 0,
              gh.isMultiple(of: positions.mergeSize),
              gw.isMultiple(of: positions.mergeSize),
              !product.overflow, product.partialValue <= maxPatches,
              pixelValues.count == product.partialValue * preprocessor.featuresPerPatch,
              pixelValues.allSatisfy(\.isFinite) else {
            throw VisionCoreMLEncoderMasked.EncoderError.invalidInput("image pixels or grid are invalid")
        }
        let L = product.partialValue
        let (pe, cosv, sinv, merged) = positions.compute(gh: gh, gw: gw)
        return PreparedVisionInputs(
            patches: L, pixelValues: pixelValues, posEmbeds: pe, ropeCos: cosv, ropeSin: sinv,
            temporalGroups: 1, groupPatches: L, mergedTokens: merged,
            tokenIDs: tokens.ids(prompt: prompt, count: merged),
            scatterOffset: tokens.resolvedPrefix(for: prompt).count)
    }
}

/// True arbitrary-resolution image embedder: runtime-position masked ViT + the unified general
/// decoder. Given pixel_values (processor output) for grid (gh,gw), produces the L2-normalized
/// embedding. The URL/data methods perform bounded decode, smart-resize, and patchify;
/// `embed(pixelValues:gh:gw:)` also permits independent tensor-path verification.
///
/// Every `embed(...)` is `infer(preparer.prepare(...))`: the CPU half lives in ``ImageInputPreparer``
/// (usable off the accelerator) and this class owns only the Core ML half.
internal final class GlossImageEmbedderMasked {
    /// The image prompt wrapper (`prefix + placeholder × merged + suffix`). Bundles supply the
    /// manifest's resolved ids; direct construction passes a ``MediaTokens`` preset.
    public let tokens: MediaTokens
    /// Media feature width == the model's embedding dimension (merger projects into the shared space).
    public let featureDim: Int

    public let positions: VisionPositions
    public let encoder: VisionCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder
    public let preprocessor: GlossImagePreprocessor
    /// The CPU half of the pipeline, safe to use from any thread.
    public let preparer: ImageInputPreparer

    /// Assembles an embedder from already-built parts. The daemon uses this to share ONE
    /// `GeneralMediaDecoder` (and its loaded functions) across the image, audio, and video pipelines.
    public init(visionModelURL: URL, positions: VisionPositions, decoder: GeneralMediaDecoder,
                tokens: MediaTokens, featureDim: Int = 1024,
                patchBuckets: [Int] = VisionCoreMLEncoderMasked.defaultPatchBuckets,
                preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                encoderUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.tokens = tokens
        self.featureDim = featureDim
        self.preprocessor = preprocessor
        self.positions = positions
        self.decoder = decoder
        encoder = try VisionCoreMLEncoderMasked(modelURL: visionModelURL, computeUnits: encoderUnits, patchBuckets: patchBuckets)
        preparer = ImageInputPreparer(
            preprocessor: preprocessor, positions: positions, tokens: tokens,
            maxPatches: encoder.patchBuckets.last ?? 0)
    }

    /// `encoderUnits`: `.cpuAndGPU` (default) is the recommended choice — BOTH more accurate (fp32
    /// accumulation ~0.99996 vs the ANE's fp16 ~0.9995 end-to-end) AND much faster (measured 219ms vs
    /// 496ms for 512²; the (1,1,1,N) key-mask + masked-SDPA are not ANE-friendly). `.cpuAndNeuralEngine`
    /// runs the ViT on the ANE too (full-ANE) but is slower and less accurate — for GPU-contended cases only.
    /// `decoderUnits`: `nil` (default) = adaptive placement (ANE for S≤256, GPU for S≥512). Pass
    /// `.cpuAndNeuralEngine` together with `encoderUnits: .cpuAndNeuralEngine` for a TRUE full-ANE
    /// deployment (encoder + decoder both on the ANE) — measured end-to-end cos 0.999495, above the
    /// model's bf16 floor (lowest-power, GPU-free; slower than the hybrid default).
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            metaURL: URL, posTableURL: URL, invFreqURL: URL,
                            tokens: MediaTokens, featureDim: Int = 1024,
                            patchBuckets: [Int] = VisionCoreMLEncoderMasked.defaultPatchBuckets,
                            padTokenID: Int32 = 0,
                            preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                            encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil,
                            sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        try self.init(
            visionModelURL: visionModelURL,
            positions: try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: invFreqURL),
            decoder: try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                                             computeUnits: decoderUnits, featDim: featureDim,
                                             padTokenID: padTokenID, sequenceBuckets: sequenceBuckets),
            tokens: tokens, featureDim: featureDim, patchBuckets: patchBuckets,
            preprocessor: preprocessor, encoderUnits: encoderUnits)
    }

    /// Convenience: the 3 host-side resources (`meta.json`, `pos_embed_table.f32`, `rope_inv_freq.f32`,
    /// produced by `export_vision_swift_refs.py`) are loaded by name from `resourcesDir`.
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            resourcesDir: URL, tokens: MediaTokens, featureDim: Int = 1024,
                            patchBuckets: [Int] = VisionCoreMLEncoderMasked.defaultPatchBuckets,
                            padTokenID: Int32 = 0,
                            preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                            encoderUnits: MLComputeUnits = .cpuAndGPU,
                            decoderUnits: MLComputeUnits? = nil,
                            sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        try self.init(visionModelURL: visionModelURL, embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                      metaURL: resourcesDir.appendingPathComponent("meta.json"),
                      posTableURL: resourcesDir.appendingPathComponent("pos_embed_table.f32"),
                      invFreqURL: resourcesDir.appendingPathComponent("rope_inv_freq.f32"),
                      tokens: tokens, featureDim: featureDim, patchBuckets: patchBuckets,
                      padTokenID: padTokenID, preprocessor: preprocessor,
                      encoderUnits: encoderUnits, decoderUnits: decoderUnits,
                      sequenceBuckets: sequenceBuckets)
    }

    /// The largest image (in mel-patch terms) the converted ViT buckets hold exactly.
    public var maxPatches: Int { preparer.maxPatches }

    /// Raw RGB (h*w*3, h/w factor-aligned) -> embedding. Exact (no resample) — the full host path.
    /// Requires (h/16)*(w/16) ≤ maxPatches; use `embed(imageURL:)` for arbitrary sizes (it downscales).
    /// `prompt` selects the retrieval side: the bundle's conditioned "Query: "/"Document: " wrapper
    /// (`.none` uses the plain wrapper — supported, but the model card's recipe conditions every
    /// modality for retrieval).
    public func embed(rgb: [UInt8], h: Int, w: Int, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(rgb: rgb, h: h, w: w, prompt: prompt), dim: dim)
    }

    /// Image file -> embedding for ANY resolution (see ``ImageInputPreparer/prepare(imageURL:prompt:)``).
    public func embed(imageURL: URL, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(imageURL: imageURL, prompt: prompt), dim: dim)
    }

    /// In-memory image data -> embedding. This avoids a temporary file for callers that receive
    /// image bytes over a local transport.
    public func embed(imageData: Data, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(imageData: imageData, prompt: prompt), dim: dim)
    }

    /// `pixelValues` = (gh·gw · pixelDim) row-major (processor output). Returns the embedding.
    public func embed(pixelValues: [Float], gh: Int, gw: Int, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(pixelValues: pixelValues, gh: gh, gw: gw, prompt: prompt), dim: dim)
    }

    /// The Core ML half: ViT prediction, merged-feature slice, scatter into the prompt, decoder
    /// prediction, optional Matryoshka truncation. Nothing here touches the CPU-side decode.
    public func infer(_ prepared: PreparedVisionInputs, dim: Int? = nil) throws -> [Float] {
        let full = try encoder.encode(pixelValues: prepared.pixelValues, pixelDim: preprocessor.featuresPerPatch,
                                      posEmbeds: prepared.posEmbeds, hidden: positions.hidden,
                                      cos: prepared.ropeCos, sin: prepared.ropeSin, ropeDim: positions.ropeDim,
                                      patches: prepared.patches)
        let used = Array(full[0 ..< (prepared.mergedTokens * featureDim)])
        let emb = try decoder.decode(tokenIds: prepared.tokenIDs, features: used,
                                     scatterOffset: prepared.scatterOffset)
        if let d = dim { return matryoshka(emb, dim: d) }
        return emb
    }
}

/// VIDEO ViT encoder (`vision_tower_video_multifunc`): per-N f{N} taking pixel_values (N,1536) +
/// pos_embeds + rope cos/sin + a dense (1,1,N,N) attn_bias that the host builds PER-FRAME block-
/// diagonal (each frame's gh·gw patches attend within the frame, cf. docs/VIDEO_PATH.md) + key-pad.
/// Output is the full-layout merged features (N/4, featDim); caller truncates to the real merged count.
internal final class VideoCoreMLEncoderMasked {
    /// Conversion-pipeline bucket convention; bundles override via `manifest.video.patchBuckets`.
    public static let defaultPatchBuckets = [256, 512, 1024, 2048]
    public static let neg: Float = -1e4
    public let patchBuckets: [Int]
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    /// Per-bucket functions, each loaded once on first use without blocking other buckets.
    private let models = KeyedLoadCache<Int, MLModel>()

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU,
                patchBuckets: [Int] = VideoCoreMLEncoderMasked.defaultPatchBuckets) throws {
        guard !patchBuckets.isEmpty,
              patchBuckets.allSatisfy({ $0 > 0 && $0 <= 2_048 && $0.isMultiple(of: 4) }) else {
            throw EncoderError.invalidInput("video patch buckets must be positive native multiples of four")
        }
        self.computeUnits = computeUnits
        self.patchBuckets = patchBuckets.sorted()
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput, tooLarge(Int), invalidInput(String) }
    public func bucket(forPatches L: Int) -> Int { patchBuckets.first { $0 >= L } ?? patchBuckets.last! }

    /// The `f{N}` function for a patch bucket, loaded on first use.
    func model(_ N: Int) throws -> MLModel {
        try models.value(for: N) {
            let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(N)"
            return try MLModel(contentsOf: compiledURL, configuration: cfg)
        }
    }

    /// `t` frames, `fp` = patches per frame (gh·gw); L = t·fp. Returns full-layout features (N/4·1024).
    /// `forcedBucket` runs the same patches through a specific declared bucket instead of the
    /// smallest that fits (startup verification compares every bucket on one clip).
    public func encode(pixelValues: [Float], pixelDim: Int, posEmbeds: [Float], hidden: Int,
                       cos: [Float], sin: [Float], ropeDim: Int, frames t: Int, framePatches fp: Int,
                       bucket forcedBucket: Int? = nil) throws -> [Float] {
        let product = t.multipliedReportingOverflow(by: fp)
        guard t > 0, fp > 0, !product.overflow else {
            throw EncoderError.invalidInput("video frame geometry is invalid")
        }
        let L = product.partialValue
        let N = forcedBucket ?? bucket(forPatches: L)
        guard L <= N, forcedBucket == nil || patchBuckets.contains(N) else { throw EncoderError.tooLarge(L) }
        guard pixelDim == 1_536, hidden == 1_024, ropeDim == 64,
              pixelValues.count == L * pixelDim,
              posEmbeds.count == L * hidden,
              cos.count == L * ropeDim,
              sin.count == L * ropeDim else {
            throw EncoderError.invalidInput("video tensors do not match the native patch geometry")
        }
        let pv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: pixelDim)], dataType: .float32)
        let pe = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: hidden)], dataType: .float32)
        let cv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let sv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let ab = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: N)], dataType: .float32)
        try CoreMLArrayReader.fillFloat32(pv, with: pixelValues, label: "video pixels")
        try CoreMLArrayReader.fillFloat32(pe, with: posEmbeds, label: "video positions")
        try CoreMLArrayReader.fillFloat32(cv, with: cos, label: "video rope cosine")
        try CoreMLArrayReader.fillFloat32(sv, with: sin, label: "video rope sine")
        let abp = ab.dataPointer.bindMemory(to: Float.self, capacity: N * N)
        for i in 0..<(N * N) { abp[i] = Self.neg }
        for f in 0..<t {   // per-frame block-diagonal: rows/cols [f·fp, f·fp+fp)
            for i in (f * fp)..<((f + 1) * fp) {
                let row = i * N
                for j in (f * fp)..<((f + 1) * fp) { abp[row + j] = 0.0 }
            }
        }
        let out = try model(N).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": pv, "pos_embeds": pe, "rope_cos": cv, "rope_sin": sv, "attn_bias": ab,
        ]))
        guard let fr = out.featureValue(for: "vision_features")?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try CoreMLArrayReader.float32(
            fr, shape: [N / 4, hidden], label: "video encoder features")
    }
}

/// Host-side video preparation: bounded frame decode and sampling, patchify, positions, prompt ids.
/// A pure `Sendable` value with no Core ML state, so it can run on any thread.
internal struct VideoInputPreparer: Sendable {
    let preprocessor: GlossImagePreprocessor
    let positions: VisionPositions
    let tokens: MediaTokens
    /// Largest converted patch bucket; sampling and resize are planned to fit it.
    let maxPatches: Int

    /// Video file -> encoder inputs with the bounded, aspect-preserving 2 fps serving profile.
    /// Sampling and resize are not bit-matched to the reference, but the resulting frame path is
    /// independently validated against the source model. `checkCancellation` is polled between
    /// frames so a cancelled request stops decoding instead of finishing seconds of wasted work.
    func prepare(videoURL: URL, prompt: GlossTextEmbedder.Prompt,
                 checkCancellation: (() throws -> Void)? = nil) throws -> PreparedVisionInputs {
        let input = try GlossVideoFile.extractFrames(
            videoURL, maxPatches: maxPatches, preprocessor: preprocessor, checkCancellation: checkCancellation)
        try checkCancellation?()
        return try prepare(frames: input.frames, h: input.h, w: input.w, prompt: prompt)
    }

    /// Raw RGB frames (count = 2·t, each h*w*3, h/w factor-aligned) -> encoder inputs. (Frame
    /// *sampling* is the caller's job.)
    func prepare(frames: [[UInt8]], h: Int, w: Int, prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        let (pv, t, gh, gw) = try preprocessor.videoPixelValues(frames: frames, h: h, w: w)
        return try prepare(pixelValues: pv, t: t, gh: gh, gw: gw, prompt: prompt)
    }

    /// `pixelValues` = (t·gh·gw · pixelDim) row-major (video processor output); grid (t,gh,gw).
    func prepare(pixelValues: [Float], t: Int, gh: Int, gw: Int,
                 prompt: GlossTextEmbedder.Prompt) throws -> PreparedVisionInputs {
        let spatial = gh.multipliedReportingOverflow(by: gw)
        let temporal = t.multipliedReportingOverflow(by: spatial.partialValue)
        guard t > 0, gh > 0, gw > 0,
              gh.isMultiple(of: positions.mergeSize),
              gw.isMultiple(of: positions.mergeSize),
              !spatial.overflow, !temporal.overflow,
              temporal.partialValue <= maxPatches,
              pixelValues.count == temporal.partialValue * preprocessor.featuresPerPatch,
              pixelValues.allSatisfy(\.isFinite) else {
            throw VideoCoreMLEncoderMasked.EncoderError.invalidInput("video pixels or grid are invalid")
        }
        let (pe, cosv, sinv, merged) = positions.computeVideo(t: t, gh: gh, gw: gw)
        return PreparedVisionInputs(
            patches: temporal.partialValue, pixelValues: pixelValues, posEmbeds: pe, ropeCos: cosv, ropeSin: sinv,
            temporalGroups: t, groupPatches: spatial.partialValue, mergedTokens: merged,
            tokenIDs: tokens.ids(prompt: prompt, count: merged),
            scatterOffset: tokens.resolvedPrefix(for: prompt).count)
    }
}

/// On-device VIDEO embedder: bounded frame patchify, block-diagonal ViT, and general decoder.
/// `embed(videoURL:)` uses AVFoundation frame extraction; `embed(pixelValues:...)` accepts prepared
/// tensors for converter parity tests. As with images, every `embed(...)` is
/// `infer(preparer.prepare(...))`.
internal final class GlossVideoEmbedderMasked {
    /// The video prompt wrapper (`prefix + placeholder × merged + suffix`).
    public let tokens: MediaTokens
    /// Media feature width == the model's embedding dimension.
    public let featureDim: Int

    public let positions: VisionPositions
    public let encoder: VideoCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder
    public let preprocessor: GlossImagePreprocessor
    /// The CPU half of the pipeline, safe to use from any thread.
    public let preparer: VideoInputPreparer

    /// Assembles an embedder from already-built parts (see ``GlossImageEmbedderMasked/init(visionModelURL:positions:decoder:tokens:featureDim:patchBuckets:preprocessor:encoderUnits:)``).
    public init(visionModelURL: URL, positions: VisionPositions, decoder: GeneralMediaDecoder,
                tokens: MediaTokens, featureDim: Int = 1024,
                patchBuckets: [Int] = VideoCoreMLEncoderMasked.defaultPatchBuckets,
                preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                encoderUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.tokens = tokens
        self.featureDim = featureDim
        self.preprocessor = preprocessor
        self.positions = positions
        self.decoder = decoder
        encoder = try VideoCoreMLEncoderMasked(modelURL: visionModelURL, computeUnits: encoderUnits, patchBuckets: patchBuckets)
        preparer = VideoInputPreparer(
            preprocessor: preprocessor, positions: positions, tokens: tokens,
            maxPatches: encoder.patchBuckets.last ?? 0)
    }

    /// `encoderUnits`/`decoderUnits` as `GlossImageEmbedderMasked`: default = GPU ViT + adaptive decoder;
    /// pass both as `.cpuAndNeuralEngine` for a true full-ANE deployment (above the bf16 floor, slower).
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            metaURL: URL, posTableURL: URL, invFreqURL: URL,
                            tokens: MediaTokens, featureDim: Int = 1024,
                            patchBuckets: [Int] = VideoCoreMLEncoderMasked.defaultPatchBuckets,
                            padTokenID: Int32 = 0,
                            preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                            encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil,
                            sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        try self.init(
            visionModelURL: visionModelURL,
            positions: try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: invFreqURL),
            decoder: try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                                             computeUnits: decoderUnits, featDim: featureDim,
                                             padTokenID: padTokenID, sequenceBuckets: sequenceBuckets),
            tokens: tokens, featureDim: featureDim, patchBuckets: patchBuckets,
            preprocessor: preprocessor, encoderUnits: encoderUnits)
    }

    /// Convenience: vision resources loaded by name from `resourcesDir` (see `export_vision_swift_refs.py`).
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            resourcesDir: URL, tokens: MediaTokens, featureDim: Int = 1024,
                            patchBuckets: [Int] = VideoCoreMLEncoderMasked.defaultPatchBuckets,
                            padTokenID: Int32 = 0,
                            preprocessor: GlossImagePreprocessor = GlossImagePreprocessor(),
                            encoderUnits: MLComputeUnits = .cpuAndGPU,
                            decoderUnits: MLComputeUnits? = nil,
                            sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        try self.init(visionModelURL: visionModelURL, embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                      metaURL: resourcesDir.appendingPathComponent("meta.json"),
                      posTableURL: resourcesDir.appendingPathComponent("pos_embed_table.f32"),
                      invFreqURL: resourcesDir.appendingPathComponent("rope_inv_freq.f32"),
                      tokens: tokens, featureDim: featureDim, patchBuckets: patchBuckets,
                      padTokenID: padTokenID, preprocessor: preprocessor,
                      encoderUnits: encoderUnits, decoderUnits: decoderUnits,
                      sequenceBuckets: sequenceBuckets)
    }

    /// Raw RGB frames (count = 2·t, each h*w*3, h/w factor-aligned) -> embedding. The full host path:
    /// frame-patchify -> block-diagonal ViT -> general decoder. (Frame *sampling* is the caller's job.)
    public func embed(frames: [[UInt8]], h: Int, w: Int, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(frames: frames, h: h, w: w, prompt: prompt), dim: dim)
    }

    /// Video file -> embedding with the bounded, aspect-preserving 2 fps serving profile.
    /// Sampling and resize are not bit-matched to the reference, but the resulting frame path is
    /// independently validated against the source model.
    public func embed(videoURL: URL, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(videoURL: videoURL, prompt: prompt), dim: dim)
    }

    /// Explicit fixed-square profile for direct extraction and model tests.
    public func embed(videoURL: URL, frameCount: Int, frameSize: Int, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        guard frameCount > 0, frameCount <= 32, frameCount.isMultiple(of: 2),
              frameSize >= 32, frameSize <= 512,
              frameSize.isMultiple(of: preprocessor.factor),
              (frameCount / 2) * (frameSize / preprocessor.patch) * (frameSize / preprocessor.patch)
                  <= encoder.patchBuckets.last! else {
            throw VideoCoreMLEncoderMasked.EncoderError.invalidInput(
                "video frame count or size exceeds the native patch grid")
        }
        let frames = try GlossVideoFile.extractSquareFrames(videoURL, count: frameCount, size: frameSize, preprocessor: preprocessor)
        return try embed(frames: frames, h: frameSize, w: frameSize, dim: dim, prompt: prompt)
    }

    /// `pixelValues` = (t·gh·gw · pixelDim) row-major (video processor output); grid (t,gh,gw).
    public func embed(pixelValues: [Float], t: Int, gh: Int, gw: Int, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try infer(try preparer.prepare(pixelValues: pixelValues, t: t, gh: gh, gw: gw, prompt: prompt), dim: dim)
    }

    /// The Core ML half: block-diagonal ViT prediction, merged-feature slice, scatter into the
    /// prompt, decoder prediction, optional Matryoshka truncation.
    public func infer(_ prepared: PreparedVisionInputs, dim: Int? = nil) throws -> [Float] {
        let full = try encoder.encode(pixelValues: prepared.pixelValues, pixelDim: preprocessor.featuresPerPatch,
                                      posEmbeds: prepared.posEmbeds, hidden: positions.hidden,
                                      cos: prepared.ropeCos, sin: prepared.ropeSin, ropeDim: positions.ropeDim,
                                      frames: prepared.temporalGroups, framePatches: prepared.groupPatches)
        let used = Array(full[0 ..< (prepared.mergedTokens * featureDim)])
        let emb = try decoder.decode(tokenIds: prepared.tokenIDs, features: used,
                                     scatterOffset: prepared.scatterOffset)
        if let d = dim { return matryoshka(emb, dim: d) }
        return emb
    }
}
