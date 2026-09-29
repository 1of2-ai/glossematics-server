import AVFoundation
import CoreVideo
import Foundation

/// Decodes the frames the source video processor would see, the way the reference pipeline
/// produces them (ffmpeg decode + Hugging Face `Qwen3VLVideoProcessor`):
///
/// * Frames are counted and addressed by decode order, as `len(frames)` and frame indices are in
///   the reference; there is no timestamp arithmetic (seeking to `index / fps` lands on the
///   previous frame whenever the product rounds down).
/// * Y'CbCr is converted to RGB here, not by ColorSync: the matrix is the stream's tagged one
///   (BT.601, BT.709, or BT.2020) and BT.601 when untagged, video or full range from the stream,
///   chroma replicated per 2x2 block, as ffmpeg's unscaled `yuv420p -> rgb24` path does. The
///   display rotation is applied like ffmpeg's autorotate.
/// * Frames are resized with an antialiased separable bicubic filter (a = -0.5, support scaled by
///   the downscale factor, pixel-center mapping) and rounded to 8 bits: torchvision's
///   `resize(..., BICUBIC, antialias=True)` on uint8 frames.
/// * The SOURCE frame size is capped before anything is decoded (see ``maximumSourceEdge``), the
///   video counterpart of the image path's 40-megapixel header check: a small compressed upload
///   must not be able to force an unbounded frame decode.
enum VideoFrameDecoder {
    enum Failure: Error, CustomStringConvertible {
        case noVideoTrack
        case readerFailed(String)
        case tooManyFrames(Int)
        case unsupportedTransform
        case frameMissing(Int)
        /// The declared frame size is zero, negative, non-finite, or rounds to less than one pixel.
        case invalidFrameSize(width: Double, height: Double)
        /// The displayed (or coded) frame exceeds the per-frame source cap.
        case frameTooLarge(width: Int, height: Int)

        var description: String {
            switch self {
            case .noVideoTrack: "file has no readable video track"
            case let .readerFailed(reason): "video could not be decoded: \(reason)"
            case let .tooManyFrames(limit): "video has more than \(limit) frames"
            case .unsupportedTransform: "video display transform is not a 0/90/180/270 degree rotation"
            case let .frameMissing(index): "video frame \(index) could not be decoded"
            case let .invalidFrameSize(width, height):
                "video declares an invalid frame size (\(width) × \(height)); width and height must be positive finite pixel counts"
            case let .frameTooLarge(width, height):
                "video frames are \(width) × \(height); the maximum source frame is "
                    + "\(VideoFrameDecoder.maximumSourceEdge) pixels on either edge and "
                    + "\(VideoFrameDecoder.maximumSourcePixels / 1_000_000) megapixels "
                    + "(DCI 4K, 4096 × 2160, is accepted in either orientation; 8K is not)"
            }
        }
    }

    /// Per-frame SOURCE cap: at most 4096 pixels on either edge and 4096 × 4096 = 16,777,216 pixels.
    ///
    /// * It admits every common 4K delivery format in EITHER orientation — DCI 4K (4096 × 2160,
    ///   8.8 MP), UHD (3840 × 2160, 8.3 MP), and the same rotated 90 degrees for portrait phone
    ///   video — and rejects 5K (5120 × 2880) and 8K (7680 × 4320, 33 MP).
    /// * The serving profile spends the converted tower's whole 2048-patch budget on at most 32
    ///   sampled frames, so every source frame is downscaled to roughly a megapixel or less; source
    ///   pixels beyond 4K buy no embedding quality. They only cost decode time and memory, because
    ///   Y'CbCr -> RGB runs per pixel in Double at source resolution and each decoded frame is held
    ///   at that resolution while it is resized.
    /// * The edge cap is symmetric, so it is unaffected by the display rotation.
    static let maximumSourceEdge = 4_096
    static let maximumSourcePixels = 16_777_216

    /// Validate a declared frame size (in pixels) against the source cap BEFORE any decoding. Takes
    /// `Double` because `AVAssetTrack.naturalSize` is a `CGSize`; a hostile container can declare
    /// zero, negative, NaN, or infinite dimensions, and converting those to `Int` would trap the
    /// whole server. Returns the rounded integer size.
    static func validateSourceSize(width: Double, height: Double) throws -> (width: Int, height: Int) {
        guard width.isFinite, height.isFinite, width.rounded() >= 1, height.rounded() >= 1 else {
            throw Failure.invalidFrameSize(width: width, height: height)
        }
        let edge = Double(maximumSourceEdge)
        let widest = max(width.rounded(), height.rounded())
        // Compared in Double first: a huge finite value must be rejected before the Int conversion.
        guard widest <= edge,
              width.rounded() * height.rounded() <= Double(maximumSourcePixels) else {
            let clamp = { (value: Double) -> Int in Int(min(value.rounded(), Double(Int32.max))) }
            throw Failure.frameTooLarge(width: clamp(width), height: clamp(height))
        }
        return (Int(width.rounded()), Int(height.rounded()))
    }

    struct Track {
        let asset: AVURLAsset
        let track: AVAssetTrack
        /// Displayed (rotated) size.
        let width: Int
        let height: Int
        let quarterTurns: Int
    }

    static let maximumFrames = 60 * 240

    static func open(_ url: URL) throws -> Track {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { throw Failure.noVideoTrack }
        let t = track.preferredTransform
        let turns: Int
        switch (Int(t.a.rounded()), Int(t.b.rounded()), Int(t.c.rounded()), Int(t.d.rounded())) {
        case (1, 0, 0, 1): turns = 0
        case (0, 1, -1, 0): turns = 1        // 90 degrees clockwise
        case (-1, 0, 0, -1): turns = 2
        case (0, -1, 1, 0): turns = 3
        default: throw Failure.unsupportedTransform
        }
        // Validate the declared size before anything is decoded or converted to Int. The decoded
        // buffer's real size is checked again before its RGB copy is allocated (`rgbFrame`), since
        // a crafted container can declare one size and encode another.
        let natural = track.naturalSize
        let (w, h) = try validateSourceSize(width: Double(natural.width), height: Double(natural.height))
        return Track(asset: asset, track: track, width: turns % 2 == 0 ? w : h, height: turns % 2 == 0 ? h : w,
                     quarterTurns: turns)
    }

    /// Average frame rate as ffmpeg reports it (`avg_frame_rate`: frames over track duration).
    /// `nominalFrameRate` is a Float32 estimate (30 fps reads as 30.000002), which moves
    /// `int(frames / fps * 2)` across an integer boundary.
    static func averageFrameRate(_ source: Track, frames: Int) -> Double {
        let duration = source.track.timeRange.duration.seconds
        if duration.isFinite, duration > 0, frames > 0 { return Double(frames) / duration }
        let nominal = Double(source.track.nominalFrameRate)
        return nominal.isFinite && nominal > 0 ? nominal : 0
    }

    /// Number of video samples, without decoding them. `checkCancellation` is polled per sample so
    /// a cancelled request stops scanning.
    static func frameCount(_ source: Track, checkCancellation: (() throws -> Void)? = nil) throws -> Int {
        let reader = try AVAssetReader(asset: source.asset)
        let output = AVAssetReaderTrackOutput(track: source.track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw Failure.readerFailed("cannot read the video track") }
        reader.add(output)
        guard reader.startReading() else { throw Failure.readerFailed(String(describing: reader.error)) }
        var count = 0
        while let sample = output.copyNextSampleBuffer() {
            try checkCancellation?()
            count += CMSampleBufferGetNumSamples(sample)
            if count > maximumFrames { reader.cancelReading(); throw Failure.tooManyFrames(maximumFrames) }
        }
        if reader.status == .failed { throw Failure.readerFailed(String(describing: reader.error)) }
        return count
    }

    /// Decode the frames at `indices` (decode order, ascending, may repeat) as displayed RGB,
    /// resized to `width` x `height`. `checkCancellation` is polled per sample.
    static func frames(_ source: Track, indices: [Int], width: Int, height: Int,
                       checkCancellation: (() throws -> Void)? = nil) throws -> [[UInt8]] {
        let format = source.track.formatDescriptions.first.map { $0 as! CMFormatDescription }
        let extensions = format.flatMap { CMFormatDescriptionGetExtensions($0) as? [String: Any] } ?? [:]
        let fullRange = (extensions[kCMFormatDescriptionExtension_FullRangeVideo as String] as? Bool) ?? false
        let matrix = YCbCrMatrix(tag: extensions[kCMFormatDescriptionExtension_YCbCrMatrix as String] as? String)
        let reader = try AVAssetReader(asset: source.asset)
        let pixelFormat = fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let output = AVAssetReaderTrackOutput(track: source.track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw Failure.readerFailed("cannot decode the video track") }
        reader.add(output)
        guard reader.startReading() else { throw Failure.readerFailed(String(describing: reader.error)) }
        defer { if reader.status == .reading { reader.cancelReading() } }

        let wanted = Set(indices)
        let last = indices.max() ?? -1
        var decoded = [Int: [UInt8]]()
        var index = 0
        while index <= last, let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            try checkCancellation?()
            guard wanted.contains(index) else { continue }
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw Failure.frameMissing(index) }
            var (rgb, w, h) = try rgbFrame(buffer, matrix: matrix, fullRange: fullRange)
            (rgb, w, h) = rotate(rgb, width: w, height: h, quarterTurns: source.quarterTurns)
            decoded[index] = AntialiasedBicubic.resize(rgb, width: w, height: h, toWidth: width, toHeight: height)
        }
        if reader.status == .failed { throw Failure.readerFailed(String(describing: reader.error)) }
        return try indices.map { i in
            guard let frame = decoded[i] else { throw Failure.frameMissing(i) }
            return frame
        }
    }

    struct YCbCrMatrix {
        let kr: Double, kb: Double
        init(tag: String?) {
            if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String {
                (kr, kb) = (0.2126, 0.0722)
            } else if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as String {
                (kr, kb) = (0.2627, 0.0593)
            } else {
                (kr, kb) = (0.299, 0.114)        // ITU-R BT.601, also for untagged streams
            }
        }
    }

    /// Bi-planar 4:2:0 8-bit -> packed RGB at the coded size.
    static func rgbFrame(_ buffer: CVPixelBuffer, matrix: YCbCrMatrix, fullRange: Bool) throws -> ([UInt8], Int, Int) {
        guard CVPixelBufferGetPlaneCount(buffer) == 2 else { throw Failure.readerFailed("unexpected pixel layout") }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0), h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        // The decoder's actual buffer, not the declared size, decides the allocation below.
        _ = try validateSourceSize(width: Double(w), height: Double(h))
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else {
            throw Failure.readerFailed("frame has no pixel data")
        }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let yPlane = yBase.assumingMemoryBound(to: UInt8.self)
        let cPlane = cBase.assumingMemoryBound(to: UInt8.self)
        let kr = matrix.kr, kb = matrix.kb, kg = 1 - kr - kb
        let yScale = fullRange ? 1.0 : 255.0 / 219.0, yOffset = fullRange ? 0.0 : 16.0
        let cScale = fullRange ? 1.0 : 255.0 / 224.0
        let crR = 2 * (1 - kr), cbB = 2 * (1 - kb)
        let cbG = 2 * kb * (1 - kb) / kg, crG = 2 * kr * (1 - kr) / kg
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        rgb.withUnsafeMutableBufferPointer { out in
            for y in 0..<h {
                let yRow = yPlane + y * yStride, cRow = cPlane + (y / 2) * cStride
                for x in 0..<w {
                    let luma = (Double(yRow[x]) - yOffset) * yScale
                    let cb = (Double(cRow[(x / 2) * 2]) - 128) * cScale
                    let cr = (Double(cRow[(x / 2) * 2 + 1]) - 128) * cScale
                    let o = (y * w + x) * 3
                    out[o] = clamp8(luma + crR * cr)
                    out[o + 1] = clamp8(luma - cbG * cb - crG * cr)
                    out[o + 2] = clamp8(luma + cbB * cb)
                }
            }
        }
        return (rgb, w, h)
    }

    @inline(__always) static func clamp8(_ v: Double) -> UInt8 { UInt8(max(0, min(255, v.rounded()))) }

    /// Rotate packed RGB clockwise by `quarterTurns` x 90 degrees.
    static func rotate(_ rgb: [UInt8], width w: Int, height h: Int, quarterTurns: Int) -> ([UInt8], Int, Int) {
        guard quarterTurns % 4 != 0 else { return (rgb, w, h) }
        let (ow, oh) = quarterTurns % 2 == 0 ? (w, h) : (h, w)
        var out = [UInt8](repeating: 0, count: rgb.count)
        for y in 0..<h {
            for x in 0..<w {
                let (ox, oy): (Int, Int)
                switch quarterTurns % 4 {
                case 1: (ox, oy) = (h - 1 - y, x)
                case 2: (ox, oy) = (w - 1 - x, h - 1 - y)
                default: (ox, oy) = (y, w - 1 - x)
                }
                let s = (y * w + x) * 3, d = (oy * ow + ox) * 3
                out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2]
            }
        }
        return (out, ow, oh)
    }
}

/// torchvision / PIL antialiased bicubic resampling (a = -0.5) for packed 8-bit RGB.
enum AntialiasedBicubic {
    private static func kernel(_ x: Double) -> Double {
        let a = -0.5, x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Per output index: first source index and normalized weights.
    static func weights(input: Int, output: Int) -> [(start: Int, weights: [Double])] {
        let scale = Double(input) / Double(output)
        let support = 2.0 * max(scale, 1)
        let inverse = 1 / max(scale, 1)
        return (0..<output).map { i in
            let center = (Double(i) + 0.5) * scale
            let lo = max(Int(center - support + 0.5), 0)
            let hi = min(Int(center + support + 0.5), input)
            var w = (lo..<hi).map { kernel((Double($0) - center + 0.5) * inverse) }
            let total = w.reduce(0, +)
            if total != 0 { w = w.map { $0 / total } }
            return (lo, w)
        }
    }

    static func resize(_ rgb: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        if width == toWidth && height == toHeight { return rgb }
        let hw = weights(input: width, output: toWidth)
        let vw = weights(input: height, output: toHeight)
        // Horizontal pass into floats, then vertical pass with rounding.
        var mid = [Double](repeating: 0, count: height * toWidth * 3)
        mid.withUnsafeMutableBufferPointer { m in
            rgb.withUnsafeBufferPointer { src in
                DispatchQueue.concurrentPerform(iterations: height) { y in
                    let row = y * width * 3
                    for x in 0..<toWidth {
                        let (start, w) = hw[x]
                        var r = 0.0, g = 0.0, b = 0.0
                        for (k, weight) in w.enumerated() {
                            let s = row + (start + k) * 3
                            r += weight * Double(src[s]); g += weight * Double(src[s + 1]); b += weight * Double(src[s + 2])
                        }
                        let o = (y * toWidth + x) * 3
                        m[o] = r; m[o + 1] = g; m[o + 2] = b
                    }
                }
            }
        }
        var out = [UInt8](repeating: 0, count: toHeight * toWidth * 3)
        out.withUnsafeMutableBufferPointer { o in
            mid.withUnsafeBufferPointer { m in
                DispatchQueue.concurrentPerform(iterations: toHeight) { y in
                    let (start, w) = vw[y]
                    for x in 0..<(toWidth * 3) {
                        var v = 0.0
                        for (k, weight) in w.enumerated() { v += weight * m[(start + k) * toWidth * 3 + x] }
                        o[y * toWidth * 3 + x] = UInt8(max(0, min(255, v.rounded())))
                    }
                }
            }
        }
        return out
    }
}
