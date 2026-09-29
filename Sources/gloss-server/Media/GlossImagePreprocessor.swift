import CoreGraphics
import Foundation
import ImageIO

/// Replicates Qwen3VL image preprocessing for the variable-resolution masked ViT: smart-resize
/// to the native patch grid, normalize `(v/255 - 0.5)/0.5`, then patchify into rows of 1,536
/// values. The fixed `size` path remains available for direct parity tests.
///
/// Patchify index map (verified against the processor):
///   pixel_values[((gh*GW+gw)*merge+mh)*merge+mw][((c*temporal+t)*patch+ph)*patch+pw]
///     = norm(image[c][gh*merge*patch + mh*patch + ph][gw*merge*patch + mw*patch + pw])
/// with temporal frames identical (single image repeated).
internal struct GlossImagePreprocessor {
    public let size: Int
    public let patch = 16, merge = 2, temporal = 2

    /// Smart-resize pixel bounds of the model's image processor. Defaults are the Qwen-VL-family
    /// values (jina-v5-omni-small); bundles override them from `manifest.image.preprocess`
    /// (jina-v5-omni-nano uses 262144/1310720, for example).
    public let minPixels: Int, maxPixels: Int

    public init(size: Int = 512, minPixels: Int = 65536, maxPixels: Int = 16777216) {
        self.size = size
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    public var numPatches: Int { let g = size / patch; return g * g }              // 1024
    public var featuresPerPatch: Int { 3 * temporal * patch * patch }             // 1536

    public var factor: Int { patch * merge }

    /// Qwen smart_resize: round H,W to multiples of `factor`, keep aspect ratio, clamp the pixel
    /// budget to [minPixels, maxPixels]. Returns (Hbar, Wbar) — the ViT then sees grid (Hbar/16, Wbar/16).
    /// `maxPixelsOverride` caps the budget below the model's native max (used to fit a patch-bucket
    /// ceiling: images larger than the largest converted bucket are downscaled instead of unsupported).
    public func smartResize(h: Int, w: Int, maxPixelsOverride: Int? = nil) -> (Int, Int) {
        let f = Double(factor)
        let sourceH = max(1, h), sourceW = max(1, w)
        let maxPixels = max(factor * factor, min(self.maxPixels, maxPixelsOverride ?? self.maxPixels))
        let minPixels = min(self.minPixels, maxPixels)
        // Python `round()` — half to EVEN — exactly as the source processor's `smart_resize`.
        // Swift's plain `.rounded()` rounds ties away from zero, which put every side of
        // k*32+16 (k even: 400, 464, ..., 720, ...) on the wrong grid row: 720 / 32 = 22.5 is 22
        // rows (704) in the source but 23 (736) with `.rounded()`. Only this initial rounding is
        // half-to-even; the floor/ceil budget branches below match Python's floor/ceil.
        func roundF(_ x: Double) -> Int { Int((x / f).rounded(.toNearestOrEven)) * factor }
        var hbar = max(factor, roundF(Double(sourceH)))
        var wbar = max(factor, roundF(Double(sourceW)))
        let sourcePixels = Double(sourceH) * Double(sourceW)
        if Double(hbar) * Double(wbar) > Double(maxPixels) {
            let beta = (sourcePixels / Double(maxPixels)).squareRoot()
            hbar = max(factor, Int((Double(sourceH) / beta / f).rounded(.down)) * factor)
            wbar = max(factor, Int((Double(sourceW) / beta / f).rounded(.down)) * factor)
        } else if Double(hbar) * Double(wbar) < Double(minPixels) {
            let beta = (Double(minPixels) / sourcePixels).squareRoot()
            hbar = max(factor, Int((Double(sourceH) * beta / f).rounded(.up)) * factor)
            wbar = max(factor, Int((Double(sourceW) * beta / f).rounded(.up)) * factor)
        }
        // Aspect ratios narrower than one alignment unit can still exceed the budget after
        // clamping that side to 32 pixels. Correct the longer side before allocating RGB/tensors.
        if hbar > maxPixels / wbar {
            hbar = max(factor, (maxPixels / wbar / factor) * factor)
        }
        if wbar > maxPixels / hbar {
            wbar = max(factor, (maxPixels / hbar / factor) * factor)
        }
        return (hbar, wbar)
    }

    /// General variable-resolution patchify from a row-major RGB buffer already sized to (h,w) with
    /// h,w factor-aligned (e.g. a smart-resized image). Returns pixel_values (gh*gw, featuresPerPatch)
    /// in the merger's 2×2-block order plus the patch grid (gh=h/16, gw=w/16).
    public func pixelValues(rgb: [UInt8], h: Int, w: Int) throws -> (pixels: [Float], gh: Int, gw: Int) {
        let pixels = h.multipliedReportingOverflow(by: w)
        guard h > 0, w > 0,
              h.isMultiple(of: factor), w.isMultiple(of: factor),
              !pixels.overflow, pixels.partialValue <= 5_120 * patch * patch,
              rgb.count == pixels.partialValue * 3 else {
            throw ImageError.invalidGeometry("RGB dimensions or byte count exceed the native image grid")
        }
        let GH = h / (patch * merge), GW = w / (patch * merge)   // merge-block grid
        let gh = h / patch, gw = w / patch                       // patch grid (for positions)
        let FPP = featuresPerPatch, bpr = w * 3
        var out = [Float](repeating: 0, count: gh * gw * FPP)
        for bh in 0..<GH {
            for bw in 0..<GW {
                for mh in 0..<merge {
                    for mw in 0..<merge {
                        let patchIdx = ((bh * GW + bw) * merge + mh) * merge + mw
                        let base = patchIdx * FPP
                        for c in 0..<3 {
                            for ph in 0..<patch {
                                let H = bh * (merge * patch) + mh * patch + ph
                                let row = H * bpr
                                for pw in 0..<patch {
                                    let W = bw * (merge * patch) + mw * patch + pw
                                    let v = Float(rgb[row + W * 3 + c]) / 127.5 - 1.0
                                    for t in 0..<temporal {
                                        out[base + ((c * temporal + t) * patch + ph) * patch + pw] = v
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, gh, gw)
    }

    public enum ImageError: Error {
        case load(String)
        case context
        case invalidGeometry(String)
        case badVideoFrameCount(Int)
        case sourceTooLargePixels(Int)
    }

    /// The source-image ceiling prevents a small compressed upload from forcing an unbounded
    /// image decode before the model's normal smart-resize step.
    private static let maximumSourcePixels = 40_000_000

    public static func loadCGImage(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw ImageError.load(url.path)
        }
        guard let image = try decode(src) else {
            throw ImageError.load(url.path)
        }
        return image
    }

    public static func loadCGImage(_ data: Data) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw ImageError.load("image data")
        }
        guard let image = try decode(src) else {
            throw ImageError.load("image data")
        }
        return image
    }

    private static func decode(_ source: CGImageSource) throws -> CGImage? {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            throw ImageError.load("image dimensions are unavailable")
        }
        let product = width.multipliedReportingOverflow(by: height)
        guard !product.overflow, product.partialValue <= maximumSourcePixels else {
            throw ImageError.sourceTooLargePixels(product.overflow ? Int.max : product.partialValue)
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let decodedProduct = image.width.multipliedReportingOverflow(by: image.height)
        guard !decodedProduct.overflow,
              decodedProduct.partialValue <= maximumSourcePixels else {
            throw ImageError.sourceTooLargePixels(
                decodedProduct.overflow ? Int.max : decodedProduct.partialValue)
        }
        return image
    }

    /// VIDEO frame-patchify: `frames` = `2·t` RGB buffers (h*w*3 each, h/w factor-aligned). Pairs
    /// consecutive frames into the temporal_patch_size=2 slots (patch's 1536 = c×temporal×16×16, slot
    /// 0 = frame 2g, slot 1 = frame 2g+1), in the merger's 2×2-block order. Returns pixel_values_videos
    /// (t·gh·gw, featuresPerPatch) + grid (t, gh, gw). (fps frame *sampling* is the caller's job.)
    public func videoPixelValues(frames: [[UInt8]], h: Int, w: Int) throws -> (pixels: [Float], t: Int, gh: Int, gw: Int) {
        guard frames.count > 0, frames.count % temporal == 0 else { throw ImageError.badVideoFrameCount(frames.count) }
        let pixels = h.multipliedReportingOverflow(by: w)
        let temporalPatches = (frames.count / temporal).multipliedReportingOverflow(
            by: pixels.partialValue / (patch * patch))
        guard h > 0, w > 0,
              h.isMultiple(of: factor), w.isMultiple(of: factor),
              !pixels.overflow, pixels.partialValue <= 2_048 * patch * patch,
              !temporalPatches.overflow, temporalPatches.partialValue <= 2_048,
              frames.allSatisfy({ $0.count == pixels.partialValue * 3 }) else {
            throw ImageError.invalidGeometry("video frames exceed the native grid or have an invalid byte count")
        }
        let t = frames.count / temporal
        let GH = h / (patch * merge), GW = w / (patch * merge)
        let gh = h / patch, gw = w / patch
        let FPP = featuresPerPatch, bpr = w * 3, fpatch = gh * gw
        var out = [Float](repeating: 0, count: t * fpatch * FPP)
        for g in 0..<t {
            for bh in 0..<GH {
                for bw in 0..<GW {
                    for mh in 0..<merge {
                        for mw in 0..<merge {
                            let patchIdx = g * fpatch + ((bh * GW + bw) * merge + mh) * merge + mw
                            let base = patchIdx * FPP
                            for c in 0..<3 {
                                for ph in 0..<patch {
                                    let H = bh * (merge * patch) + mh * patch + ph
                                    let row = H * bpr
                                    for pw in 0..<patch {
                                        let W = bw * (merge * patch) + mw * patch + pw
                                        for tt in 0..<temporal {
                                            let v = Float(frames[g * temporal + tt][row + W * 3 + c]) / 127.5 - 1.0
                                            out[base + ((c * temporal + tt) * patch + ph) * patch + pw] = v
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, t, gh, gw)
    }

    /// Draw a CGImage resized to (w,h) into an RGBA8 buffer and return packed RGB (h*w*3) row-major.
    /// NOTE: CoreGraphics resampling is not bit-identical to PIL bicubic, so for images whose native
    /// size differs from (w,h) the pixel_values (hence embedding) carry a small resample-only error.
    public func resizedRGB(_ cgImage: CGImage, w: Int, h: Int) throws -> [UInt8] {
        let pixels = h.multipliedReportingOverflow(by: w)
        guard h > 0, w > 0, !pixels.overflow,
              pixels.partialValue <= 5_120 * patch * patch else {
            throw ImageError.invalidGeometry("resized image exceeds the native image grid")
        }
        let bpr = w * 4
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ImageError.context
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let dp = ctx.data, ctx.bytesPerRow == bpr else { throw ImageError.context }
        let buf = dp.bindMemory(to: UInt8.self, capacity: h * bpr)
        var rgb = [UInt8](repeating: 0, count: h * w * 3)
        for y in 0..<h {
            for x in 0..<w {
                let s = y * bpr + x * 4, d = (y * w + x) * 3
                rgb[d] = buf[s]; rgb[d + 1] = buf[s + 1]; rgb[d + 2] = buf[s + 2]
            }
        }
        return rgb
    }

    /// The source video processor uses torchvision bicubic interpolation. CoreGraphics `.high`
    /// differs noticeably when small decoded frames are enlarged, so use a separable cubic
    /// kernel (a = -0.75, pixel-center mapping) for that bounded case. Downscales retain the
    /// platform's filtered resize; a full antialiased source match remains a separate concern.
    public func videoResizedRGB(_ cgImage: CGImage, w: Int, h: Int) throws -> [UInt8] {
        let pixels = w.multipliedReportingOverflow(by: h)
        guard w > 0, h > 0, !pixels.overflow,
              pixels.partialValue <= 5_120 * patch * patch else {
            throw ImageError.invalidGeometry("resized video frame exceeds the native grid")
        }
        guard w > cgImage.width || h > cgImage.height else {
            return try resizedRGB(cgImage, w: w, h: h)
        }
        guard w >= cgImage.width, h >= cgImage.height else {
            return try resizedRGB(cgImage, w: w, h: h)
        }
        let source = try resizedRGB(cgImage, w: cgImage.width, h: cgImage.height)
        let sourceW = cgImage.width, sourceH = cgImage.height

        func weights(_ output: Int, sourceCount: Int, outputCount: Int) -> [(Int, Float)] {
            let coordinate = (Double(output) + 0.5) * Double(sourceCount) / Double(outputCount) - 0.5
            let base = Int(floor(coordinate))
            return (-1...2).map { offset in
                let sample = base + offset
                let x = abs(coordinate - Double(sample))
                let a = -0.75
                let weight: Double
                if x <= 1 {
                    weight = (a + 2) * x * x * x - (a + 3) * x * x + 1
                } else if x < 2 {
                    weight = a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a
                } else {
                    weight = 0
                }
                return (min(max(sample, 0), sourceCount - 1), Float(weight))
            }
        }

        let xWeights = (0..<w).map { weights($0, sourceCount: sourceW, outputCount: w) }
        let yWeights = (0..<h).map { weights($0, sourceCount: sourceH, outputCount: h) }
        var horizontal = [Float](repeating: 0, count: sourceH * w * 3)
        for row in 0..<sourceH {
            for column in 0..<w {
                let destination = (row * w + column) * 3
                for (sample, weight) in xWeights[column] {
                    let origin = (row * sourceW + sample) * 3
                    for channel in 0..<3 {
                        horizontal[destination + channel] += Float(source[origin + channel]) * weight
                    }
                }
            }
        }
        var vertical = [Float](repeating: 0, count: h * w * 3)
        for row in 0..<h {
            for column in 0..<w {
                let destination = (row * w + column) * 3
                for (sample, weight) in yWeights[row] {
                    let origin = (sample * w + column) * 3
                    for channel in 0..<3 {
                        vertical[destination + channel] += horizontal[origin + channel] * weight
                    }
                }
            }
        }
        return vertical.map { UInt8(clamping: Int($0.rounded(.toNearestOrEven))) }
    }

    /// Normalized image as row-major `(H, W, 3)` — for parity debugging.
    public func normalizedHWC(_ cgImage: CGImage) throws -> [Float] {
        let S = size, bpr = S * 4
        let rgba = try drawRGBA(cgImage)
        var out = [Float](repeating: 0, count: S * S * 3)
        for h in 0..<S { for w in 0..<S { for c in 0..<3 {
            out[(h * S + w) * 3 + c] = Float(rgba[h * bpr + w * 4 + c]) / 127.5 - 1.0
        }}}
        return out
    }

    /// Draw into an SxS RGBA8 buffer and return a COPY (the CGContext owns the backing store,
    /// which is freed when ctx deinits — never return a pointer into it).
    private func drawRGBA(_ cgImage: CGImage) throws -> [UInt8] {
        let S = size, bytesPerRow = S * 4
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ImageError.context
        }
        ctx.interpolationQuality = .high
        // No CTM flip: CGContext into this RGBA buffer already yields top-down rows matching
        // PIL's row order (verified by corner-pixel parity).
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: S, height: S))
        guard let dataPtr = ctx.data, ctx.bytesPerRow == bytesPerRow else { throw ImageError.context }
        let buf = dataPtr.bindMemory(to: UInt8.self, capacity: S * bytesPerRow)
        return Array(UnsafeBufferPointer(start: buf, count: S * bytesPerRow))
    }

    /// Flattened `pixel_values` (row-major `(numPatches, featuresPerPatch)`).
    public func pixelValues(_ cgImage: CGImage) throws -> [Float] {
        let S = size
        let bytesPerRow = S * 4
        let rgba = try drawRGBA(cgImage)

        let GH = S / (patch * merge), GW = S / (patch * merge)
        let FPP = featuresPerPatch
        var out = [Float](repeating: 0, count: numPatches * FPP)
        for gh in 0..<GH {
            for gw in 0..<GW {
                for mh in 0..<merge {
                    for mw in 0..<merge {
                        let patchIdx = ((gh * GW + gw) * merge + mh) * merge + mw
                        let base = patchIdx * FPP
                        for c in 0..<3 {
                            for ph in 0..<patch {
                                let H = gh * (merge * patch) + mh * patch + ph
                                let row = H * bytesPerRow
                                for pw in 0..<patch {
                                    let W = gw * (merge * patch) + mw * patch + pw
                                    let v = Float(rgba[row + W * 4 + c]) / 127.5 - 1.0
                                    for t in 0..<temporal {
                                        out[base + ((c * temporal + t) * patch + ph) * patch + pw] = v
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
