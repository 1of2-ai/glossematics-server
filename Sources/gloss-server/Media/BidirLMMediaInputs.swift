import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

/// Host-side media preprocessing matching the pinned upstream processor
/// (`BidirLMOmniProcessor`: Qwen2-VL fast image processor + Whisper feature extractor).
enum BidirLMMediaInputs {
    enum Failure: Error, CustomStringConvertible {
        case invalid(String)

        var description: String {
            switch self {
            case let .invalid(reason): reason
            }
        }
    }

    // MARK: - images

    /// Decode, convert to RGB (alpha composited on white, no color management, like PIL),
    /// smart-resize to the 32-pixel grid, resample exactly like the processor, normalize to
    /// [-1, 1], and patchify into rows of 1536 values in 2x2 merge-window order.
    static func image(_ data: Data, config v: BidirLMManifest.Vision) throws -> PreparedImage {
        let decoded: CGImage
        do {
            decoded = try GlossImagePreprocessor.loadCGImage(data)
        } catch GlossImagePreprocessor.ImageError.sourceTooLargePixels(let pixels) {
            throw Failure.invalid("image has \(pixels) pixels; the maximum source size is 40 megapixels")
        } catch {
            throw Failure.invalid("image data could not be decoded (JPEG, PNG, or WebP expected)")
        }
        let (rgb, height, width) = try rawRGB(decoded)
        let factor = v.patchSize * v.mergeSize
        let (h, w) = try smartResize(height: height, width: width, factor: factor,
                                     minPixels: v.minPixels, maxPixels: v.maxPixels)
        guard (h / v.patchSize) * (w / v.patchSize) <= BidirLMContract.visionMaxPatches else {
            throw Failure.invalid("image resizes to more than \(BidirLMContract.visionMaxPatches) patches")
        }
        let resized = BicubicResampler.resize(rgb, height: height, width: width, toHeight: h, toWidth: w)
        let (pixels, gh, gw) = try GlossImagePreprocessor(minPixels: v.minPixels, maxPixels: v.maxPixels)
            .pixelValues(rgb: resized, h: h, w: w)
        return PreparedImage(pixels: pixels, gridH: gh, gridW: gw)
    }

    /// Qwen2-VL `smart_resize` (Python semantics: banker's rounding in the first step).
    static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) throws -> (Int, Int) {
        guard height > 0, width > 0 else { throw Failure.invalid("image has no pixels") }
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw Failure.invalid("image aspect ratio must be at most 200:1")
        }
        let f = Double(factor)
        var hBar = Int((Double(height) / f).rounded(.toNearestOrEven)) * factor
        var wBar = Int((Double(width) / f).rounded(.toNearestOrEven)) * factor
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((Double(height) / beta / f).rounded(.down)) * factor)
            wBar = max(factor, Int((Double(width) / beta / f).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = Int((Double(height) * beta / f).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / f).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    /// 8-bit RGB (row-major, h*w*3) without color conversion: RGB images are drawn in their own
    /// color space, grayscale in its own gray space and replicated. Transparent pixels are
    /// composited onto white, as the processor's `convert_rgb` does.
    static func rawRGB(_ image: CGImage) throws -> ([UInt8], Int, Int) {
        let w = image.width, h = image.height
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        let model = image.colorSpace?.model
        if model == .monochrome, let space = image.colorSpace {
            guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                throw Failure.invalid("image could not be converted to RGB")
            }
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(rect)
            ctx.draw(image, in: rect)
            guard let base = ctx.data else { throw Failure.invalid("image could not be converted to RGB") }
            let gray = base.bindMemory(to: UInt8.self, capacity: ctx.bytesPerRow * h)
            var out = [UInt8](repeating: 0, count: w * h * 3)
            for y in 0..<h {
                for x in 0..<w {
                    let v = gray[y * ctx.bytesPerRow + x]
                    let d = (y * w + x) * 3
                    out[d] = v; out[d + 1] = v; out[d + 2] = v
                }
            }
            return (out, h, w)
        }
        let space: CGColorSpace
        if model == .rgb, let own = image.colorSpace {
            space = own
        } else if model == .indexed, let baseSpace = image.colorSpace?.baseColorSpace, baseSpace.model == .rgb {
            space = baseSpace
        } else {
            space = CGColorSpace(name: CGColorSpace.sRGB)!
        }
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw Failure.invalid("image could not be converted to RGB")
        }
        ctx.setFillColor(CGColor(colorSpace: space, components: [1, 1, 1, 1]) ?? CGColor(gray: 1, alpha: 1))
        ctx.fill(rect)
        ctx.interpolationQuality = .none
        ctx.draw(image, in: rect)
        guard let base = ctx.data else { throw Failure.invalid("image could not be converted to RGB") }
        let rgba = base.bindMemory(to: UInt8.self, capacity: ctx.bytesPerRow * h)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let row = y * ctx.bytesPerRow
            for x in 0..<w {
                let s = row + x * 4, d = (y * w + x) * 3
                out[d] = rgba[s]; out[d + 1] = rgba[s + 1]; out[d + 2] = rgba[s + 2]
            }
        }
        return (out, h, w)
    }

    // MARK: - audio

    /// Minimum clip length: one 25 ms analysis window plus the reflect padding.
    static let minimumSamples = 1_600

    /// WAV bytes -> 16 kHz mono samples. Channels are averaged; other sample rates are
    /// converted with AVAudioConverter (the reference resamples with librosa, so non-16 kHz
    /// input carries a small resampler difference; 16 kHz input is exact).
    static func samples16k(_ data: Data, maximumSeconds: Double) throws -> [Float] {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-audio-\(UUID().uuidString).wav")
        try data.write(to: url, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: url) }
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw Failure.invalid("audio data could not be decoded as WAV")
        }
        let format = file.processingFormat
        guard format.sampleRate >= 8_000, format.sampleRate <= 192_000, format.channelCount >= 1 else {
            throw Failure.invalid("unsupported WAV format (\(format.sampleRate) Hz, \(format.channelCount) channels)")
        }
        guard Double(file.length) / format.sampleRate <= maximumSeconds else {
            throw Failure.invalid("audio is longer than \(Int(maximumSeconds)) seconds")
        }
        guard file.length > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw Failure.invalid("audio has no samples")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw Failure.invalid("audio could not be decoded") }
        let count = Int(buffer.frameLength), channelCount = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for c in 0..<channelCount {
            let ptr = channels[c]
            for i in 0..<count { mono[i] += ptr[i] }
        }
        if channelCount > 1 {
            let scale = 1 / Float(channelCount)
            for i in 0..<count { mono[i] *= scale }
        }
        if format.sampleRate == 16_000 { return mono }
        return try resample(mono, from: format.sampleRate)
    }

    private static func resample(_ mono: [Float], from rate: Double) throws -> [Float] {
        guard let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
              let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: output),
              let source = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(mono.count)),
              let chunk = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: 16_384) else {
            throw Failure.invalid("audio could not be resampled")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        source.frameLength = AVAudioFrameCount(mono.count)
        mono.withUnsafeBufferPointer { source.floatChannelData![0].update(from: $0.baseAddress!, count: mono.count) }
        var fed = false
        var out = [Float]()
        out.reserveCapacity(Int(Double(mono.count) * 16_000 / rate) + 16)
        while true {
            chunk.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: chunk, error: &error) { _, state in
                if fed { state.pointee = .endOfStream; return nil }
                fed = true
                state.pointee = .haveData
                return source
            }
            if error != nil || status == .error { throw Failure.invalid("audio could not be resampled") }
            if chunk.frameLength > 0 {
                out.append(contentsOf: UnsafeBufferPointer(start: chunk.floatChannelData![0], count: Int(chunk.frameLength)))
            }
            if status == .endOfStream || (status == .inputRanDry && chunk.frameLength == 0) { break }
        }
        return out
    }

    /// Whisper log-mel features for the whole clip (no 30-second padding or truncation):
    /// centered STFT with reflect padding, `samples / 160` frames, clip-level dynamic range.
    static func audio(_ samples: [Float], frontend: GlossMelFrontend) throws -> PreparedAudio {
        guard samples.count >= minimumSamples else {
            throw Failure.invalid("audio must be at least \(minimumSamples / 16) ms long")
        }
        // The length guard above already exceeds the frontend's own minimum, so a throw here is a
        // defensive backstop: either way the caller gets the invalid-input path (HTTP 400), never a
        // crash or a generic server error.
        let mel: [Float], frames: Int
        do {
            (mel, frames) = try frontend.wholeClipLogMel(samples)
        } catch GlossMelFrontend.MelError.audioTooShort {
            throw Failure.invalid("audio must be at least \(minimumSamples / 16) ms long")
        } catch {
            throw Failure.invalid("audio could not be converted to mel features: \(error)")
        }
        return PreparedAudio(mel: mel, frames: frames)
    }
}

/// Separable antialiased bicubic resampling of 8-bit RGB, bit-exact with the processor's
/// torchvision path (`F.resize(uint8, BICUBIC, antialias=True)`): Keys a = -0.5 filter with
/// support scaled by the downsampling factor, weights normalized per output pixel and
/// quantized to int16 with the widest precision that fits, horizontal pass then vertical,
/// each rounded and clamped to 8 bits.
enum BicubicResampler {
    private static func filter(_ x: Double) -> Double {
        let a = -0.5, t = abs(x)
        if t < 1 { return ((a + 2) * t - (a + 3)) * t * t + 1 }
        if t < 2 { return (((t - 5) * t + 8) * t - 4) * a }
        return 0
    }

    struct Coefficients {
        let starts: [Int]
        let weights: [[Int32]]
        let precision: Int
    }

    static func coefficients(input: Int, output: Int) -> Coefficients {
        let scale = Double(input) / Double(output)
        let support = scale >= 1 ? 2 * scale : 2
        let inverse = scale >= 1 ? 1 / scale : 1
        var starts = [Int](), floats = [[Double]]()
        var maxWeight = 0.0
        for i in 0..<output {
            let center = scale * (Double(i) + 0.5)
            let lo = max(Int(center - support + 0.5), 0)
            let size = min(Int(center + support + 0.5), input) - lo
            var w = (0..<size).map { filter((Double($0 + lo) - center + 0.5) * inverse) }
            let total = w.reduce(0, +)
            if total != 0 { w = w.map { $0 / total } }
            maxWeight = max(maxWeight, w.max() ?? 0)
            starts.append(lo)
            floats.append(w)
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + maxWeight * Double(1 << (precision + 1))) >= (1 << 15) { break }
            precision += 1
        }
        let unit = Double(1 << precision)
        let weights = floats.map { row in row.map { $0 < 0 ? Int32(-0.5 + $0 * unit) : Int32(0.5 + $0 * unit) } }
        return Coefficients(starts: starts, weights: weights, precision: precision)
    }

    static func resize(_ rgb: [UInt8], height: Int, width: Int, toHeight: Int, toWidth: Int) -> [UInt8] {
        var image = rgb, w = width
        if toWidth != width {
            let c = coefficients(input: width, output: toWidth)
            var out = [UInt8](repeating: 0, count: height * toWidth * 3)
            let round = Int32(1 << (c.precision - 1))
            for y in 0..<height {
                let row = y * width * 3
                for x in 0..<toWidth {
                    let lo = c.starts[x], ws = c.weights[x]
                    for ch in 0..<3 {
                        var acc = round
                        for (j, wt) in ws.enumerated() { acc &+= Int32(image[row + (lo + j) * 3 + ch]) &* wt }
                        out[(y * toWidth + x) * 3 + ch] = UInt8(clamping: max(0, min(255, acc >> Int32(c.precision))))
                    }
                }
            }
            image = out
            w = toWidth
        }
        if toHeight != height {
            let c = coefficients(input: height, output: toHeight)
            var out = [UInt8](repeating: 0, count: toHeight * w * 3)
            let round = Int32(1 << (c.precision - 1))
            let stride = w * 3
            for y in 0..<toHeight {
                let lo = c.starts[y], ws = c.weights[y]
                for i in 0..<stride {
                    var acc = round
                    for (j, wt) in ws.enumerated() { acc &+= Int32(image[(lo + j) * stride + i]) &* wt }
                    out[y * stride + i] = UInt8(clamping: max(0, min(255, acc >> Int32(c.precision))))
                }
            }
            image = out
        }
        return image
    }
}
