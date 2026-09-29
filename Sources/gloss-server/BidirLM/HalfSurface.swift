import CoreML
import CoreVideo
import Foundation

/// Reusable FP16 storage. CPU access is scoped to a pixel-buffer lock, never a retained
/// MLMultiArray.dataPointer, so an output backing is unlocked when Core ML receives it.
final class HalfSurface {
    let shape: [Int]
    let strides: [Int]
    let pixelBuffer: CVPixelBuffer
    let array: MLMultiArray

    init(_ shape: [Int], zeroed: Bool = false) throws {
        guard shape.count == 4, shape.allSatisfy({ $0 > 0 }) else {
            throw BidirLMBundle.Failure.invalid("FP16 surface requires a positive rank-four shape")
        }
        self.shape = shape
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, shape[3], shape[0] * shape[1] * shape[2],
                                        kCVPixelFormatType_OneComponent16Half,
                                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw BidirLMBundle.Failure.invalid("could not allocate FP16 surface: \(status)")
        }
        self.pixelBuffer = pixelBuffer
        let row = CVPixelBufferGetBytesPerRow(pixelBuffer) / MemoryLayout<Float16>.stride
        strides = [shape[1] * shape[2] * row, shape[2] * row, row, 1]
        array = MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map { NSNumber(value: $0) })
        if zeroed {
            try withMutable { pointer, _ in
                pointer.initialize(repeating: 0, count: CVPixelBufferGetDataSize(pixelBuffer) / 2)
            }
        }
    }

    func withMutable<T>(_ body: (UnsafeMutablePointer<Float16>, [Int]) throws -> T) throws -> T {
        let status = CVPixelBufferLockBaseAddress(pixelBuffer, [])
        guard status == kCVReturnSuccess else {
            throw BidirLMBundle.Failure.invalid("could not lock FP16 surface: \(status)")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw BidirLMBundle.Failure.invalid("FP16 surface has no base address")
        }
        return try body(base.assumingMemoryBound(to: Float16.self), strides)
    }

    func withBuffer<T>(_ body: (UnsafePointer<Float16>, [Int]) throws -> T) throws -> T {
        let status = CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        guard status == kCVReturnSuccess else {
            throw BidirLMBundle.Failure.invalid("could not read FP16 surface: \(status)")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw BidirLMBundle.Failure.invalid("FP16 surface has no base address")
        }
        return try body(UnsafePointer(base.assumingMemoryBound(to: Float16.self)), strides)
    }

    func copy(from source: MLMultiArray) throws {
        try HalfArrays.check(source, shape, "surface source")
        try source.withUnsafeBufferPointer(ofType: Float16.self) { src in
            let ss = source.strides.map(\.intValue)
            try withMutable { dst, ds in
                for a in 0..<shape[0] {
                    for b in 0..<shape[1] {
                        for c in 0..<shape[2] {
                            let si = a * ss[0] + b * ss[1] + c * ss[2]
                            let di = a * ds[0] + b * ds[1] + c * ds[2]
                            for d in 0..<shape[3] { dst[di + d * ds[3]] = src[si + d * ss[3]] }
                        }
                    }
                }
            }
        }
    }

    /// A declined backing is copied explicitly, never mistaken for a populated buffer.
    @discardableResult
    static func predict(_ model: MLModel, feeds: [String: MLMultiArray],
                        outputs: [String: HalfSurface]) throws -> Int {
        guard !outputs.values.contains(where: { output in feeds.values.contains { $0 === output.array } }) else {
            throw BidirLMBundle.Failure.invalid("Core ML inputs and output backings must not alias")
        }
        let options = MLPredictionOptions()
        options.outputBackings = outputs.mapValues(\.array)
        let result = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: feeds.mapValues { MLFeatureValue(multiArray: $0) }),
            options: options)
        var declined = 0
        for (name, backing) in outputs {
            let actual = try HalfArrays.output(result, name, backing.shape)
            if actual !== backing.array {
                declined += 1
                try backing.copy(from: actual)
            }
        }
        return declined
    }
}

/// One completed layer's K/V bank, already in the contractions' layouts. A separate bank is
/// written for the following layer; neither head nor query dispatch copies a whole bank.
final class StreamedKVBank {
    let block: Int
    let heads: Int
    let dim: Int
    let keys: [[HalfSurface]]
    let values: [[HalfSurface]]
    let masks: [HalfSurface]

    init(keys count: Int, block: Int, valid: [Bool], heads: Int = 8, dim: Int = 128) throws {
        guard count > 0, block > 0, heads > 0, dim > 0, count.isMultiple(of: block), valid.count == count else {
            throw BidirLMBundle.Failure.invalid("invalid streamed K/V geometry")
        }
        self.block = block
        self.heads = heads
        self.dim = dim
        keys = try (0..<(count / block)).map { _ in
            try (0..<heads).map { _ in try HalfSurface([1, block, 1, dim], zeroed: true) }
        }
        values = try (0..<(count / block)).map { _ in
            try (0..<heads).map { _ in try HalfSurface([1, dim, 1, block], zeroed: true) }
        }
        masks = try (0..<(count / block)).map { b in
            let mask = try HalfSurface([1, block, 1, 1], zeroed: true)
            try mask.withMutable { p, s in
                for t in 0..<block { p[t * s[1]] = valid[b * block + t] ? 1 : 0 }
            }
            return mask
        }
    }

    func store(start: Int, key: HalfSurface, value: HalfSurface) throws {
        let width = key.shape[3], b = start / block, offset = start % block
        guard key.shape == [1, heads, dim, width], value.shape == key.shape,
              start >= 0, b < keys.count, offset + width <= block else {
            throw BidirLMBundle.Failure.invalid("projection does not fit streamed K/V bank")
        }
        try key.withBuffer { src, ss in
            for h in 0..<heads {
                try keys[b][h].withMutable { dst, ds in
                    for d in 0..<dim {
                        for t in 0..<width { dst[(offset + t) * ds[1] + d] = src[h * ss[1] + d * ss[2] + t] }
                    }
                }
            }
        }
        try value.withBuffer { src, ss in
            for h in 0..<heads {
                try values[b][h].withMutable { dst, ds in
                    for d in 0..<dim {
                        (dst + d * ds[1] + offset).update(from: src + h * ss[1] + d * ss[2], count: width)
                    }
                }
            }
        }
    }

    func feeds(block b: Int, query: HalfSurface) -> [String: MLMultiArray] {
        var feeds = ["query": query.array, "valid": masks[b].array]
        for h in 0..<heads {
            feeds["key_\(h)"] = keys[b][h].array
            feeds["value_\(h)"] = values[b][h].array
        }
        return feeds
    }
}
