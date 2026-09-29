import CoreML
import Foundation

/// FP16 `MLMultiArray` helpers. Core ML outputs (especially from the Neural Engine) may carry
/// padded strides, so every read and copy honours the array's declared strides.
enum HalfArrays {
    enum Failure: Error, CustomStringConvertible {
        case unexpectedType(String, MLMultiArrayDataType)
        case unexpectedShape(String, [Int], [Int])
        case missingOutput(String)
        case nonFinite(String)

        var description: String {
            switch self {
            case let .unexpectedType(label, type): "\(label) has data type \(type.rawValue), expected float16"
            case let .unexpectedShape(label, actual, expected): "\(label) has shape \(actual), expected \(expected)"
            case let .missingOutput(name): "model produced no \(name) output"
            case let .nonFinite(label): "\(label) contains non-finite values"
            }
        }
    }

    /// Zero-filled contiguous FP16 array (MLMultiArray memory is not initialized).
    static func zeros(_ shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        let count = shape.reduce(1, *)
        array.dataPointer.initializeMemory(as: UInt16.self, repeating: 0, count: count)
        return array
    }

    static func pointer(_ array: MLMultiArray) -> UnsafeMutablePointer<Float16> {
        array.dataPointer.assumingMemoryBound(to: Float16.self)
    }

    static func check(_ array: MLMultiArray, _ shape: [Int], _ label: String) throws {
        guard array.dataType == .float16 else { throw Failure.unexpectedType(label, array.dataType) }
        let actual = array.shape.map(\.intValue)
        guard actual == shape else { throw Failure.unexpectedShape(label, actual, shape) }
    }

    static func output(_ provider: MLFeatureProvider, _ name: String, _ shape: [Int]) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw Failure.missingOutput(name)
        }
        try check(array, shape, name)
        return array
    }

    /// Copy `source[.., .., row, 0..<width]` rows of a rank-4 array into `destination` at a
    /// column offset: used to place a chunk's keys/values into a key block.
    static func copyColumns(from source: MLMultiArray, into destination: MLMultiArray, columnOffset: Int) {
        let s = source.shape.map(\.intValue), ss = source.strides.map(\.intValue)
        let ds = destination.strides.map(\.intValue)
        let src = pointer(source), dst = pointer(destination)
        let width = s[3]
        for a in 0..<s[1] {
            for b in 0..<s[2] {
                let from = src + a * ss[1] + b * ss[2]
                let to = dst + a * ds[1] + b * ds[2] + columnOffset * ds[3]
                if ss[3] == 1, ds[3] == 1 {
                    to.update(from: from, count: width)
                } else {
                    for w in 0..<width { to[w * ds[3]] = from[w * ss[3]] }
                }
            }
        }
    }

    /// Column `column` of a (1, 2048, W) or (1, 2048, 1, W) array as Float.
    static func column(_ array: MLMultiArray, _ column: Int) throws -> [Float] {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let channelAxis = 1
        let widthAxis = shape.count - 1
        let base = pointer(array)
        var out = [Float](repeating: 0, count: shape[channelAxis])
        for c in 0..<shape[channelAxis] {
            out[c] = Float(base[c * strides[channelAxis] + column * strides[widthAxis]])
        }
        guard out.allSatisfy(\.isFinite) else { throw Failure.nonFinite("embedding column \(column)") }
        return out
    }

    /// Columns `0..<count` of a (1, D, 1, W) array as row-major FP16 rows (count x D).
    static func rows(_ array: MLMultiArray, count: Int) throws -> [Float16] {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let channels = shape[1], width = shape[shape.count - 1]
        precondition(count <= width)
        let base = pointer(array)
        let cs = strides[1], ws = strides[shape.count - 1]
        var out = [Float16](repeating: 0, count: count * channels)
        out.withUnsafeMutableBufferPointer { dst in
            for c in 0..<channels {
                let column = base + c * cs
                for t in 0..<count { dst[t * channels + c] = column[t * ws] }
            }
        }
        guard out.allSatisfy({ $0.isFinite }) else { throw Failure.nonFinite("encoder features") }
        return out
    }
}
