import CoreML
import Foundation

/// Check the runtime result, not only the converter's metadata, before reading its raw storage.
enum CoreMLArrayReader {
    enum ArrayError: Error, CustomStringConvertible {
        case invalid(String)

        var description: String {
            switch self {
            case let .invalid(reason): reason
            }
        }
    }

    static func float32(_ array: MLMultiArray, shape: [Int], label: String) throws -> [Float] {
        guard array.dataType == .float32, array.shape.map(\.intValue) == shape else {
            throw ArrayError.invalid(
                "\(label) must be Float32 \(shape); got \(array.dataType) \(array.shape)")
        }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !product.overflow else {
                throw ArrayError.invalid("\(label) has an invalid expected shape")
            }
            count = product.partialValue
        }
        guard array.count == count else {
            throw ArrayError.invalid("\(label) has \(array.count) values; expected \(count)")
        }
        var stride = 1
        for index in shape.indices.reversed() {
            guard array.strides[index].intValue == stride else {
                throw ArrayError.invalid("\(label) has a non-contiguous Core ML layout")
            }
            stride *= shape[index]
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    static func fillFloat32(
        _ array: MLMultiArray,
        with values: [Float],
        label: String
    ) throws {
        guard array.dataType == .float32, values.count <= array.count else {
            throw ArrayError.invalid("\(label) has more values than its Float32 Core ML input")
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        pointer.update(repeating: 0, count: array.count)
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                pointer.update(from: base, count: source.count)
            }
        }
    }
}
