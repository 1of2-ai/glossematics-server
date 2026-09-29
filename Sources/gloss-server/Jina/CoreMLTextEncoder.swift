import CoreML
import Foundation

/// Low-level Core ML wrapper for a converted text tower bucket (`text_b{S}.mlpackage` /
/// `text_multifunc.mlpackage` function).
///
/// The wrapper adapts to the model's DECLARED inputs, so one implementation serves both
/// architecture families:
///   - causal Qwen3 towers: `position_ids` is (3,1,S) M-RoPE and there is no mask input
///     (causal + right padding makes pads invisible);
///   - bidirectional EuroBERT/Llama towers: `position_ids` is (1,S) and an `attention_mask`
///     (1,S) input (1=real, 0=pad) is required, since pads are otherwise visible to real tokens.
/// Output: `embedding` (1,D) L2-normed, pooled at the `selector` one-hot (last real token).
internal final class CoreMLTextEncoder {
    /// Fills unused tail rows of `input_ids`. The value is semantically irrelevant — causal towers
    /// never attend to it, bidirectional towers mask it — but it must be a valid vocab id, so any
    /// non-negative id works; bundles pass their real pad id for cleanliness.
    public let padTokenID: Int32

    public let model: MLModel
    public let seqLen: Int
    public let batchSize: Int
    public let embeddingDim: Int
    /// The compiled `.mlmodelc` URL (reuse to avoid recompiling the same package per function).
    public let compiledURL: URL
    /// Whether `position_ids` is the (3,1,S) M-RoPE layout (true) or plain (1,S) (false).
    public let usesMRoPEPositions: Bool
    /// Whether the model declares an `attention_mask` input (bidirectional towers).
    public let requiresAttentionMask: Bool

    public init(
        modelURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        functionName: String? = nil,
        padTokenID: Int32 = 0
    ) throws {
        self.padTokenID = padTokenID
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        if let functionName { config.functionName = functionName }
        let compiled = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.compiledURL = compiled
        self.model = try MLModel(contentsOf: compiled, configuration: config)

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let idsConstraint = inputs["input_ids"]?.multiArrayConstraint else {
            throw EncoderError.badModel("missing input_ids constraint")
        }
        guard idsConstraint.dataType == .int32,
              idsConstraint.shape.count == 2,
              idsConstraint.shape.allSatisfy({ $0.intValue > 0 }) else {
            throw EncoderError.badModel("input_ids must be a positive Int32 [batch, sequence] array")
        }
        self.seqLen = idsConstraint.shape[1].intValue
        self.batchSize = idsConstraint.shape[0].intValue
        guard let out = model.modelDescription.outputDescriptionsByName["embedding"]?.multiArrayConstraint,
              out.dataType == .float32,
              out.shape.count == 2,
              out.shape[0].intValue == batchSize,
              let dim = out.shape.last?.intValue,
              dim > 0 else {
            throw EncoderError.badModel("missing embedding output constraint")
        }
        self.embeddingDim = dim
        self.usesMRoPEPositions = (inputs["position_ids"]?.multiArrayConstraint?.shape.count ?? 2) == 3
        self.requiresAttentionMask = inputs["attention_mask"] != nil
    }

    public enum EncoderError: Error { case badModel(String), tooLong(Int, Int), noOutput, batchMismatch(Int, Int) }

    /// Encode a real (unpadded) token id sequence into an L2-normalized embedding.
    public func encode(tokenIds: [Int32]) throws -> [Float] {
        try encodeBatch(tokenIds: [tokenIds])[0]
    }

    /// Encode a batch of real token id sequences in ONE forward pass (batch-B functions in the
    /// multi-function package). Amortizes the memory-bound weight stream — measured ~3.2×
    /// throughput at B=4 on short buckets. Rows are independent (per-row mask + selector), so
    /// results match the single-row path. `tokenIds.count` must equal the model's batch size.
    public func encodeBatch(tokenIds rows: [[Int32]]) throws -> [[Float]] {
        let S = seqLen, B = batchSize
        guard rows.count == B else { throw EncoderError.batchMismatch(rows.count, B) }
        for r in rows {
            guard r.count <= S, !r.isEmpty else { throw EncoderError.tooLong(r.count, S) }
        }

        let ids = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: S)], dataType: .int32)
        let sel = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: S)], dataType: .float32)
        let idsPtr = ids.dataPointer.bindMemory(to: Int32.self, capacity: B * S)
        let selPtr = sel.dataPointer.bindMemory(to: Float.self, capacity: B * S)
        for (b, row) in rows.enumerated() {
            let last = row.count - 1
            for i in 0..<S {
                idsPtr[b * S + i] = i < row.count ? row[i] : padTokenID
                selPtr[b * S + i] = (i == last) ? 1.0 : 0.0
            }
        }

        var features: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "selector": MLFeatureValue(multiArray: sel),
        ]

        if usesMRoPEPositions {
            let pos = try MLMultiArray(shape: [3, NSNumber(value: B), NSNumber(value: S)], dataType: .int32)
            let pp = pos.dataPointer.bindMemory(to: Int32.self, capacity: 3 * B * S)
            for i in 0..<(B * S) {
                let v = Int32(i % S)
                pp[i] = v; pp[B * S + i] = v; pp[2 * B * S + i] = v
            }
            features["position_ids"] = MLFeatureValue(multiArray: pos)
        } else {
            let pos = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: S)], dataType: .int32)
            let pp = pos.dataPointer.bindMemory(to: Int32.self, capacity: B * S)
            for i in 0..<(B * S) { pp[i] = Int32(i % S) }
            features["position_ids"] = MLFeatureValue(multiArray: pos)
        }

        if requiresAttentionMask {
            let mask = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: S)], dataType: .float32)
            let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: B * S)
            for (b, row) in rows.enumerated() {
                for i in 0..<S { mp[b * S + i] = i < row.count ? 1.0 : 0.0 }
            }
            features["attention_mask"] = MLFeatureValue(multiArray: mask)
        }

        let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: features))
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        let dim = embeddingDim
        let flat = try CoreMLArrayReader.float32(
            arr, shape: [B, dim], label: "text embedding")
        return (0..<B).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    }
}

/// L2-normalized truncation for Matryoshka dims: `normalize(full[:dim])`.
internal func matryoshka(_ embedding: [Float], dim: Int) -> [Float] {
    let d = min(dim, embedding.count)
    var v = Array(embedding[0..<d])
    var norm: Float = 0
    for x in v { norm += x * x }
    norm = max(norm.squareRoot(), 1e-12)
    for i in 0..<d { v[i] /= norm }
    return v
}

internal func cosine(_ a: [Float], _ b: [Float]) -> Double {
    let n = min(a.count, b.count)
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<n { dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i]) }
    return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : .nan
}
