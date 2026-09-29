import CoreML
import Foundation

/// General media decoder: the unified `embed_multifunc` (input_ids -> inputs_embeds) plus
/// `decoder_embeds_multifunc` (inputs_embeds + position_ids + selector -> L2-normed embedding),
/// each a multi-function package over sequence-length buckets.
///
/// The host builds `input_ids = [prefix, <media_pad>*L, suffix]` right-padded to the chosen S,
/// embeds them, scatters the L real media features (image/audio/...) into the media positions,
/// then decodes with a selector at the real last token. This is the single decoder shared by
/// text/image/audio/omni (replaces fixed per-size decoders).
///
/// The wrapper adapts per loaded function to the model's DECLARED inputs, serving both tower
/// families: causal Qwen3 decoders take (3,1,S) M-RoPE positions and no mask (right padding is
/// invisible), while bidirectional EuroBERT/Llama decoders take (1,S) positions and an
/// `attention_mask` (1,S) input that the host fills 1=real / 0=pad.
internal final class GeneralMediaDecoder {
    /// Conversion-pipeline bucket convention; bundles override via `manifest.decoder.sequenceBuckets`.
    public static let defaultSequenceBuckets = [128, 256, 512, 1024]

    let embedCompiled: URL
    let decoderCompiled: URL
    /// `nil` = ADAPTIVE placement (measured optimal on small): ANE for small sequences, GPU for
    /// large. The decoder is shallow enough to be accurate on either, so placement is a pure latency
    /// choice and there's a crossover near S≈384 (ANE: 13/30/85/231 ms vs GPU: 18/31/51/132 ms at
    /// S=128/256/512/1024). A non-nil value forces all buckets onto that unit.
    let forcedUnits: MLComputeUnits?
    let featDim: Int
    /// Fills unused tail rows of `input_ids`. Semantically irrelevant (causal towers never attend to
    /// it; bidirectional towers mask it) but must be a valid vocab id; bundles pass their real pad id.
    let padTokenID: Int32
    let buckets: [Int]
    // Per-S loaded functions (loading is the expensive part; cache once used). `lock` guards the
    // lazy caches so a shared decoder is safe under concurrent embed() calls (predictions run
    // outside the lock — MLModel.prediction is itself thread-safe).
    let lock = NSLock()
    var embedModels: [Int: MLModel] = [:]
    var decoderModels: [Int: LoadedDecoder] = [:]

    /// A loaded decoder function plus its declared input contract.
    struct LoadedDecoder {
        let model: MLModel
        let usesMRoPEPositions: Bool      // position_ids (3,1,S) vs (1,S)
        let requiresAttentionMask: Bool   // bidirectional towers declare attention_mask
    }

    public init(embedModelURL: URL, decoderModelURL: URL,
                computeUnits: MLComputeUnits? = nil, featDim: Int = 1024,
                padTokenID: Int32 = 0,
                sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        guard featDim > 0,
              !sequenceBuckets.isEmpty,
              sequenceBuckets.allSatisfy({ $0 > 0 }) else {
            throw DecoderError.invalidInput("decoder dimensions and buckets must be positive")
        }
        self.forcedUnits = computeUnits
        self.featDim = featDim
        self.padTokenID = padTokenID
        self.buckets = sequenceBuckets.sorted()
        embedCompiled = embedModelURL.pathExtension == "mlmodelc"
            ? embedModelURL : try MLModel.compileModel(at: embedModelURL)
        decoderCompiled = decoderModelURL.pathExtension == "mlmodelc"
            ? decoderModelURL : try MLModel.compileModel(at: decoderModelURL)
    }

    public enum DecoderError: Error {
        case noOutput, tooLong(Int), invalidInput(String)
    }

    public func sBucket(forSeq n: Int) -> Int {
        buckets.first { $0 >= n } ?? buckets.last!
    }

    /// Latency-optimal unit for a bucket: ANE ≤256, GPU ≥512 (measured crossover near S≈384).
    func units(forS S: Int) -> MLComputeUnits { forcedUnits ?? (S <= 256 ? .cpuAndNeuralEngine : .cpuAndGPU) }

    private func embedModel(_ S: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = embedModels[S] { return m }
        let cfg = MLModelConfiguration(); cfg.computeUnits = units(forS: S); cfg.functionName = "f\(S)"
        let m = try MLModel(contentsOf: embedCompiled, configuration: cfg)
        embedModels[S] = m; return m
    }

    private func decoderModel(_ S: Int) throws -> LoadedDecoder {
        lock.lock(); defer { lock.unlock() }
        if let d = decoderModels[S] { return d }
        let cfg = MLModelConfiguration(); cfg.computeUnits = units(forS: S); cfg.functionName = "f\(S)"
        let m = try MLModel(contentsOf: decoderCompiled, configuration: cfg)
        let inputs = m.modelDescription.inputDescriptionsByName
        let d = LoadedDecoder(
            model: m,
            usesMRoPEPositions: (inputs["position_ids"]?.multiArrayConstraint?.shape.count ?? 2) == 3,
            requiresAttentionMask: inputs["attention_mask"] != nil
        )
        decoderModels[S] = d; return d
    }

    /// `tokenIds` = the full real sequence (prefix + media pads + suffix), length ≤ S.
    /// `features` = (L * featDim) row-major, scattered into rows [scatterOffset, scatterOffset+L).
    /// Returns the L2-normalized embedding.
    public func decode(tokenIds: [Int32], features: [Float], scatterOffset: Int) throws -> [Float] {
        let realLen = tokenIds.count
        guard realLen > 0,
              !features.isEmpty,
              features.count.isMultiple(of: featDim),
              features.allSatisfy(\.isFinite) else {
            throw DecoderError.invalidInput("media decoder requires a nonempty, finite feature matrix")
        }
        let L = features.count / featDim
        guard scatterOffset >= 0,
              scatterOffset <= realLen,
              L <= realLen - scatterOffset else {
            throw DecoderError.invalidInput("media feature scatter range exceeds the token sequence")
        }
        let S = sBucket(forSeq: realLen)
        guard realLen <= S else { throw DecoderError.tooLong(realLen) }

        // 1) embed_multifunc: input_ids (1,S) -> inputs_embeds (1,S,featDim)
        let ids = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
        let idp = ids.dataPointer.bindMemory(to: Int32.self, capacity: S)
        for i in 0..<S { idp[i] = i < realLen ? tokenIds[i] : padTokenID }
        let embOut = try embedModel(S).prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": ids]))
        guard let embArr = embOut.featureValue(for: "out")?.multiArrayValue else {
            throw DecoderError.noOutput
        }
        let embedded = try CoreMLArrayReader.float32(
            embArr, shape: [1, S, featDim], label: "media token embeddings")

        // 2) copy embeds into a fresh (1,S,featDim) array and scatter the media features
        let embeds = try MLMultiArray(shape: [1, NSNumber(value: S), NSNumber(value: featDim)], dataType: .float32)
        let ep = embeds.dataPointer.bindMemory(to: Float.self, capacity: S * featDim)
        embedded.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                ep.update(from: base, count: source.count)
            }
        }
        features.withUnsafeBufferPointer { fb in
            guard let source = fb.baseAddress else { return }
            for t in 0..<L {
                let dst = (scatterOffset + t) * featDim
                ep.advanced(by: dst).update(from: source.advanced(by: t * featDim), count: featDim)
            }
        }

        // 3) decoder_embeds_multifunc: inputs_embeds + position_ids + selector [+ attention_mask]
        let decoder = try decoderModel(S)
        let sel = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .float32)
        let sp = sel.dataPointer.bindMemory(to: Float.self, capacity: S)
        let last = realLen - 1
        for i in 0..<S { sp[i] = (i == last) ? 1.0 : 0.0 }

        var inputs: [String: MLFeatureValue] = [
            "inputs_embeds": MLFeatureValue(multiArray: embeds),
            "selector": MLFeatureValue(multiArray: sel),
        ]
        if decoder.usesMRoPEPositions {
            let pos = try MLMultiArray(shape: [3, 1, NSNumber(value: S)], dataType: .int32)
            let pp = pos.dataPointer.bindMemory(to: Int32.self, capacity: 3 * S)
            for i in 0..<S { pp[i] = Int32(i); pp[S + i] = Int32(i); pp[2 * S + i] = Int32(i) }
            inputs["position_ids"] = MLFeatureValue(multiArray: pos)
        } else {
            let pos = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
            let pp = pos.dataPointer.bindMemory(to: Int32.self, capacity: S)
            for i in 0..<S { pp[i] = Int32(i) }
            inputs["position_ids"] = MLFeatureValue(multiArray: pos)
        }
        if decoder.requiresAttentionMask {
            let mask = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .float32)
            let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: S)
            for i in 0..<S { mp[i] = i < realLen ? 1.0 : 0.0 }
            inputs["attention_mask"] = MLFeatureValue(multiArray: mask)
        }

        let out = try decoder.model.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw DecoderError.noOutput
        }
        return try CoreMLArrayReader.float32(
            e, shape: [1, featDim], label: "media decoder embedding")
    }
}

/// Host-built masks for the runtime-masked audio encoder. Computed from a clip's real frame count
/// so a partial boundary chunk (and any fully-silent chunks beyond it) are masked out — giving exact
/// reference parity at ANY length, not just 200-frame multiples.
internal struct AudioMasks {
    public static let chunkSize = 200      // n_window*2 mel frames per chunk
    public static let tpc = 100            // attention tokens per chunk (after conv1 stride-2)
    public static let neg: Float = -1e4    // fp16-safe additive mask

    public let bucketFrames: Int           // F (a converted bucket >= exactFrames)
    public let chunks: Int                  // C = F/200
    public let convMask: [Float]            // (C,1,200) row-major
    public let attnBias: [Float]            // (1,1,T,T) row-major, T = C*tpc
    public let realTokens: Int              // pooled token count to keep (= reference length)

    /// Round up to the next converted frame bucket; extra chunks are fully masked.
    public static func bucket(forFrames n: Int, buckets: [Int] = AudioCoreMLEncoderMasked.frameBuckets) -> Int {
        buckets.first { $0 >= n } ?? buckets.last!
    }

    public init(exactFrames: Int, bucketFrames F: Int) {
        bucketFrames = F
        let cs = Self.chunkSize, tpc = Self.tpc
        let C = F / cs; chunks = C
        let T = C * tpc
        let fullChunks = exactFrames / cs
        let rem = exactFrames - fullChunks * cs

        var cm = [Float](repeating: 0, count: C * cs)
        var rtok = [Int](repeating: 0, count: C)
        for c in 0..<C {
            let real = c < fullChunks ? cs : (c == fullChunks ? rem : 0)
            if real > 0 { for i in 0..<real { cm[c * cs + i] = 1.0 } }
            rtok[c] = real > 0 ? (real - 1) / 2 + 1 : 0
        }
        convMask = cm

        var bias = [Float](repeating: Self.neg, count: T * T)
        for c in 0..<C {
            let r = rtok[c]; if r == 0 { continue }
            let base = c * tpc
            for i in 0..<tpc {
                let row = (base + i) * T
                for j in 0..<r { bias[row + base + j] = 0.0 }
            }
        }
        attnBias = bias
        // reference pooled-token count: num_pooled = ((after_conv1) - 2)/2 + 1, after_conv1=(n-1)/2+1
        let afterConv1 = (exactFrames - 1) / 2 + 1
        realTokens = (afterConv1 - 2) / 2 + 1
    }
}

/// Runtime-masked audio encoder (`audio_tower_masked_multifunc`): per-bucket f{F} taking
/// packed_mel (nMels,F) + conv_mask (C,1,200) + attn_bias (1,1,T,T), returning the full-layout
/// pooled features (C*50, 1024). With host masks this matches the reference at ANY length ≤ 16 s.
/// Runs on GPU (fp16 matmul accumulation).
internal final class AudioCoreMLEncoderMasked {
    /// audio_tower_masked_multifunc package functions: 2/4/8/16/32 s (32 s covers the model's
    /// 30 s / 3000-frame Whisper limit). Separate from AudioCoreMLEncoderMultifunction (the non-masked package).
    public static let frameBuckets = [200, 400, 800, 1600, 3200]
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    let lock = NSLock()   // guards the lazy cache for concurrent use
    var models: [Int: MLModel] = [:]

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.computeUnits = computeUnits
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput, invalidInput(String) }

    private func model(_ F: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[F] { return m }
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(F)"
        let m = try MLModel(contentsOf: compiledURL, configuration: cfg)
        models[F] = m; return m
    }

    /// Returns the full-layout features (C*50 * 1024) row-major; caller truncates to `realTokens`.
    public func encode(packedMel: [Float], nMels: Int, masks: AudioMasks) throws -> [Float] {
        let F = masks.bucketFrames, C = masks.chunks
        let cs = AudioMasks.chunkSize, T = C * AudioMasks.tpc
        guard Self.frameBuckets.contains(F), nMels == 128,
              packedMel.count == nMels * F,
              masks.convMask.count == C * cs,
              masks.attnBias.count == T * T else {
            throw EncoderError.invalidInput("audio masks and mel values do not match the native bucket")
        }
        let pk = try MLMultiArray(shape: [NSNumber(value: nMels), NSNumber(value: F)], dataType: .float32)
        let cm = try MLMultiArray(shape: [NSNumber(value: C), 1, NSNumber(value: cs)], dataType: .float32)
        let ab = try MLMultiArray(shape: [1, 1, NSNumber(value: T), NSNumber(value: T)], dataType: .float32)
        try CoreMLArrayReader.fillFloat32(pk, with: packedMel, label: "packed mel")
        try CoreMLArrayReader.fillFloat32(cm, with: masks.convMask, label: "audio convolution mask")
        try CoreMLArrayReader.fillFloat32(ab, with: masks.attnBias, label: "audio attention mask")
        let out = try model(F).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "packed_mel": pk, "conv_mask": cm, "attn_bias": ab,
        ]))
        guard let f = out.featureValue(for: "audio_features")?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try CoreMLArrayReader.float32(
            f, shape: [C * 50, 1024], label: "audio encoder features")
    }
}
