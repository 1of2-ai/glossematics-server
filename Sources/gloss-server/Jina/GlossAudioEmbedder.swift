import AVFoundation
import CoreML
import Foundation

/// Decode an audio file to a 16 kHz mono `[Float]` (the model's input rate) via AVFoundation.
/// EXACT for files already 16 kHz mono (passthrough); other sample rates are resampled by
/// AVAudioConverter, which is not bit-identical to the reference's librosa resampler — a small
/// resample-only caveat (analogous to the image CoreGraphics-vs-PIL note).
internal enum GlossAudioFile {
    public enum DecodeError: Error { case converter, noData }

    public static func decode16kMono(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat   // always float32, file's rate + channels
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: file.length,
            sampleRate: inFormat.sampleRate,
            channelCount: inFormat.channelCount)
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw DecodeError.converter
        }
        try file.read(into: inBuf)
        // Already 16 kHz mono -> read the PCM directly (no converter, hence no priming latency): exact.
        if inFormat.sampleRate == 16000, inFormat.channelCount == 1, let ch = inBuf.floatChannelData {
            return Array(UnsafeBufferPointer(start: ch[0], count: Int(inBuf.frameLength)))
        }
        // Otherwise resample/downmix via AVAudioConverter (small resample-only caveat vs librosa).
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw DecodeError.converter
        }
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: 8_192) else {
            throw DecodeError.converter
        }
        var fed = false
        var output = [Float]()
        output.reserveCapacity(Int((Double(file.length) * 16_000 / inFormat.sampleRate).rounded(.up)))
        var emptyRounds = 0
        while true {
            try Task.checkCancellation()
            outBuf.frameLength = 0
            var convErr: NSError?
            let status = converter.convert(to: outBuf, error: &convErr) { _, inStatus in
                if fed { inStatus.pointee = .endOfStream; return nil }
                fed = true; inStatus.pointee = .haveData; return inBuf
            }
            if let convErr { throw convErr }
            guard status != .error else { throw DecodeError.converter }

            let count = Int(outBuf.frameLength)
            if count > 0 {
                guard let channels = outBuf.floatChannelData,
                      output.count + count <= OmniSmallInputLimits.maximumAudioSamples else {
                    throw DecodeError.converter
                }
                output.append(contentsOf: UnsafeBufferPointer(start: channels[0], count: count))
                emptyRounds = 0
            } else {
                emptyRounds += 1
            }
            if status == .endOfStream {
                guard !output.isEmpty else { throw DecodeError.noData }
                return output
            }
            guard emptyRounds < 3 else { throw DecodeError.noData }
        }
    }
}

/// True arbitrary-length audio embedder: the runtime-MASKED encoder (host-built conv_mask +
/// attn_bias mask the partial boundary chunk) + the unified general decoder. This matches the
/// reference at ANY clip length up to ~30 s (the model's WhisperFeatureExtractor limit; buckets
/// 2/4/8/16/32 s). Encoder on GPU (fp32 matmul accumulation), general decoder adaptive.
internal final class GlossAudioEmbedderMasked {
    /// The audio prompt wrapper (`prefix + placeholder × L + suffix`).
    public let tokens: MediaTokens
    /// Media feature width == the model's embedding dimension.
    public let featureDim: Int

    public let mel: GlossMelFrontend
    public let encoder: AudioCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder

    /// `encoderUnits` selects the encoder's compute placement. `.cpuAndGPU` (default) is the
    /// recommended choice — it is BOTH more accurate (fp32 matmul accumulation) AND faster
    /// (measured 83ms vs 129ms end-to-end for an 8s clip; the large attn_bias + masked-SDPA don't
    /// map well to the ANE). `.cpuAndNeuralEngine` runs the encoder on the ANE too (full-ANE) but
    /// is slower and less accurate — provided only for GPU-contended cases.
    /// `decoderUnits`: `nil` (default) = adaptive (ANE for S≤256, GPU for S≥512). Pass
    /// `.cpuAndNeuralEngine` with `encoderUnits: .cpuAndNeuralEngine` for a true full-ANE
    /// deployment (lowest-power, GPU-free; slower).
    public init(audioModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                tokens: MediaTokens, featureDim: Int = 1024, padTokenID: Int32 = 0,
                encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil,
                sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        self.tokens = tokens
        self.featureDim = featureDim
        mel = try GlossMelFrontend()
        encoder = try AudioCoreMLEncoderMasked(modelURL: audioModelURL, computeUnits: encoderUnits)
        decoder = try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                                          computeUnits: decoderUnits, featDim: featureDim,
                                          padTokenID: padTokenID, sequenceBuckets: sequenceBuckets)
    }

    /// The model's audio limit: WhisperFeatureExtractor caps at 30 s = 3000 mel frames. Longer clips
    /// are truncated here to match (the reference can't see past 30 s either).
    public static let maxFrames = 3000

    /// Audio file -> embedding (AVFoundation decode to 16 kHz mono, then `embed(_:)`). Exact for
    /// already-16 kHz-mono files; other rates carry the documented resample caveat. Clips >30 s truncate.
    /// `prompt` selects the retrieval side (bundle's conditioned "Query: "/"Document: " wrapper).
    public func embed(audioURL: URL, dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        try embed(GlossAudioFile.decode16kMono(audioURL), dim: dim, prompt: prompt)
    }

    /// 16 kHz mono waveform -> embedding. Exact reference parity for any length up to ~30 s.
    public func embed(_ audio: [Float], dim: Int? = nil,
                      prompt: GlossTextEmbedder.Prompt = .none) throws -> [Float] {
        let exactFrames = min(audio.count / mel.hop, Self.maxFrames)
        let F = AudioMasks.bucket(forFrames: exactFrames)
        let masks = AudioMasks(exactFrames: exactFrames, bucketFrames: F)
        // Zero the packed mel beyond the real frames (mel-space zeros, NOT the log-mel floor): conv1
        // runs per-chunk before the mask, so its kernel reaches from the last real frame into the
        // partial chunk's padding — floor values there contaminate it. Reference pads with zeros.
        var packed = try mel.packedMel(audio, frames: F)
        if exactFrames < F {
            for m in 0..<mel.nMels { for t in exactFrames..<F { packed[m * F + t] = 0.0 } }
        }
        let full = try encoder.encode(packedMel: packed, nMels: mel.nMels, masks: masks)
        let L = masks.realTokens
        let used = Array(full[0 ..< (L * featureDim)])
        let ids = tokens.ids(prompt: prompt, count: L)
        let emb = try decoder.decode(tokenIds: ids, features: used,
                                     scatterOffset: tokens.resolvedPrefix(for: prompt).count)
        if let d = dim { return matryoshka(emb, dim: d) }
        return emb
    }
}
