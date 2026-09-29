import AVFoundation
import CoreML
import Foundation

/// Decode an audio file to a 16 kHz mono `[Float]` (the model's input rate) via AVFoundation.
/// Multichannel audio is averaged to mono by the host (librosa `mono=True`), never by the
/// converter. EXACT for files already 16 kHz (any channel count: the mean is the only arithmetic);
/// other sample rates are resampled by AVAudioConverter, which is not bit-identical to the
/// reference's librosa resampler — a small resample-only caveat (analogous to the image
/// CoreGraphics-vs-PIL note).
internal enum GlossAudioFile {
    public enum DecodeError: Error { case converter, noData }

    public static func decode16kMono(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat   // always float32, file's rate + channels
        try OmniSmallInputLimits.validateEstimatedAudio(
            frameCount: file.length,
            sampleRate: inFormat.sampleRate,
            channelCount: inFormat.channelCount)
        // Read in bounded chunks until the declared length is reached. A single `read(into:)` of
        // the whole file is NOT reliable: for mono float32 WAV files it returns early (15,349 of
        // 16,000 frames), which silently cut the tail off the clip.
        let channelCount = Int(inFormat.channelCount)
        let chunkFrames: AVAudioFrameCount = 65_536
        guard channelCount > 0,
              let chunk = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: chunkFrames) else {
            throw DecodeError.converter
        }

        // Downmix to mono OURSELVES, by the mean of all channels, before any resampling. The
        // source loads audio with librosa `mono=True`, which averages; AVAudioConverter's default
        // N -> 1 conversion keeps only the LEFT channel, so a stereo clip with a different voice
        // per side embedded as one voice. A single channel is copied untouched (bit-exact).
        var mono = [Float]()
        mono.reserveCapacity(Int(file.length))
        let scale = 1 / Float(channelCount)
        while file.framePosition < file.length {
            try Task.checkCancellation()
            try file.read(into: chunk, frameCount: chunkFrames)
            let frames = Int(chunk.frameLength)
            if frames == 0 { break }
            guard let channels = chunk.floatChannelData else { throw DecodeError.converter }
            if channelCount == 1 {
                mono.append(contentsOf: UnsafeBufferPointer(start: channels[0], count: frames))
            } else {
                for index in 0..<frames {
                    var sum = channels[0][index]
                    for channel in 1..<channelCount { sum += channels[channel][index] }
                    mono.append(sum * scale)
                }
            }
        }
        let frames = mono.count
        guard frames > 0 else { throw DecodeError.noData }
        // Already 16 kHz -> no converter, hence no priming latency: exact.
        if inFormat.sampleRate == 16000 { return mono }

        // Otherwise resample MONO -> 16 kHz mono via AVAudioConverter (small resample-only caveat
        // vs librosa's soxr_hq).
        guard let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inFormat.sampleRate,
                                             channels: 1, interleaved: false),
              let monoBuf = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: AVAudioFrameCount(frames)),
              let monoData = monoBuf.floatChannelData,
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: monoFormat, to: outFormat) else {
            throw DecodeError.converter
        }
        // Measured against the FP32 reference (soxr_hq), the Mastering algorithm at maximum
        // quality is NOT better overall: 44.1 kHz mono 0.9747 vs 0.9731 (query), but stereo 0.9770
        // vs 0.9795. The default stays.
        mono.withUnsafeBufferPointer { source in
            if let base = source.baseAddress { monoData[0].update(from: base, count: frames) }
        }
        monoBuf.frameLength = AVAudioFrameCount(frames)
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
                fed = true; inStatus.pointee = .haveData; return monoBuf
            }
            if let convErr { throw convErr }
            guard status != .error else { throw DecodeError.converter }

            let count = Int(outBuf.frameLength)
            if count > 0 {
                guard let outChannels = outBuf.floatChannelData,
                      output.count + count <= OmniSmallInputLimits.maximumAudioSamples else {
                    throw DecodeError.converter
                }
                output.append(contentsOf: UnsafeBufferPointer(start: outChannels[0], count: count))
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

/// Encoder-ready tensors for one audio clip: the bucketed mel, the runtime masks, and the prompt
/// ids. Computed on the CPU with no Core ML involvement (see ``PreparedVisionInputs`` for why the
/// split exists).
internal struct PreparedAudioInputs: Sendable {
    /// (nMels × bucketFrames) packed mel, zeroed past the clip's real frames.
    let packedMel: [Float]
    /// Conv and attention masks for the real frame count within `masks.bucketFrames`.
    let masks: AudioMasks
    /// Real merged audio tokens the decoder receives (`masks.realTokens`, always ≥ 1 here).
    let tokenCount: Int
    /// `prefix + placeholder × tokenCount + suffix` for the requested retrieval role.
    let tokenIDs: [Int32]
    /// Where the audio features are scattered into `tokenIDs` (the role's prefix length).
    let scatterOffset: Int
}

/// Host-side audio preparation: framing, log-mel, bucket choice, masks, and prompt ids. A pure
/// `Sendable` value with no Core ML state, so it can run on any thread.
internal struct AudioInputPreparer: Sendable {
    /// The model's audio limit: WhisperFeatureExtractor caps at 30 s = 3000 mel frames. Longer clips
    /// are truncated here to match (the reference can't see past 30 s either).
    static let maxFrames = 3000

    let mel: GlossMelFrontend
    let tokens: MediaTokens

    /// 16 kHz mono waveform -> encoder inputs. Exact reference parity for any length up to ~30 s.
    /// A clip too short to yield a single pooled token throws `MelError.audioTooShort`.
    /// `bucketFrames` pins the frame bucket instead of choosing the smallest that fits; startup
    /// verification uses it to run one clip through every bucket.
    func prepare(_ audio: [Float], prompt: GlossTextEmbedder.Prompt,
                 bucketFrames: Int? = nil) throws -> PreparedAudioInputs {
        // The source model counts ceil(samples / hop) frames: its attention mask is sliced with
        // stride `hop`, so a partial last frame counts, and that frame's mel is computed over
        // zero padding (`packedMel` zero-pads past the clip). Floor division dropped it: a
        // 40037-sample clip is 251 frames / 63 tokens in the source but was 250 / 62 here.
        let exactFrames = min((audio.count + mel.hop - 1) / mel.hop, Self.maxFrames)
        let F = bucketFrames ?? AudioMasks.bucket(forFrames: exactFrames)
        guard F >= exactFrames, AudioCoreMLEncoderMasked.frameBuckets.contains(F) else {
            throw GlossMelFrontend.MelError.audioTooShort(audio.count)
        }
        let masks = AudioMasks(exactFrames: exactFrames, bucketFrames: F)
        // The reference keeps `realTokens` pooled tokens. Zero (one or two mel frames) means there
        // is nothing to embed; scattering an empty feature matrix would only fail later, deeper.
        guard masks.realTokens > 0 else { throw GlossMelFrontend.MelError.audioTooShort(audio.count) }
        // Zero the packed mel beyond the real frames (mel-space zeros, NOT the log-mel floor): conv1
        // runs per-chunk before the mask, so its kernel reaches from the last real frame into the
        // partial chunk's padding — floor values there contaminate it. Reference pads with zeros.
        var packed = try mel.packedMel(audio, frames: F)
        if exactFrames < F {
            for m in 0..<mel.nMels { for t in exactFrames..<F { packed[m * F + t] = 0.0 } }
        }
        return PreparedAudioInputs(
            packedMel: packed, masks: masks, tokenCount: masks.realTokens,
            tokenIDs: tokens.ids(prompt: prompt, count: masks.realTokens),
            scatterOffset: tokens.resolvedPrefix(for: prompt).count)
    }
}

/// True arbitrary-length audio embedder: the runtime-MASKED encoder (host-built conv_mask +
/// attn_bias mask the partial boundary chunk) + the unified general decoder. This matches the
/// reference at ANY clip length up to ~30 s (the model's WhisperFeatureExtractor limit; buckets
/// 2/4/8/16/32 s). Encoder on GPU (fp32 matmul accumulation), general decoder adaptive.
///
/// Every `embed(...)` is `infer(preparer.prepare(...))`: the CPU half lives in
/// ``AudioInputPreparer`` (usable off the accelerator) and this class owns only the Core ML half.
internal final class GlossAudioEmbedderMasked {
    /// The audio prompt wrapper (`prefix + placeholder × L + suffix`).
    public let tokens: MediaTokens
    /// Media feature width == the model's embedding dimension.
    public let featureDim: Int

    public let mel: GlossMelFrontend
    public let encoder: AudioCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder
    /// The CPU half of the pipeline, safe to use from any thread.
    public let preparer: AudioInputPreparer

    /// Assembles an embedder from already-built parts. The daemon uses this to share ONE
    /// `GeneralMediaDecoder` (and its loaded functions) across the image, audio, and video pipelines.
    public init(audioModelURL: URL, mel: GlossMelFrontend, decoder: GeneralMediaDecoder,
                tokens: MediaTokens, featureDim: Int = 1024,
                encoderUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.tokens = tokens
        self.featureDim = featureDim
        self.mel = mel
        self.decoder = decoder
        encoder = try AudioCoreMLEncoderMasked(modelURL: audioModelURL, computeUnits: encoderUnits)
        preparer = AudioInputPreparer(mel: mel, tokens: tokens)
    }

    /// `encoderUnits` selects the encoder's compute placement. `.cpuAndGPU` (default) is the
    /// recommended choice — it is BOTH more accurate (fp32 matmul accumulation) AND faster
    /// (measured 83ms vs 129ms end-to-end for an 8s clip; the large attn_bias + masked-SDPA don't
    /// map well to the ANE). `.cpuAndNeuralEngine` runs the encoder on the ANE too (full-ANE) but
    /// is slower and less accurate — provided only for GPU-contended cases.
    /// `decoderUnits`: `nil` (default) = adaptive (ANE for S≤256, GPU for S≥512). Pass
    /// `.cpuAndNeuralEngine` with `encoderUnits: .cpuAndNeuralEngine` for a true full-ANE
    /// deployment (lowest-power, GPU-free; slower).
    public convenience init(audioModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            tokens: MediaTokens, featureDim: Int = 1024, padTokenID: Int32 = 0,
                            encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil,
                            sequenceBuckets: [Int] = GeneralMediaDecoder.defaultSequenceBuckets) throws {
        try self.init(
            audioModelURL: audioModelURL,
            mel: try GlossMelFrontend(),
            decoder: try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                                             computeUnits: decoderUnits, featDim: featureDim,
                                             padTokenID: padTokenID, sequenceBuckets: sequenceBuckets),
            tokens: tokens, featureDim: featureDim, encoderUnits: encoderUnits)
    }

    /// The model's audio limit: WhisperFeatureExtractor caps at 30 s = 3000 mel frames. Longer clips
    /// are truncated here to match (the reference can't see past 30 s either).
    public static let maxFrames = AudioInputPreparer.maxFrames

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
        try infer(try preparer.prepare(audio, prompt: prompt), dim: dim)
    }

    /// The Core ML half: masked encoder prediction, real-token slice, scatter into the prompt,
    /// decoder prediction, optional Matryoshka truncation.
    public func infer(_ prepared: PreparedAudioInputs, dim: Int? = nil) throws -> [Float] {
        let full = try encoder.encode(packedMel: prepared.packedMel, nMels: mel.nMels, masks: prepared.masks)
        let used = Array(full[0 ..< (prepared.tokenCount * featureDim)])
        let emb = try decoder.decode(tokenIds: prepared.tokenIDs, features: used,
                                     scatterOffset: prepared.scatterOffset)
        if let d = dim { return matryoshka(emb, dim: d) }
        return emb
    }
}
