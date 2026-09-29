import Accelerate
import Foundation

/// Whisper-style log-mel front-end (feature_size=128, n_fft=400, hop=160), matching
/// transformers WhisperFeatureExtractor. n_fft=400 is not a power of two, so the DFT is done as
/// a matmul against precomputed cos/sin bases (Accelerate BLAS) rather than an FFT.
///
/// `packedMel(audio)` returns the packed mel `(nMels=128, nFrames)` row-major — the encoder input.
internal struct GlossMelFrontend {
    public let nFFT = 400, hop = 160, nMels = 128, nFreq = 201
    public let nFrames: Int
    let window: [Float]       // (400,)
    let melFilters: [Float]   // (201,128) row-major
    let cosMat: [Float]       // (400,201) row-major
    let sinMat: [Float]

    public enum MelError: Error { case missingBundledResource, badResource, audioTooShort(Int) }

    /// Load the mel filterbank + window bundled with the package (no external files needed).
    public init(nFrames: Int = 200) throws {
        guard let mf = Bundle.module.url(forResource: "mel_filters", withExtension: "f32"),
              let w = Bundle.module.url(forResource: "mel_window", withExtension: "f32") else {
            throw MelError.missingBundledResource
        }
        try self.init(melFiltersURL: mf, windowURL: w, nFrames: nFrames)
    }

    public init(melFiltersURL: URL, windowURL: URL, nFrames: Int = 200) throws {
        self.nFrames = nFrames
        melFilters = try Self.loadF32(melFiltersURL)
        window = try Self.loadF32(windowURL)
        guard melFilters.count == nFreq * nMels, window.count == nFFT else { throw MelError.badResource }
        var c = [Float](repeating: 0, count: nFFT * nFreq)
        var s = [Float](repeating: 0, count: nFFT * nFreq)
        for n in 0..<nFFT {
            for k in 0..<nFreq {
                let a = 2.0 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                c[n * nFreq + k] = cos(a); s[n * nFreq + k] = sin(a)
            }
        }
        cosMat = c; sinMat = s
    }

    static func loadF32(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Row-major C(M,N) = A(M,K) @ B(K,N).
    private func gemm(_ a: [Float], _ b: [Float], _ M: Int, _ K: Int, _ N: Int) -> [Float] {
        var c = [Float](repeating: 0, count: M * N)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(M), Int32(N), Int32(K), 1.0,
                            ap.baseAddress, Int32(K), bp.baseAddress, Int32(N),
                            0.0, &c, Int32(N))
            }
        }
        return c
    }

    /// Whole-clip features exactly as `WhisperFeatureExtractor` computes them for one unpadded,
    /// untruncated waveform (the processor passes `padding=True, truncation=False`): centered STFT
    /// with numpy/torch reflect padding at both ends, `samples / hop` frames (the final STFT frame
    /// is dropped), log10 power mel, dynamic range clipped to 8 below the clip maximum, then
    /// `(x + 4) / 4`. Returns (128 x frames row-major, frames). Throws `MelError.audioTooShort` unless
    /// `samples.count > nFFT / 2`: the reflect padding needs that many samples, and a violated
    /// precondition here would crash the whole daemon on a hostile or truncated upload.
    func wholeClipLogMel(_ audio: [Float]) throws -> ([Float], Int) {
        let p = nFFT / 2, n = audio.count
        guard n > p else { throw MelError.audioTooShort(n) }
        let frames = n / hop
        var padded = [Float](repeating: 0, count: n + 2 * p)
        for j in 0..<p { padded[j] = audio[p - j] }
        for i in 0..<n { padded[p + i] = audio[i] }
        for k in 0..<p { padded[p + n + k] = audio[n - 2 - k] }
        var logmel = [Float]()
        logmel.reserveCapacity(frames * nMels)
        // Bounded blocks keep the frame matrix small for long clips.
        let block = 4_096
        var start = 0
        while start < frames {
            let count = min(block, frames - start)
            var F = [Float](repeating: 0, count: count * nFFT)
            for t in 0..<count {
                let base = (start + t) * hop
                for s in 0..<nFFT { F[t * nFFT + s] = padded[base + s] * window[s] }
            }
            let real = gemm(F, cosMat, count, nFFT, nFreq)
            let imag = gemm(F, sinMat, count, nFFT, nFreq)
            var power = [Float](repeating: 0, count: count * nFreq)
            for i in 0..<power.count { power[i] = real[i] * real[i] + imag[i] * imag[i] }
            let mel = gemm(power, melFilters, count, nFreq, nMels)
            logmel += mel.map { Foundation.log10(Swift.max($0, 1e-10)) }
            start += count
        }
        let gmax = logmel.max() ?? 0
        var packed = [Float](repeating: 0, count: nMels * frames)
        for t in 0..<frames {
            for m in 0..<nMels {
                packed[m * frames + t] = (Swift.max(logmel[t * nMels + m], gmax - 8.0) + 4.0) / 4.0
            }
        }
        return (packed, frames)
    }

    /// 16 kHz mono audio -> packed mel (nMels, nFrames) row-major. Uses the configured `nFrames`.
    /// Throws `MelError.audioTooShort` for clips shorter than the FFT half-window (instead of crashing).
    public func packedMel(_ audio: [Float]) throws -> [Float] {
        try packedMel(audio, frames: nFrames)
    }

    /// As above but for an explicit frame count (duration bucket) — the cos/sin/mel matrices are
    /// frame-independent, so one frontend serves every bucket. Audio shorter than `frames*hop` is
    /// zero-padded at the end (silent trailing frames), matching WhisperFeatureExtractor max-length.
    public func packedMel(_ audio: [Float], frames: Int) throws -> [Float] {
        let p = nFFT / 2
        guard audio.count > p else { throw MelError.audioTooShort(audio.count) }
        let needed = p + frames * hop + nFFT   // last frame base = (frames-1)*hop, +nFFT samples
        // center reflect-pad (numpy 'reflect': padded[j]=audio[p-j]) + audio + trailing zeros
        var padded = [Float](repeating: 0, count: Swift.max(needed, p + audio.count + nFFT))
        for j in 0..<p { padded[j] = audio[Swift.min(p - j, audio.count - 1)] }
        for i in 0..<audio.count { padded[p + i] = audio[i] }

        // windowed frames F (frames, nFFT)
        var F = [Float](repeating: 0, count: frames * nFFT)
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { F[t * nFFT + n] = padded[base + n] * window[n] }
        }

        let real = gemm(F, cosMat, frames, nFFT, nFreq)   // (frames, nFreq)
        let imag = gemm(F, sinMat, frames, nFFT, nFreq)
        var power = [Float](repeating: 0, count: frames * nFreq)
        for i in 0..<power.count { power[i] = real[i] * real[i] + imag[i] * imag[i] }

        let mel = gemm(power, melFilters, frames, nFreq, nMels)  // (frames, nMels)
        var logmel = mel.map { Foundation.log10(Swift.max($0, 1e-10)) }
        let gmax = logmel.max() ?? 0
        for i in 0..<logmel.count { logmel[i] = (Swift.max(logmel[i], gmax - 8.0) + 4.0) / 4.0 }

        // transpose (frames, nMels) -> packed (nMels, frames)
        var packed = [Float](repeating: 0, count: nMels * frames)
        for t in 0..<frames {
            for m in 0..<nMels { packed[m * frames + t] = logmel[t * nMels + m] }
        }
        return packed
    }
}
