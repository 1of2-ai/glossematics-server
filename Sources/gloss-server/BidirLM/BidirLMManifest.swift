import CoreML
import Foundation

/// Operator-selected Core ML placement. The server never falls back between these: a bundle is
/// loaded for exactly one mode and, in `ane` mode, every function's compute plan is verified at
/// startup so an operating-system or compiler change cannot silently move work to the CPU.
enum ComputeMode: String, CaseIterable, Sendable {
    case ane
    case gpu
    case cpu

    var units: MLComputeUnits {
        switch self {
        case .ane: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        case .cpu: .cpuOnly
        }
    }

    var coreMLName: String {
        switch self {
        case .ane: "cpuAndNeuralEngine"
        case .gpu: "cpuAndGPU"
        case .cpu: "cpuOnly"
        }
    }
}

/// `manifest.json` of a `bidirlm-omni-ane-v2` bundle (see GlossematicsCoreML/bidirlm).
struct BidirLMManifest: Decodable, Sendable {
    struct TokenEmbeddings: Decodable, Sendable {
        let file: String
        let shape: [Int]
        let dtype: String
    }

    struct Text: Decodable, Sendable {
        let maxTokens: Int
        let chunks: [Int]
        /// Layers per packed stack program (`stack_c64` / `stack_c512`).
        let stackLayers: Int
        let keyBlock: Int
        let longKeys: [Int]
        let poolWidth: Int
        let layers: Int
        let hidden: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let ropeTheta: Double
        let mropeSection: [Int]
        let maskNegative: Double
        let deepstackLayers: Int
        let userPrefixIDs: [Int32]
        let userSuffixIDs: [Int32]
        /// One package per stack group: `stack_c64`, `stack_c512`, and that group's long-input
        /// functions (`head_c512`, `mid_c512_l{i}`, `tail_c512`).
        let groupModels: [String]
        /// Long-input attention, one compiled package per key bucket (keyed by function name).
        let attentionModels: [String: String]
    }

    /// Chunked Qwen3-VL-style vision tower (see `ane_media.py`).
    struct Vision: Decodable, Sendable {
        let chunk: Int
        let hidden: Int
        let heads: Int
        let headDim: Int
        let layers: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let mergeSize: Int
        let patchFeatures: Int
        let minPixels: Int
        let maxPixels: Int
        let imageMean: [Double]
        let imageStd: [Double]
        let ropeTheta: Double
        let keyBlock: Int
        let singleBlockMaxKeys: Int
        let positionTable: String
        let positionTableShape: [Int]
        let keys: [Int]
        /// Blocks after which a DeepStack merger output feeds text layers 0, 1, ...
        let deepstackBlocks: [Int]
        /// `head`, `mid_l{i}` (with a `deepstack` output after DeepStack blocks), `tail`.
        let towerModel: String
        let attentionModels: [String: String]
    }

    /// Chunked audio encoder: convolutional front end over 200-frame mel chunks, then full
    /// attention across every audio token of the clip.
    struct Audio: Decodable, Sendable {
        let chunk: Int
        let hidden: Int
        let heads: Int
        let headDim: Int
        let layers: Int
        let sampleRate: Int
        let melBins: Int
        let nFFT: Int
        let hopLength: Int
        let chunkFrames: Int
        let tokensPerChunk: Int
        let frontBatch: Int
        let keys: [Int]
        let keyBlock: Int
        let singleBlockMaxKeys: Int
        let frontModel: String
        /// `head`, `mid_l{i}`, `tail`.
        let towerModel: String
        let attentionModels: [String: String]
    }

    struct MediaTokens: Decodable, Sendable, Equatable {
        let imagePad: Int32
        let visionStart: Int32
        let visionEnd: Int32
        let audioPad: Int32
        let audioStart: Int32
        let audioEnd: Int32
    }

    /// Numeric recipe. The only shipped recipe is W8A16: int8 per-channel weights, float16
    /// activations.
    struct Precision: Decodable, Sendable, Equatable {
        let weights: String
        let activations: String
    }

    struct Parity: Decodable, Sendable {
        let pass: Bool
    }

    struct Qualification: Decodable, Sendable {
        let aneStrictFunctions: Int
        let aneStrictAll: Bool
        let sealed: Bool
        let parity: [String: Parity]
    }

    struct Checksums: Decodable, Sendable {
        let algorithm: String
        let files: [String: String]
    }

    let format: String
    let modelID: String
    let revision: String
    let embeddingDimension: Int
    let pooling: String
    let padTokenID: Int32
    let tokenizer: String
    let tokenEmbeddings: TokenEmbeddings
    let text: Text
    let streamed: StreamedTextManifest?
    /// Staged W8A16 vision/audio towers of a streamed release (`streamed_media.json`).
    let streamedMedia: StreamedMediaManifest?
    let vision: Vision?
    let audio: Audio?
    let mediaTokens: MediaTokens?
    let precision: Precision
    let neuralCompute: String
    let spaceID: String
    let qualification: Qualification
    let checksums: Checksums?
    /// Present only on test fixtures (`"dummy-noop"`); the daemon requires `--allow-dummy`.
    let fixture: String?

    var isFixture: Bool { fixture == "dummy-noop" }

    /// DeepStack feature sets each image contributes: one per DeepStack tower block. The complete
    /// language functions consume exactly that many (`text.deepstackLayers`, validated equal). The
    /// staged language encoder has inputs for the config's three DeepStack indexes and zero-fills
    /// the third: the 24-block tower never reaches index 24, so it produces sets after blocks 8
    /// and 16 only, exactly as the upstream model does.
    var imageDeepstackSets: Int { vision?.deepstackBlocks.count ?? text.deepstackLayers }
}

enum BidirLMContract {
    static let format = "bidirlm-omni-ane-v2"
    static let modelID = "BidirLM/BidirLM-Omni-2.5B-Embedding"
    static let revision = "447a6e31be61b84443144afda21374339ce408e6"
    static let dimension = 2048
    static let vocabulary = 151_936
    static let spaceVersion = "ane-chunked-mean-v1"
    /// The only supported numeric recipe (int8 weights, float16 activations).
    static let recipe = "w8a16"
    static let precision = BidirLMManifest.Precision(weights: "int8", activations: "float16")
    static let maxTokens = 32_768
    static let chunks = [64, 512]
    static let fusedChunk = 64
    static let stackLayers = 7
    static var stacks: Int { layers / stackLayers }
    static let keyBlock = 2048
    static let longKeys = [1024] + (1...16).map { $0 * 2048 }
    static let poolWidth = 64
    static let layers = 28
    static let hidden = 2048
    static let heads = 16
    static let kvHeads = 8
    static let headDim = 128
    static let userPrefixIDs: [Int32] = [151_644, 872, 198]
    static let userSuffixIDs: [Int32] = [151_645, 198]
    static let padTokenID: Int32 = 151_643
    /// Additive attention mask for padding keys (finite in FP16).
    static let maskNegative = -30_000.0

    static var spaceID: String { "\(modelID):\(revision):\(dimension):\(recipe):\(spaceVersion)" }

    static func attentionName(chunk: Int, keys: Int, packed: Bool) -> String {
        "attn_c\(chunk)_s\(keys)_\(packed ? "packed" : "keys")"
    }

    /// Long-input attention buckets (packed chunks attend inside the stacks).
    static var attentionFunctions: [(chunk: Int, keys: Int, packed: Bool)] {
        longKeys.map { (512, $0, false) }
    }

    // Media towers (pinned source geometry).
    static let mediaTokens = BidirLMManifest.MediaTokens(
        imagePad: 151_655, visionStart: 151_652, visionEnd: 151_653,
        audioPad: 151_676, audioStart: 151_669, audioEnd: 151_670)
    static let encoderChunk = 512
    static let encoderHidden = 1024
    static let encoderHeads = 16
    static let encoderHeadDim = 64
    static let visionLayers = 24
    static let visionKeys = (1...8).map { $0 * 512 }
    static let visionMaxPatches = 4096
    static let patchFeatures = 1536
    static let visionDeepstackBlocks = [8, 16]
    static let positionTableShape = [2304, 1024]
    static let audioLayers = 24
    static let audioKeys = [512, 1024] + (1...16).map { $0 * 2048 }
    static let audioChunkFrames = 200
    static let audioTokensPerChunk = 25
    static let audioFrontBatch = 20
    static let melBins = 128

    /// Functions shipped in text stack group `group`.
    static func groupFunctions(_ group: Int) -> [String] {
        let first = group * stackLayers, last = (group + 1) * stackLayers
        var names = chunks.map { "stack_c\($0)" }
        if group == 0 { names.append("head_c512") }
        names += (first..<min(last, layers - 1)).map { "mid_c512_l\($0)" }
        if last >= layers { names.append("tail_c512") }
        return names
    }

    /// Functions shipped in a media tower package.
    static func towerFunctions(layers: Int) -> [String] {
        ["head"] + (0..<(layers - 1)).map { "mid_l\($0)" } + ["tail"]
    }
}

extension BidirLMManifest.Vision {
    /// Pinned preprocessing geometry (for host-side tests that run without a bundle).
    static let reference = BidirLMManifest.Vision(
        chunk: 512, hidden: 1024, heads: 16, headDim: 64, layers: 24, patchSize: 16, temporalPatchSize: 2,
        mergeSize: 2, patchFeatures: 1536, minPixels: 65_536, maxPixels: 1_048_576, imageMean: [0.5, 0.5, 0.5],
        imageStd: [0.5, 0.5, 0.5], ropeTheta: 10_000, keyBlock: 2048, singleBlockMaxKeys: 4096,
        positionTable: "vision/pos_embed_table.f32", positionTableShape: [2304, 1024], keys: BidirLMContract.visionKeys,
        deepstackBlocks: [8, 16], towerModel: "vision/tower.mlmodelc", attentionModels: [:])
}
