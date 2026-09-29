import Foundation

/// Versioned Core ML artifact bundle consumed by the Swift inference API.
///
/// Conversion tools produce a directory with `manifest.json` at its root. Inference code reads that
/// manifest and resolves artifact paths from it, instead of depending on converter script layout.
///
/// **Manifest v2** is model-agnostic: it carries everything the runtime needs that differs between
/// models — embedding dimension, Matryoshka dims, prompts, resolved media token ids, pixel budgets,
/// bucket sets, and whether the text tower needs an attention mask. Tower sections (`image`,
/// `audio`, `video`) are optional so text-only bundles are valid; ``capabilities`` reports what a
/// bundle supports. V2 removes every default that encoded jina-v5-omni-small's identity.
public struct GlossModelBundle: Sendable {
    public static let defaultManifestFilename = "manifest.json"

    public let rootDirectory: URL
    public let manifest: Manifest

    public init(rootDirectory: URL, manifest: Manifest) {
        self.rootDirectory = rootDirectory
        self.manifest = manifest
    }

    /// Read and validate `manifest.json` from a converted model bundle directory.
    public init(url: URL, manifestFilename: String = Self.defaultManifestFilename) throws {
        let manifestURL = url.appendingPathComponent(manifestFilename)
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch {
            throw GlossModelBundleError.missingManifest(url: manifestURL)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.formatVersion == Manifest.currentFormatVersion else {
            throw GlossModelBundleError.unsupportedFormatVersion(
                expected: Manifest.currentFormatVersion,
                actual: manifest.formatVersion
            )
        }
        self.init(rootDirectory: url, manifest: manifest)
    }

    /// Resolve a manifest path. Relative paths are interpreted inside `rootDirectory`; absolute paths
    /// are preserved so advanced callers can keep artifacts outside one directory.
    public func resolve(_ path: String) -> URL {
        Self.resolve(rootDirectory: rootDirectory, path: path)
    }

    public static func resolve(rootDirectory: URL, path: String) -> URL {
        path.hasPrefix("/") ? URL(fileURLWithPath: path) : rootDirectory.appendingPathComponent(path)
    }

    /// What a bundle can do — derived from which optional tower sections the manifest carries.
    public struct Capabilities: Equatable, Sendable {
        public let embeddingDimension: Int
        public let matryoshkaDimensions: [Int]
        public let supportsText: Bool
        public let supportsColbert: Bool
        public let supportsImage: Bool
        public let supportsAudio: Bool
        public let supportsVideo: Bool
    }

    public var capabilities: Capabilities {
        Capabilities(
            embeddingDimension: manifest.embeddingDimension,
            matryoshkaDimensions: manifest.matryoshkaDimensions,
            supportsText: manifest.text != nil,
            supportsColbert: manifest.colbert != nil,
            supportsImage: manifest.image != nil,
            supportsAudio: manifest.audio != nil,
            supportsVideo: manifest.video != nil
        )
    }

    public struct Manifest: Codable, Equatable, Sendable {
        public static let currentFormatVersion = 2

        public var formatVersion: Int
        public var modelID: String
        public var embeddingDimension: Int
        public var matryoshkaDimensions: [Int]
        public var minimumDeployment: MinimumDeployment
        public var source: Source?
        public var converter: Converter?
        public var prompts: Prompts?
        public var tokens: TokenArtifacts
        public var text: TextArtifacts?
        public var colbert: ColbertArtifacts?
        public var image: ImageArtifacts?
        public var audio: AudioArtifacts?
        public var video: VideoArtifacts?
        public var decoder: DecoderArtifacts?
        /// Compile provenance when the bundle's Core ML artifacts are precompiled `.mlmodelc`
        /// (see `CompiledArtifacts`). Absent for source (`.mlpackage`) bundles.
        public var compiled: CompiledArtifacts?
        /// Declared embedding-space identity of this bundle's numeric recipe. Hosts that
        /// persist vectors key them by this value. When absent, hosts fall back to
        /// fingerprinting the decoded manifest — correct for the bundle as shipped, but not
        /// stable across artifact regeneration (same weights, new layout) and blind to weight
        /// quantization. Optional v2 field: readers that predate it must ignore it.
        public var spaceID: String?
        /// The numeric recipe the towers were converted with. Identity-relevant: two recipes
        /// of one model produce different embeddings and are different spaces (see `spaceID`).
        public var precision: Precision?
        /// Per-file checksums for production artifact integrity. Paths are bundle-relative and
        /// cover the actual model, tokenizer, and preprocessing resource files used at runtime.
        public var artifactChecksums: ArtifactChecksums?
        /// Task-specific instruction pairs (code models: nl2code/code2code/code2nl/code2completion/qa).
        /// The main `prompts` pair is the default task; `taskPrompts` lets hosts select any task
        /// per the official best practice ("always use appropriate task-specific instruction
        /// prefixes"). Optional: bundles without tasks ignore it.
        public var taskPrompts: [String: TaskPrompts]?

        public init(
            formatVersion: Int = currentFormatVersion,
            modelID: String,
            embeddingDimension: Int,
            matryoshkaDimensions: [Int],
            minimumDeployment: MinimumDeployment = .default,
            source: Source? = nil,
            converter: Converter? = nil,
            prompts: Prompts? = nil,
            tokens: TokenArtifacts,
            text: TextArtifacts? = nil,
            colbert: ColbertArtifacts? = nil,
            image: ImageArtifacts? = nil,
            audio: AudioArtifacts? = nil,
            video: VideoArtifacts? = nil,
            decoder: DecoderArtifacts? = nil,
            compiled: CompiledArtifacts? = nil,
            spaceID: String? = nil,
            precision: Precision? = nil,
            artifactChecksums: ArtifactChecksums? = nil,
            taskPrompts: [String: TaskPrompts]? = nil
        ) {
            self.formatVersion = formatVersion
            self.modelID = modelID
            self.embeddingDimension = embeddingDimension
            self.matryoshkaDimensions = matryoshkaDimensions
            self.minimumDeployment = minimumDeployment
            self.source = source
            self.converter = converter
            self.prompts = prompts
            self.tokens = tokens
            self.text = text
            self.colbert = colbert
            self.image = image
            self.audio = audio
            self.video = video
            self.decoder = decoder
            self.compiled = compiled
            self.spaceID = spaceID
            self.precision = precision
            self.artifactChecksums = artifactChecksums
            self.taskPrompts = taskPrompts
        }
    }

    public struct MinimumDeployment: Codable, Equatable, Sendable {
        public static let `default` = MinimumDeployment()

        public var macOS: String
        public var iOS: String

        public init(macOS: String = "15.0", iOS: String = "18.0") {
            self.macOS = macOS
            self.iOS = iOS
        }
    }

    /// Where the weights came from (HF source model + pinned revision, when known).
    public struct Source: Codable, Equatable, Sendable {
        public var repo: String
        public var revision: String?

        public init(repo: String, revision: String? = nil) {
            self.repo = repo
            self.revision = revision
        }
    }

    /// What produced the bundle (converter name/version/git commit, when known).
    public struct Converter: Codable, Equatable, Sendable {
        public var name: String
        public var version: String
        public var commit: String?

        public init(name: String, version: String, commit: String? = nil) {
            self.name = name
            self.version = version
            self.commit = commit
        }
    }

    /// The numeric recipe of a bundle's towers (weight and activation dtypes, e.g.
    /// `{"weights": "int8", "activations": "float16"}` for a W8A16 bundle). Recorded as
    /// strings rather than an enum so future recipes don't break old readers.
    public struct Precision: Codable, Equatable, Sendable {
        public var weights: String
        public var activations: String

        public init(weights: String, activations: String) {
            self.weights = weights
            self.activations = activations
        }
    }

    /// Compile provenance for a precompiled bundle: the Core ML artifacts are `.mlmodelc`
    /// directories produced once by the system compiler (`python/converter/compile_bundle.py`),
    /// so hosts load them directly with no on-device compile step. Absent for source bundles.
    /// Optional v2 field: readers that predate it must ignore it and can still load the bundle
    /// (the runtime infers compiled form from the artifact path extension).
    public struct CompiledArtifacts: Codable, Equatable, Sendable {
        /// Artifact format, `"mlmodelc"`.
        public var format: String
        /// Compiler used, `"coremlcompiler"`.
        public var tool: String
        /// Platform the compile targeted (compatibility-checked), e.g. `"macOS"`.
        public var platform: String
        /// Deployment target the compile was checked against, e.g. `"15.0"`.
        public var deploymentTarget: String?
        /// Toolchain identity at compile time (Xcode version + build), when known.
        public var compilerVersion: String?
        /// Converter repo commit at compile time, when known.
        public var commit: String?

        public init(format: String, tool: String, platform: String,
                    deploymentTarget: String? = nil, compilerVersion: String? = nil,
                    commit: String? = nil) {
            self.format = format
            self.tool = tool
            self.platform = platform
            self.deploymentTarget = deploymentTarget
            self.compilerVersion = compilerVersion
            self.commit = commit
        }
    }

    /// Checksums over the regular files inside the artifact paths used by production inference.
    /// `files` keys use bundle-relative POSIX paths and lowercase SHA-256 hex values.
    public struct ArtifactChecksums: Codable, Equatable, Sendable {
        public var algorithm: String
        public var files: [String: String]

        public init(algorithm: String = "sha256", files: [String: String]) {
            self.algorithm = algorithm
            self.files = files
        }
    }

    /// Task prompt prefixes for text embedding (e.g. "Query: " / "Document: ").
    public struct Prompts: Codable, Equatable, Sendable {
        public var query: String
        public var document: String

        public init(query: String, document: String) {
            self.query = query
            self.document = document
        }
    }

    /// A task's instruction pair (code models: nl2code/code2code/code2nl/code2completion/qa).
    public struct TaskPrompts: Codable, Equatable, Sendable {
        public var query: String
        public var document: String

        public init(query: String, document: String) {
            self.query = query
            self.document = document
        }
    }

    /// Resolved token ids the runtime needs: the pad id plus each media wrapper. The converter
    /// resolves these with the model's own tokenizer (see `assemble_bundle.py`); the raw strings
    /// ride along in the JSON for debugging but are not decoded here.
    public struct TokenArtifacts: Codable, Equatable, Sendable {
        public var padID: Int32
        public var image: MediaTokenIDs?
        public var video: MediaTokenIDs?
        public var audio: MediaTokenIDs?

        public init(padID: Int32, image: MediaTokenIDs? = nil, video: MediaTokenIDs? = nil, audio: MediaTokenIDs? = nil) {
            self.padID = padID
            self.image = image
            self.video = video
            self.audio = audio
        }
    }

    public struct MediaTokenIDs: Codable, Equatable, Sendable {
        public var prefixIDs: [Int32]
        public var suffixIDs: [Int32]
        public var placeholderID: Int32
        /// Retrieval-side conditioned prefixes ("Query: "/"Document: " inserted after the chat
        /// template's `user\n` segment, per the model card's `text="Query: <|vision_start|>..."`
        /// recipe). Optional: bundles without them embed media unconditioned (`.none` prompt).
        public var queryPrefixIDs: [Int32]?
        public var documentPrefixIDs: [Int32]?

        public init(prefixIDs: [Int32], suffixIDs: [Int32], placeholderID: Int32,
                    queryPrefixIDs: [Int32]? = nil, documentPrefixIDs: [Int32]? = nil) {
            self.prefixIDs = prefixIDs
            self.suffixIDs = suffixIDs
            self.placeholderID = placeholderID
            self.queryPrefixIDs = queryPrefixIDs
            self.documentPrefixIDs = documentPrefixIDs
        }

        internal var mediaTokens: MediaTokens {
            MediaTokens(prefix: prefixIDs, suffix: suffixIDs, placeholder: placeholderID,
                        queryPrefix: queryPrefixIDs, documentPrefix: documentPrefixIDs)
        }
    }

    /// Batch-N text functions present in the package (`bucket_<S>_b<N>`), for memory-bound
    /// throughput. Optional: bundles without it still answer `embed(texts:)` row-by-row.
    ///
    /// Two manifest shapes are accepted:
    ///  - legacy single size: `{"size": N, "buckets": [S...]}` — one N for every bucket;
    ///  - ladder: `{"sizes": [N...], "buckets": [S...]}` — sizes[i] pairs with buckets[i].
    /// ``pairs`` resolves either shape into concrete (batchSize, bucket) functions.
    public struct BatchArtifacts: Codable, Equatable, Sendable {
        /// Legacy single-size form (`{size, buckets}`).
        public var size: Int?
        /// Ladder form (`{sizes, buckets}`), parallel to `buckets`; preferred over `size`.
        public var sizes: [Int]?
        public var buckets: [Int]

        public init(size: Int? = nil, sizes: [Int]? = nil, buckets: [Int]) {
            self.size = size
            self.sizes = sizes
            self.buckets = buckets
        }

        /// The concrete (batchSize, bucket) functions, in profile order. Ladder when `sizes` is
        /// present; otherwise the legacy single-size expansion.
        public var pairs: [(size: Int, bucket: Int)] {
            if let sizes, sizes.count == buckets.count {
                return zip(sizes, buckets).map { (size: $0, bucket: $1) }
            }
            if let size {
                return buckets.map { (size: size, bucket: $0) }
            }
            return []
        }
    }

    public struct TextArtifacts: Codable, Equatable, Sendable {
        public var model: String
        public var tokenizer: String
        public var buckets: [Int]
        /// Maximum total token count, including retrieval conditioning. Optional for legacy
        /// generic bundles; the production Omni Small API requires the native 32768-token value.
        public var maxTokens: Int?
        /// Bidirectional towers (e.g. EuroBERT/Llama) take an explicit `attention_mask` input;
        /// causal towers (Qwen3) do not. The runtime also adapts from the model's declared inputs;
        /// this field is the contract-level statement used for validation.
        public var requiresAttentionMask: Bool
        public var batch: BatchArtifacts?

        public init(model: String, tokenizer: String, buckets: [Int], maxTokens: Int? = nil,
                    requiresAttentionMask: Bool,
                    batch: BatchArtifacts? = nil) {
            self.model = model
            self.tokenizer = tokenizer
            self.buckets = buckets
            self.maxTokens = maxTokens
            self.requiresAttentionMask = requiresAttentionMask
            self.batch = batch
        }
    }

    /// ColBERT token-level retrieval encoder (kind=colbert bundles). Not a dense text embedder:
    /// the encoder emits per-token L2-normalized embeddings for a query (S=32) or a document
    /// (S=512) and the host scores with MaxSim, query EOS-expansion, and a document punctuation
    /// skiplist. Function names follow the text-tower convention: `bucket_<S>` (batch 1) and
    /// `bucket_<S>_b<N>` (batch-N), per role.
    public struct ColbertArtifacts: Codable, Equatable, Sendable {
        /// Multi-function encoder package (`bucket_<S>` / `bucket_<S>_b<N>` functions).
        public var encoder: String
        /// Tokenizer folder (`tokenizer.json` + `tokenizer_config.json`).
        public var tokenizer: String
        /// Per-token embedding dim (ColBERT head output, e.g. 128).
        public var dim: Int
        /// Outputs are L2-normalized in-graph.
        public var l2Normalized: Bool
        /// "per_token" — token-level embeddings, scored by the host.
        public var pooling: String
        /// "maxsim" — the retrieval similarity.
        public var similarity: String
        public var query: ColbertRoleArtifacts
        public var document: ColbertRoleArtifacts

        public init(encoder: String, tokenizer: String, dim: Int, l2Normalized: Bool,
                    pooling: String, similarity: String,
                    query: ColbertRoleArtifacts, document: ColbertRoleArtifacts) {
            self.encoder = encoder
            self.tokenizer = tokenizer
            self.dim = dim
            self.l2Normalized = l2Normalized
            self.pooling = pooling
            self.similarity = similarity
            self.query = query
            self.document = document
        }
    }

    /// One ColBERT role (query or document): prefix, max length, bucket geometry, and the
    /// role-specific retrieval knobs (query EOS-expansion; document punctuation skiplist).
    public struct ColbertRoleArtifacts: Codable, Equatable, Sendable {
        /// Role prefix, e.g. `[Q] ` / `[D] ` (resolved to token ids by the runtime tokenizer).
        public var prefix: String
        /// Max sequence length (query 32, document 512).
        public var maxLength: Int
        /// Single-row buckets (batch-1 functions `bucket_<S>`).
        public var buckets: [Int]
        /// Batch-N variant: `size` rows per call, functions `bucket_<S>_b<N>`.
        public var batch: BatchArtifacts?
        /// Query only: the token the query is expanded with to fill `maxLength` ("eos").
        public var expansionToken: String?
        /// Query only: whether expansion positions attend (false = attention-masked, scored).
        public var attendToExpansion: Bool?
        /// Document only: token strings dropped from document embeddings before MaxSim.
        public var skiplistWords: [String]?

        public init(prefix: String, maxLength: Int, buckets: [Int],
                    batch: BatchArtifacts? = nil, expansionToken: String? = nil,
                    attendToExpansion: Bool? = nil, skiplistWords: [String]? = nil) {
            self.prefix = prefix
            self.maxLength = maxLength
            self.buckets = buckets
            self.batch = batch
            self.expansionToken = expansionToken
            self.attendToExpansion = attendToExpansion
            self.skiplistWords = skiplistWords
        }
    }

    /// Host-side image preprocessing limits (the model processor's smart-resize bounds).
    public struct Preprocess: Codable, Equatable, Sendable {
        public var patch: Int
        public var merge: Int
        public var minPixels: Int
        public var maxPixels: Int

        public init(patch: Int, merge: Int, minPixels: Int, maxPixels: Int) {
            self.patch = patch
            self.merge = merge
            self.minPixels = minPixels
            self.maxPixels = maxPixels
        }
    }

    public struct ImageArtifacts: Codable, Equatable, Sendable {
        public var encoder: String
        public var resources: String
        public var patchBuckets: [Int]
        public var preprocess: Preprocess

        public init(encoder: String, resources: String, patchBuckets: [Int], preprocess: Preprocess) {
            self.encoder = encoder
            self.resources = resources
            self.patchBuckets = patchBuckets
            self.preprocess = preprocess
        }
    }

    public struct AudioArtifacts: Codable, Equatable, Sendable {
        public var encoder: String
        public var frameBuckets: [Int]
        /// Native production input contract. Optional for legacy generic bundles.
        public var sampleRate: Int?
        public var maxSamples: Int?
        public var maxFrames: Int?

        public init(encoder: String, frameBuckets: [Int], sampleRate: Int? = nil,
                    maxSamples: Int? = nil, maxFrames: Int? = nil) {
            self.encoder = encoder
            self.frameBuckets = frameBuckets
            self.sampleRate = sampleRate
            self.maxSamples = maxSamples
            self.maxFrames = maxFrames
        }
    }

    public struct VideoArtifacts: Codable, Equatable, Sendable {
        public var encoder: String
        public var patchBuckets: [Int]

        public init(encoder: String, patchBuckets: [Int]) {
            self.encoder = encoder
            self.patchBuckets = patchBuckets
        }
    }

    public struct DecoderArtifacts: Codable, Equatable, Sendable {
        public var embed: String
        public var model: String
        public var sequenceBuckets: [Int]

        public init(embed: String, model: String, sequenceBuckets: [Int]) {
            self.embed = embed
            self.model = model
            self.sequenceBuckets = sequenceBuckets
        }
    }
}

public enum GlossModelBundleError: Error, CustomStringConvertible {
    case missingManifest(url: URL)
    case unsupportedFormatVersion(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .missingManifest(url):
            return "GlossModelBundle: no manifest.json at \(url.path). Point at a converted model bundle "
                + "(python/converter/assemble_bundle.py) or a hub-downloaded snapshot."
        case let .unsupportedFormatVersion(expected, actual):
            return "GlossModelBundle: unsupported manifest formatVersion \(actual). Expected \(expected)."
        }
    }
}
