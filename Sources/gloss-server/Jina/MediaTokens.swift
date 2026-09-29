import Foundation

/// Resolved token ids for one modality's prompt wrapper: `prefix + placeholder × L + suffix`.
///
/// Converted bundles carry these in `manifest.json` — the converter resolves them with the
/// model's OWN tokenizer, so a model whose chat-template specials are plain BPE text (e.g.
/// jina-v5-omni-nano) gets multi-token prefix/suffix lists, while a model with real special
/// tokens gets single ids. The runtime never hardcodes a model's vocabulary.
///
/// The static presets below exist ONLY for direct construction in tests and dev tools
/// (which run against small's artifacts by path). Anything reading a bundle should use the
/// manifest's values via ``GlossModelBundle/Manifest``.
internal struct MediaTokens: Sendable, Equatable {
    /// Ids BEFORE the media slots (the scatter offset is `prefix.count`).
    public var prefix: [Int32]
    /// Ids AFTER the media slots.
    public var suffix: [Int32]
    /// The single id repeated L times to hold the L media features.
    public var placeholder: Int32
    /// Retrieval-side conditioned prefixes ("Query: "/"Document: " after `user\n`) resolved by the
    /// converter with the model's own tokenizer. `nil` for bundles that embed media unconditioned.
    public var queryPrefix: [Int32]?
    public var documentPrefix: [Int32]?

    public init(prefix: [Int32], suffix: [Int32], placeholder: Int32,
                queryPrefix: [Int32]? = nil, documentPrefix: [Int32]? = nil) {
        self.prefix = prefix
        self.suffix = suffix
        self.placeholder = placeholder
        self.queryPrefix = queryPrefix
        self.documentPrefix = documentPrefix
    }

    /// The prefix ids for a retrieval side: the conditioned prefix when present for that prompt,
    /// else the plain wrapper (`.none` / bundles without conditioned prefixes).
    public func resolvedPrefix(for prompt: GlossTextEmbedder.Prompt) -> [Int32] {
        switch prompt {
        case .query: return queryPrefix ?? prefix
        case .document: return documentPrefix ?? prefix
        case .none: return prefix
        }
    }

    /// The full media prompt sequence for a retrieval side: prefix + placeholder × L + suffix.
    public func ids(prompt: GlossTextEmbedder.Prompt, count L: Int) -> [Int32] {
        resolvedPrefix(for: prompt) + Array(repeating: placeholder, count: L) + suffix
    }

    /// jina-v5-omni-small (Qwen-omni): `<|im_start|>user\n<|vision_start|>` … `<|image_pad|>`.
    /// Includes the retrieval-side conditioned prefixes ("Query: "/"Document: " after `user\n`)
    /// resolved by the converter with small's tokenizer — the model card's recipe for every modality.
    public static let jinaV5OmniSmallImage = MediaTokens(
        prefix: [151644, 872, 198, 151652], suffix: [151653, 151645, 198], placeholder: 151655,
        queryPrefix: [151644, 872, 198, 2859, 25, 220, 151652],
        documentPrefix: [151644, 872, 198, 7524, 25, 220, 151652])
    /// jina-v5-omni-small: same vision wrapper, `<|video_pad|>` placeholder.
    public static let jinaV5OmniSmallVideo = MediaTokens(
        prefix: [151644, 872, 198, 151652], suffix: [151653, 151645, 198], placeholder: 151656,
        queryPrefix: [151644, 872, 198, 2859, 25, 220, 151652],
        documentPrefix: [151644, 872, 198, 7524, 25, 220, 151652])
    /// jina-v5-omni-small: `<|im_start|>user\n<|audio_start|>` … `<|audio_pad|>`.
    public static let jinaV5OmniSmallAudio = MediaTokens(
        prefix: [151644, 872, 198, 151670], suffix: [151671, 151645, 198], placeholder: 151669,
        queryPrefix: [151644, 872, 198, 2859, 25, 220, 151670],
        documentPrefix: [151644, 872, 198, 7524, 25, 220, 151670])
}
