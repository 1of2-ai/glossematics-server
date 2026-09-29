import Foundation

/// A templated multimodal input whose placeholder spans still await tower features.
struct PendingMediaSequence: Sendable {
    struct Span: Sendable {
        let input: MediaInput
        let start: Int
        var count: Int { input.tokens }
    }

    let ids: [Int32]
    /// 3-D MRoPE positions; nil for sequential positions (no image).
    let positions: [(Int32, Int32, Int32)]?
    let spans: [Span]
    /// Metrics label: "image", "audio", or "message".
    let kind: String

    var tokenCount: Int { ids.count }

    /// The language-model input once every span's features are known (in span order).
    func sequence(features: [BidirLMMediaEncoder.Features], deepstackLayers: Int) throws -> LMSequence {
        guard features.count == spans.count else {
            throw MediaSequenceBuilder.Failure.invalid("expected \(spans.count) feature sets, got \(features.count)")
        }
        var sequence = LMSequence(ids: ids)
        sequence.positions = positions
        var deepstack = [[Float16]](repeating: [], count: deepstackLayers)
        var hasImage = false
        for (span, f) in zip(spans, features) {
            guard f.count == span.count else {
                throw MediaSequenceBuilder.Failure.invalid("media produced \(f.count) features for \(span.count) placeholders")
            }
            sequence.replacements.append(.init(start: span.start, rows: f.rows))
            if case .image = span.input {
                guard f.deepstack.count == deepstackLayers else {
                    throw MediaSequenceBuilder.Failure.invalid("image features lack DeepStack rows")
                }
                hasImage = true
                sequence.deepstackPositions += Array(span.start..<(span.start + span.count))
                for k in 0..<deepstackLayers { deepstack[k] += f.deepstack[k] }
            }
        }
        if hasImage { sequence.deepstack = deepstack }
        return sequence
    }
}

/// Builds the upstream chat-templated token sequence for interleaved text and media.
///
/// The processor renders one user turn (`<|im_start|>user\n` + parts + `<|im_end|>\n`), with
/// `<|vision_start|><|image_pad|><|vision_end|>` per image and
/// `<|audio_start|><|audio_pad|><|audio_end|>` per clip, tokenizes that string, and then repeats
/// each pad token once per feature row. Text parts are inserted verbatim, with no separators.
enum MediaSequenceBuilder {
    enum Part: Sendable {
        case text(String)
        case media(MediaInput)
    }

    enum Failure: Error, CustomStringConvertible {
        case invalid(String)

        var description: String {
            switch self {
            case let .invalid(reason): reason
            }
        }
    }

    static let imagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"
    static let audioPlaceholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    /// Media control tokens may not appear inside text parts: they would change the item count
    /// the placeholders are matched against.
    static let reservedStrings = ["<|image_pad|>", "<|video_pad|>", "<|audio_pad|>", "<|vision_start|>",
                                  "<|vision_end|>", "<|audio_start|>", "<|audio_end|>"]

    static func build(_ parts: [Part], tokenizer: BidirLMTokenizer,
                      tokens: BidirLMManifest.MediaTokens, kind: String) throws -> PendingMediaSequence {
        var body = ""
        var media = [MediaInput]()
        for part in parts {
            switch part {
            case let .text(text):
                if let bad = reservedStrings.first(where: { text.contains($0) }) {
                    throw Failure.invalid("text parts must not contain the media control token \(bad)")
                }
                body += text
            case let .media(input):
                switch input {
                case .image: body += imagePlaceholder
                case .audio: body += audioPlaceholder
                }
                media.append(input)
            }
        }
        let templated = tokenizer.templated(body)
        var ids = [Int32]()
        ids.reserveCapacity(templated.count + media.reduce(0) { $0 + $1.tokens })
        var spans = [PendingMediaSequence.Span]()
        var next = 0
        for (index, id) in templated.enumerated() {
            guard id == tokens.imagePad || id == tokens.audioPad else {
                ids.append(id)
                continue
            }
            guard next < media.count else { throw Failure.invalid("unexpected media placeholder in the template") }
            let input = media[next]
            let (open, close): (Int32, Int32)
            switch input {
            case .image: (open, close) = (tokens.visionStart, tokens.visionEnd)
            case .audio: (open, close) = (tokens.audioStart, tokens.audioEnd)
            }
            let expected: Int32 = { if case .image = input { return tokens.imagePad } else { return tokens.audioPad } }()
            guard id == expected, index > 0, templated[index - 1] == open,
                  index + 1 < templated.count, templated[index + 1] == close else {
                throw Failure.invalid("media placeholders were not tokenized as expected")
            }
            spans.append(.init(input: input, start: ids.count))
            ids += Array(repeating: id, count: input.tokens)
            next += 1
        }
        guard next == media.count else { throw Failure.invalid("media placeholders were not tokenized as expected") }
        let grids: [(start: Int, h: Int, w: Int)] = spans.compactMap {
            if case let .image(image) = $0.input { return ($0.start, image.gridH / 2, image.gridW / 2) }
            return nil
        }
        return PendingMediaSequence(ids: ids, positions: grids.isEmpty ? nil : mropePositions(count: ids.count, images: grids),
                                    spans: spans, kind: kind)
    }

    /// Port of `get_rope_index` for images: text (and audio) tokens advance all three axes
    /// together; an image of `h x w` merged tokens starting at position `p` gets
    /// `(p, p + row, p + column)`, and the next token continues at `p + max(h, w)`.
    static func mropePositions(count: Int, images: [(start: Int, h: Int, w: Int)]) -> [(Int32, Int32, Int32)] {
        var out = [(Int32, Int32, Int32)]()
        out.reserveCapacity(count)
        var next: Int32 = 0
        var i = 0
        var pending = images.sorted { $0.start < $1.start }[...]
        while i < count {
            if let image = pending.first, image.start == i {
                pending.removeFirst()
                for r in 0..<image.h {
                    for c in 0..<image.w { out.append((next, next + Int32(r), next + Int32(c))) }
                }
                next += Int32(max(image.h, image.w))
                i += image.h * image.w
            } else {
                out.append((next, next, next))
                next += 1
                i += 1
            }
        }
        return out
    }
}
