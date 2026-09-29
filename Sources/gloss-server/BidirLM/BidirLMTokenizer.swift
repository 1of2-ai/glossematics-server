import Foundation
import Tokenizers

/// Upstream text conditioning at the pinned revision: Sentence Transformers wraps plain text in
/// the chat template (`<|im_start|>user\n{text}<|im_end|>\n`) and mean-pools every token,
/// template included. BidirLM has no query/document prefixes, so queries and documents share
/// one embedding space and one encoding.
///
/// A small pool of tokenizer instances lets concurrent HTTP handlers tokenize in parallel.
final class BidirLMTokenizer: @unchecked Sendable {
    static let userPrefix = "<|im_start|>user\n"
    static let userSuffix = "<|im_end|>\n"

    private let tokenizers: [any Tokenizer]
    private let condition = NSCondition()
    private var available: [Int]
    let prefixIDs: [Int32]
    let suffixIDs: [Int32]

    enum Failure: Error, CustomStringConvertible {
        case load(String)
        case template([Int32], [Int32])

        var description: String {
            switch self {
            case let .load(reason): "failed to load the BidirLM tokenizer: \(reason)"
            case let .template(prefix, suffix):
                "tokenizer produced chat-template IDs \(prefix) / \(suffix), which do not match the bundle"
            }
        }
    }

    init(folder: URL, manifest: BidirLMManifest, instances: Int = 4) throws {
        final class Box: @unchecked Sendable { var result: Result<any Tokenizer, any Error>? }
        var loaded = [any Tokenizer]()
        for _ in 0..<max(1, instances) {
            let box = Box()
            let semaphore = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                do { box.result = .success(try await AutoTokenizer.from(modelFolder: folder)) }
                catch { box.result = .failure(error) }
                semaphore.signal()
            }
            semaphore.wait()
            switch box.result {
            case let .success(tokenizer)?: loaded.append(tokenizer)
            case let .failure(error)?: throw Failure.load(String(describing: error))
            case nil: throw Failure.load("loader completed without a result")
            }
        }
        tokenizers = loaded
        available = Array(loaded.indices)
        prefixIDs = manifest.text.userPrefixIDs
        suffixIDs = manifest.text.userSuffixIDs
        let probe = tokenizers[0]
        let prefix = probe.encode(text: Self.userPrefix, addSpecialTokens: false).map(Int32.init)
        let suffix = probe.encode(text: Self.userSuffix, addSpecialTokens: false).map(Int32.init)
        guard prefix == prefixIDs, suffix == suffixIDs else { throw Failure.template(prefix, suffix) }
    }

    /// Full templated token sequence for plain text, tokenized as one string exactly like the
    /// upstream processor (special-token strings inside the text are parsed as special tokens).
    func templated(_ text: String) -> [Int32] {
        withTokenizer { $0.encode(text: Self.userPrefix + text + Self.userSuffix, addSpecialTokens: false) }
            .map(Int32.init)
    }

    /// Raw content tokens (no template), for building interleaved multimodal sequences.
    func content(_ text: String) -> [Int32] {
        withTokenizer { $0.encode(text: text, addSpecialTokens: false) }.map(Int32.init)
    }

    /// OpenAI token-array inputs are the tokenized text; wrap them in the same template.
    func templated(tokens: [Int32]) -> [Int32] { prefixIDs + tokens + suffixIDs }

    private func withTokenizer<T>(_ body: (any Tokenizer) -> T) -> T {
        condition.lock()
        while available.isEmpty { condition.wait() }
        let index = available.removeLast()
        condition.unlock()
        defer {
            condition.lock(); available.append(index); condition.signal(); condition.unlock()
        }
        return body(tokenizers[index])
    }
}
