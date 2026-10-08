import Foundation

/**
 * Invalid explicit Qwen3.5 prompt-processing chunk size, port of the
 * Rust `Qwen3_5PromptProcessingChunkSizerError` validation contract.
 * The Rust twin case `ExceedsPlatformRange` has no Swift counterpart:
 * `Int` is 64-bit on every supported platform, so the `UInt32` config
 * capacity always converts.
 */
public enum Qwen35PromptProcessingChunkSizerError: Error, Equatable, Sendable {

    /// A chunk size of zero would never advance prompt processing.
    case mustBePositive
}
