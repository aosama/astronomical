import Foundation;

/// A persistent model-state block identity could not be produced, port of
/// the Rust `PersistentPromptCacheBlockKeyError`.
public enum PersistentPromptCacheBlockKeyError: Error, Equatable, Sendable {

    case emptyBlockTokens;

    case blockTokenCountExceedsBlock(
        actualTokenCount: Int,
        maximumTokenCount: Int);

    case blockTokenCountOverflow;

    case blockIndexOverflow;

    public var errorDescription: String? {
        switch self {
        case .emptyBlockTokens:
            return "persistent model-state block tokens must not be empty";
        case let .blockTokenCountExceedsBlock(actualTokenCount, maximumTokenCount):
            return "persistent model-state block has \(actualTokenCount) tokens, "
                + "maximum \(maximumTokenCount)";
        case .blockTokenCountOverflow:
            return "persistent model-state block token count exceeds the u32 range";
        case .blockIndexOverflow:
            return "persistent model-state block index exceeds the u32 range";
        }
    }
}
