import Foundation;

import MLX;

import RuntimeIntegration;

/// Persistent prompt-cache block extraction or restoration could not bridge
/// to the live in-memory request decoder state, port of the Rust
/// `PersistentPromptCacheStateBridgeError`.
public enum PersistentPromptCacheStateBridgeError: Error, Equatable {

    case invalidBlockRange(blockStartTokens: Int, blockEndTokens: Int);

    case missingLayer(layerIndex: Int);

    case missingLayerTensor(layerIndex: Int, tensorRole: String);

    case missingBlockTensor(layerIndex: Int, tensorName: String);

    case invalidLayerTensorShape(layerIndex: Int, tensorRole: String, actualShape: [Int]);

    case blockRangeExceedsLayerTensor(
        layerIndex: Int,
        tensorRole: String,
        requestedEndTokens: Int,
        availableTokens: Int);

    case invalidRestoredSequenceTokenCount(restoredTokenCount: Int);

    case concatenatedTokenCountMismatch(
        layerIndex: Int,
        concatenatedTokenCount: Int,
        restoredTokenCount: Int);

    public var errorDescription: String? {
        switch self {
        case let .invalidBlockRange(blockStartTokens, blockEndTokens):
            return "the persistent prompt-cache block range "
                + "[\(blockStartTokens), \(blockEndTokens)) is invalid";
        case let .missingLayer(layerIndex):
            return "the request decoder state is missing layer \(layerIndex)";
        case let .missingLayerTensor(layerIndex, tensorRole):
            return "request decoder layer \(layerIndex) is missing its \(tensorRole) tensor";
        case let .missingBlockTensor(layerIndex, tensorName):
            return "the persistent prompt-cache block tensor \(tensorName) for layer "
                + "\(layerIndex) is missing";
        case let .invalidLayerTensorShape(layerIndex, tensorRole, actualShape):
            return "request decoder layer \(layerIndex) \(tensorRole) tensor has invalid "
                + "shape \(actualShape)";
        case let .blockRangeExceedsLayerTensor(layerIndex, tensorRole, requestedEndTokens, availableTokens):
            return "request decoder layer \(layerIndex) \(tensorRole) tensor has only "
                + "\(availableTokens) tokens, cannot extract through \(requestedEndTokens)";
        case let .invalidRestoredSequenceTokenCount(restoredTokenCount):
            return "the persistent prompt-cache KV restore token count "
                + "\(restoredTokenCount) is invalid";
        case let .concatenatedTokenCountMismatch(layerIndex, concatenatedTokenCount, restoredTokenCount):
            return "the persistent prompt-cache blocks concatenated to "
                + "\(concatenatedTokenCount) tokens for layer \(layerIndex), expected the "
                + "restored \(restoredTokenCount)";
        }
    }
}
