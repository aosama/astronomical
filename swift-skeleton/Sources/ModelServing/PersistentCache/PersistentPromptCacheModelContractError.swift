import Foundation;

import ModelServing;

/// A model and its live resource budgets could not form a safe
/// persistent-state contract, port of the Rust
/// `PersistentPromptCacheModelContractError`.
public enum PersistentPromptCacheModelContractError: Error, Equatable, Sendable {

    case emptyModelId;

    case emptyModelRevision;

    case zeroMaximumContextTokenCount;

    case zeroCommonPrefixCheckpointStrideBlocks;

    case invalidConfiguredBlockTokenCount(
        requiredAlignmentTokens: Int,
        maximumContextTokens: Int);

    case configuredBlockChainExceedsSsdQuota(
        configuredBlockTokens: Int,
        maximumChainBytes: UInt64,
        globalSsdQuotaBytes: UInt64);

    case noPersistentState;

    case zeroStorageGeometryDivisor;

    case blockTokenCountOverflow;

    case sequenceStateBlockPayloadByteCountOverflow;

    case capturePayloadByteCountOverflow;

    case boundarySnapshotExceedsSsdQuota(
        boundarySnapshotBytes: UInt64,
        globalSsdQuotaBytes: UInt64);

    case captureExceedsMlxMemoryCeiling(
        captureMemoryBytes: UInt64,
        effectiveMlxMemoryCeilingBytes: UInt64);

    case blockFilesExceedSsdQuota(
        blockFileBytes: UInt64,
        globalSsdQuotaBytes: UInt64);

    case serializeStorageGeometry(problem: String);

    case decoderCacheLayout(DecoderCacheLayoutError);

    public var errorDescription: String? {
        switch self {
        case .emptyModelId:
            return "persistent model-state storage requires a nonempty model ID";
        case .emptyModelRevision:
            return "persistent model-state storage requires a nonempty model revision";
        case .zeroMaximumContextTokenCount:
            return "persistent model-state storage requires a positive maximum context";
        case .zeroCommonPrefixCheckpointStrideBlocks:
            return "persistent model-state common-prefix checkpoint stride must be positive";
        case let .invalidConfiguredBlockTokenCount(requiredAlignmentTokens, maximumContextTokens):
            return "configured persistent model-state block tokens must be positive, aligned "
                + "to \(requiredAlignmentTokens) tokens, and no larger than the "
                + "\(maximumContextTokens)-token context";
        case let .configuredBlockChainExceedsSsdQuota(
            configuredBlockTokens, maximumChainBytes, globalSsdQuotaBytes):
            return "configured \(configuredBlockTokens)-token persistent model-state blocks "
                + "require a \(maximumChainBytes)-byte maximum chain, exceeding the "
                + "\(globalSsdQuotaBytes)-byte global quota";
        case .noPersistentState:
            return "decoder-cache layout declares neither sequence nor boundary state";
        case .zeroStorageGeometryDivisor:
            return "decoder-cache storage geometry used a zero divisor";
        case .blockTokenCountOverflow:
            return "persistent model-state block token count overflowed";
        case .sequenceStateBlockPayloadByteCountOverflow:
            return "persistent sequence-state block payload byte count overflowed";
        case .capturePayloadByteCountOverflow:
            return "persistent model-state capture payload byte count overflowed";
        case let .boundarySnapshotExceedsSsdQuota(boundarySnapshotBytes, globalSsdQuotaBytes):
            return "boundary snapshot requires \(boundarySnapshotBytes) bytes, above SSD quota "
                + "\(globalSsdQuotaBytes) bytes";
        case let .captureExceedsMlxMemoryCeiling(captureMemoryBytes, effectiveMlxMemoryCeilingBytes):
            return "persistent model-state capture requires \(captureMemoryBytes) memory bytes, "
                + "above MLX ceiling \(effectiveMlxMemoryCeilingBytes) bytes";
        case let .blockFilesExceedSsdQuota(blockFileBytes, globalSsdQuotaBytes):
            return "persistent model-state block files require \(blockFileBytes) SSD bytes, "
                + "above quota \(globalSsdQuotaBytes) bytes";
        case let .serializeStorageGeometry(problem):
            return "failed to derive persistent model-state storage geometry: \(problem)";
        case let .decoderCacheLayout(layoutError):
            return String(describing: layoutError);
        }
    }
}
