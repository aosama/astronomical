import Foundation

/**
 * One invalid model-owned decoder-cache layout, port of the Rust
 * `DecoderCacheLayoutError`. The Rust twin case
 * `ModelConfigurationDimensionOutsideUsizeRange` has no Swift counterpart:
 * `Int` is 64-bit on every supported platform, so model configuration
 * dimensions always convert.
 */
public enum DecoderCacheLayoutError: Error, Equatable, Sendable {

    /// Decoder-cache execution dtype count differs from the model layer count.
    case executionDtypeLayerCountMismatch(expectedLayerCount: Int, actualLayerCount: Int)

    /// Decoder-cache execution dtype family differs from the model attention family.
    case executionDtypeLayerFamilyMismatch(layerIndex: Int)

    /// A composite decoder-cache layer declares no components.
    case emptyComposite(layerIndex: Int)

    /// Append-only attention declares zero capacity growth.
    case zeroCapacityGrowthTokens(layerIndex: Int)

    /// Rotating attention declares a zero window size.
    case zeroRotatingWindowSize(layerIndex: Int)

    /// A sequence tensor declares no sequence axis.
    case sequenceTensorMissingAxis(layerIndex: Int, tensorRoleName: String)

    /// A tensor sequence axis sits outside the declared rank.
    case sequenceAxisOutsideTensorRank(
        layerIndex: Int, tensorRoleName: String, sequenceAxis: Int, tensorRank: Int)

    /// A sequence tensor must use zero for its dynamic sequence dimension.
    case sequenceAxisMustUseDynamicDimension(layerIndex: Int, tensorRoleName: String)

    /// A boundary tensor must not carry a sequence axis.
    case boundaryTensorHasSequenceAxis(layerIndex: Int, tensorRoleName: String)

    /// A boundary tensor must not carry a dynamic dimension.
    case boundaryTensorHasDynamicDimension(layerIndex: Int, tensorRoleName: String)

    /// A decoder-cache layer declares a tensor with an empty role name.
    case emptyTensorRole(layerIndex: Int)

    /// A decoder-cache tensor declares zero rank.
    case zeroTensorRank(layerIndex: Int, tensorRoleName: String)

    /// One layer repeats a tensor role name.
    case duplicateTensorRole(layerIndex: Int, tensorRoleName: String)

    /// A tensor declares payload geometry that cannot be restored.
    case invalidTensorPayloadGeometry(tensorRoleName: String, description: String)

    /// A tensor payload byte count overflowed.
    case tensorPayloadByteCountOverflow(tensorRoleName: String)

    /// The boundary snapshot payload byte count overflowed.
    case boundarySnapshotPayloadByteCountOverflow

    /// The sequence-state payload byte count per token overflowed.
    case sequenceStatePayloadByteCountPerTokenOverflow

    /// A sequence tensor payload byte count overflowed.
    case sequenceTensorPayloadByteCountOverflow

    /// The persistence alignment token count overflowed.
    case persistenceAlignmentTokenCountOverflow

    /// The persistent prompt-cache block payload byte count overflowed.
    case persistentPromptCacheBlockPayloadByteCountOverflow

    /// Append-only or rotating attention keys and values have different geometry.
    case attentionTensorContractMismatch(layerIndex: Int)
}
