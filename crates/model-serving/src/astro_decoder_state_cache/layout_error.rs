use thiserror::Error;

/// One invalid model-owned astro-decoder-state-cache layout.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum AstroDecoderStateCacheLayoutError {
    #[error("model configuration {dimension_name} dimension does not fit usize")]
    ModelConfigurationDimensionOutsideUsizeRange { dimension_name: &'static str },
    #[error(
        "astro-decoder-state-cache execution dtype count {actual_layer_count} differs from model layer count {expected_layer_count}"
    )]
    ExecutionDtypeLayerCountMismatch {
        expected_layer_count: usize,
        actual_layer_count: usize,
    },
    #[error(
        "astro-decoder-state-cache layer {layer_index} execution dtype family differs from model attention family"
    )]
    ExecutionDtypeLayerFamilyMismatch { layer_index: usize },
    #[error("astro-decoder-state-cache layer {layer_index} has no composite components")]
    EmptyComposite { layer_index: usize },
    #[error(
        "astro-decoder-state-cache layer {layer_index} append-only attention has zero capacity growth"
    )]
    ZeroCapacityGrowthTokens { layer_index: usize },
    #[error(
        "astro-decoder-state-cache layer {layer_index} rotating attention has zero window size"
    )]
    ZeroRotatingWindowSize { layer_index: usize },
    #[error(
        "astro-decoder-state-cache layer {layer_index} sequence tensor {tensor_role_name} has no sequence axis"
    )]
    SequenceTensorMissingAxis {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error(
        "astro-decoder-state-cache layer {layer_index} tensor {tensor_role_name} has sequence axis {sequence_axis} outside rank {tensor_rank}"
    )]
    SequenceAxisOutsideTensorRank {
        layer_index: usize,
        tensor_role_name: String,
        sequence_axis: usize,
        tensor_rank: usize,
    },
    #[error(
        "astro-decoder-state-cache layer {layer_index} sequence tensor {tensor_role_name} must use zero for its dynamic sequence dimension"
    )]
    SequenceAxisMustUseDynamicDimension {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error(
        "astro-decoder-state-cache layer {layer_index} boundary tensor {tensor_role_name} must not have a sequence axis"
    )]
    BoundaryTensorHasSequenceAxis {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error(
        "astro-decoder-state-cache layer {layer_index} boundary tensor {tensor_role_name} must not have a dynamic dimension"
    )]
    BoundaryTensorHasDynamicDimension {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error("astro-decoder-state-cache layer {layer_index} has an empty tensor role")]
    EmptyTensorRole { layer_index: usize },
    #[error(
        "astro-decoder-state-cache layer {layer_index} tensor {tensor_role_name} has zero rank"
    )]
    ZeroTensorRank {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error("astro-decoder-state-cache layer {layer_index} repeats tensor role {tensor_role_name}")]
    DuplicateTensorRole {
        layer_index: usize,
        tensor_role_name: String,
    },
    #[error(
        "astro-decoder-state-cache tensor {tensor_role_name} has invalid payload geometry: {description}"
    )]
    InvalidTensorPayloadGeometry {
        tensor_role_name: String,
        description: &'static str,
    },
    #[error("astro-decoder-state-cache tensor {tensor_role_name} payload byte count overflowed")]
    TensorPayloadByteCountOverflow { tensor_role_name: String },
    #[error("astro-decoder-state-cache boundary snapshot payload byte count overflowed")]
    BoundarySnapshotPayloadByteCountOverflow,
    #[error("astro-decoder-state-cache sequence-state payload byte count per token overflowed")]
    SequenceStatePayloadByteCountPerTokenOverflow,
    #[error("astro-decoder-state-cache sequence tensor payload byte count overflowed")]
    SequenceTensorPayloadByteCountOverflow,
    #[error("astro-decoder-state-cache persistence alignment token count overflowed")]
    PersistenceAlignmentTokenCountOverflow,
    #[error(
        "astro-decoder-state-cache layer {layer_index} append-only attention keys and values have different geometry"
    )]
    AttentionTensorContractMismatch { layer_index: usize },
}
