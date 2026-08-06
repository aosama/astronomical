use thiserror::Error;

/// One invalid model-owned decoder-cache layout.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecoderCacheLayoutError {
    #[error("model configuration {dimension_name} dimension does not fit usize")]
    ModelConfigurationDimensionOutsideUsizeRange { dimension_name: &'static str },
    #[error("decoder-cache layer {layer_index} has no composite components")]
    EmptyComposite { layer_index: usize },
    #[error("decoder-cache layer {layer_index} append-only attention has zero capacity growth")]
    ZeroCapacityGrowthTokens { layer_index: usize },
    #[error(
        "decoder-cache layer {layer_index} sequence tensor {qualified_role_name} has no sequence axis"
    )]
    SequenceTensorMissingAxis {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} tensor {qualified_role_name} has sequence axis {sequence_axis} outside rank {tensor_rank}"
    )]
    SequenceAxisOutsideTensorRank {
        layer_index: usize,
        qualified_role_name: String,
        sequence_axis: usize,
        tensor_rank: usize,
    },
    #[error(
        "decoder-cache layer {layer_index} sequence tensor {qualified_role_name} must use zero for its dynamic sequence dimension"
    )]
    SequenceAxisMustUseDynamicDimension {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} boundary tensor {qualified_role_name} must not have a sequence axis"
    )]
    BoundaryTensorHasSequenceAxis {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache layer {layer_index} boundary tensor {qualified_role_name} must not have a dynamic dimension"
    )]
    BoundaryTensorHasDynamicDimension {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error("decoder-cache layer {layer_index} has an empty tensor role")]
    EmptyTensorRole { layer_index: usize },
    #[error("decoder-cache layer {layer_index} tensor {qualified_role_name} has zero rank")]
    ZeroTensorRank {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error("decoder-cache layer {layer_index} repeats tensor role {qualified_role_name}")]
    DuplicateTensorRole {
        layer_index: usize,
        qualified_role_name: String,
    },
    #[error(
        "decoder-cache tensor {qualified_role_name} has invalid payload geometry: {description}"
    )]
    InvalidTensorPayloadGeometry {
        qualified_role_name: String,
        description: &'static str,
    },
    #[error("decoder-cache tensor {qualified_role_name} payload byte count overflowed")]
    TensorPayloadByteCountOverflow { qualified_role_name: String },
    #[error(
        "decoder-cache layer {layer_index} append-only attention keys and values have different contracts"
    )]
    AttentionTensorContractMismatch { layer_index: usize },
}
