//! Typed transformer failures safe to propagate through the image engine.

use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Flux2KleinTransformerGeometryError {
    #[error("FLUX.2 Klein transformer dimensions must be positive")]
    ZeroDimension,
    #[error("FLUX.2 Klein hidden width must equal attention head count times head width")]
    AttentionWidthMismatch,
    #[error("FLUX.2 Klein four-axis RoPE width {rope_width} must equal head width {head_width}")]
    RopeWidthMismatch {
        rope_width: usize,
        head_width: usize,
    },
    #[error("FLUX.2 Klein RoPE axes must be positive and even")]
    InvalidRopeAxis,
    #[error("FLUX.2 Klein floating-point constants must be finite and positive")]
    InvalidFloatingPointConstant,
    #[error("FLUX.2 Klein geometry arithmetic overflowed")]
    ArithmeticOverflow,
}

#[derive(Debug, Error)]
pub enum Flux2KleinTransformerError {
    #[error(transparent)]
    Geometry(#[from] Flux2KleinTransformerGeometryError),
    #[error("FLUX.2 Klein transformer tensor '{tensor_name}' is missing")]
    MissingWeight { tensor_name: String },
    #[error(
        "FLUX.2 Klein transformer tensor '{tensor_name}' has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    WeightShape {
        tensor_name: String,
        actual_shape: Vec<i32>,
        expected_shape: Vec<usize>,
    },
    #[error("FLUX.2 Klein transformer tensor '{tensor_name}' must retain BF16 storage")]
    WeightDtype { tensor_name: String },
    #[error("FLUX.2 Klein transformer contains unassigned tensor '{tensor_name}'")]
    UnassignedWeight { tensor_name: String },
    #[error("invalid FLUX.2 Klein transformer input: {description}")]
    InvalidInput { description: &'static str },
    #[error("FLUX.2 Klein transformer block-group size must be positive")]
    ZeroBlockGroupSize,
    #[error("FLUX.2 Klein transformer execution was cancelled at a block-group boundary")]
    Cancelled,
    #[error("failed to clone the FLUX.2 Klein transformer descriptor for bounded block loading")]
    DescriptorClone {
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Mlx(#[from] MlxRuntimeError),
}
