//! Typed Qwen-Image-2.1 engine failures shared by the family's MLX components.
//!
//! The enum is available without the `direct-mlx` feature so the pure render contract and the
//! serving engine compile everywhere; only the MLX-cause variant requires the runtime crate,
//! mirroring `QwenImage21VaeError`'s structure.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QwenImage21EngineError {
    #[error("Qwen-Image-2.1 component tensor '{tensor_name}' is missing")]
    MissingWeight { tensor_name: String },
    #[error(
        "Qwen-Image-2.1 component tensor '{tensor_name}' has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    WeightShape {
        tensor_name: String,
        actual_shape: Vec<i32>,
        expected_shape: Vec<usize>,
    },
    #[error("invalid Qwen-Image-2.1 component input: {description}")]
    InvalidInput { description: String },
    #[error("Qwen-Image-2.1 component execution failed: {description}")]
    Execution { description: String },
    #[cfg(feature = "direct-mlx")]
    #[error(transparent)]
    Mlx(#[from] astronomical_runtime_integration::MlxRuntimeError),
}
