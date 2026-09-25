//! Typed failures at the validated Qwen-Image-2.1 VAE execution boundary.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QwenImage21VaeError {
    /// A tensor, activation, or size this component was handed does not match the reviewed
    /// artifact's geometry: a weight shape, a channel count, a latent grid, or a dimension that
    /// exceeds the MLX integer range.
    ///
    /// The variant is deliberately broader than "latent": the same check runs over manifest
    /// weights at load time and over activations at decode time, and naming it after one of those
    /// two callers made every other caller's failure message misleading.
    #[error("Qwen-Image-2.1 VAE geometry is invalid: {description}")]
    InvalidGeometry { description: String },
    #[cfg(feature = "direct-mlx")]
    #[error("Qwen-Image-2.1 VAE MLX execution failed")]
    Mlx(#[from] astronomical_runtime_integration::MlxRuntimeError),
}

impl QwenImage21VaeError {
    // Only the feature-gated VAE stages produce geometry failures; the constructor stays pure
    // so the error enum compiles without the MLX runtime.
    #[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
    pub(super) fn invalid_geometry(description: impl Into<String>) -> Self {
        Self::InvalidGeometry {
            description: description.into(),
        }
    }
}
