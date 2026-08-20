//! Typed failures at the validated FLUX.2 Klein VAE execution boundary.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Flux2KleinVaeError {
    #[error("FLUX.2 Klein packed latent geometry is invalid: {reason}")]
    InvalidLatentGeometry { reason: String },
    #[error("FLUX.2 Klein VAE tiling geometry is invalid: {reason}")]
    InvalidTilingGeometry { reason: String },
    #[cfg(feature = "direct-mlx")]
    #[error("FLUX.2 Klein VAE MLX execution failed")]
    Mlx(#[from] astronomical_runtime_integration::MlxRuntimeError),
}

impl Flux2KleinVaeError {
    pub(super) fn latent_geometry(reason: impl Into<String>) -> Self {
        Self::InvalidLatentGeometry {
            reason: reason.into(),
        }
    }

    pub(super) fn tiling_geometry(reason: impl Into<String>) -> Self {
        Self::InvalidTilingGeometry {
            reason: reason.into(),
        }
    }
}
