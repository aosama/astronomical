//! Native MLX decoder owners plus hermetic packed-latent and tile geometry contracts.

mod error;
mod latent_layout;
mod tiling;

#[cfg(feature = "direct-mlx")]
mod attention;
#[cfg(feature = "direct-mlx")]
mod convolution;
#[cfg(feature = "direct-mlx")]
mod decoder;
#[cfg(feature = "direct-mlx")]
mod execution;
#[cfg(feature = "direct-mlx")]
mod normalization;
#[cfg(feature = "direct-mlx")]
mod resnet;
#[cfg(feature = "direct-mlx")]
mod up_block;

pub use error::Flux2KleinVaeError;
pub use latent_layout::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinPackedLatentLayout, flux2_klein_inverse_batch_norm_reference,
};
pub use tiling::{Flux2KleinVaeTile, Flux2KleinVaeTilePlan, Flux2KleinVaeTilingConfig};

#[cfg(feature = "direct-mlx")]
use attention::Flux2KleinVaeMiddleAttention;
#[cfg(feature = "direct-mlx")]
use convolution::Flux2KleinChannelLastConv2d;
#[cfg(feature = "direct-mlx")]
pub use decoder::{Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder};
#[cfg(feature = "direct-mlx")]
pub(in crate::flux2_klein) use execution::{Flux2KleinVaeDecodeAdvance, Flux2KleinVaeDecodeState};
#[cfg(feature = "direct-mlx")]
use normalization::Flux2KleinGroupNorm;
#[cfg(feature = "direct-mlx")]
use resnet::Flux2KleinVaeResnetBlock;
#[cfg(feature = "direct-mlx")]
use up_block::Flux2KleinVaeUpBlock;
