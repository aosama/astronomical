//! Native MLX decoder for the Qwen-Image-2.1 VAE plus its weight-free latent geometry contracts.
//!
//! The decoder is the final stage of image generation: it turns the denoised 64-channel latents
//! into RGBA pixels. Every structural fact it encodes (channel progression, the residual up
//! blocks with pixel-shuffle shortcuts, the skipped temporal path) was read from the reviewed
//! artifact's weight manifest and its diffusers reference implementation.
//!
//! The files split by owner: `decoder` assembles and drives the decode, `convolution`, `rms_norm`,
//! `resnet`, `attention`, `mid_block`, and `up_block` each own one reference component,
//! `decode_stages` owns the staged walk, `tensor_shape` owns the shape and range checks every
//! component runs, `latent_geometry` owns the numeric facts that need no weights, and `error`
//! owns the typed failure.

mod error;
mod latent_geometry;

#[cfg(feature = "direct-mlx")]
mod attention;
#[cfg(feature = "direct-mlx")]
mod convolution;
#[cfg(feature = "direct-mlx")]
mod decode_stages;
#[cfg(feature = "direct-mlx")]
mod decoder;
#[cfg(feature = "direct-mlx")]
mod mid_block;
#[cfg(feature = "direct-mlx")]
mod resnet;
#[cfg(feature = "direct-mlx")]
mod rms_norm;
#[cfg(feature = "direct-mlx")]
mod tensor_shape;
#[cfg(feature = "direct-mlx")]
mod up_block;

pub use error::QwenImage21VaeError;
pub use latent_geometry::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT,
    QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO, decoded_pixel_dimensions,
};

#[cfg(feature = "direct-mlx")]
pub use decoder::QwenImage21VaeDecoder;
