//! Weight-free Qwen-Image-2.1 VAE geometry: latent channels, output channels, and how far the
//! decoder expands the spatial grid.
//!
//! The reviewed artifact's `vae/config.json` sets `z_dim = 64`, `out_channels = 4` (the pipeline
//! treats condition images as RGBA), and `scale_factor_spatial = 16`, so decoding a
//! `[B, H, W, 64]` latent grid yields `[B, 16 * H, 16 * W, 4]` pixels.

/// Latent channels entering the decoder (`z_dim`).
pub const QWEN_IMAGE_21_LATENT_CHANNEL_COUNT: usize = 64;
/// Reconstructed pixel channels — the pipeline converts condition images to RGBA.
pub const QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT: usize = 4;
/// Spatial expansion from the latent grid to the pixel grid (`scale_factor_spatial`).
pub const QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO: usize = 16;

/// The `(pixel_height, pixel_width)` a `[B, H, W, 64]` latent grid decodes to — the same order
/// `latent_spatial_dimensions` and the generation-dimension resolution use, one per latent axis.
#[must_use]
pub fn decoded_pixel_dimensions(latent_height: usize, latent_width: usize) -> (usize, usize) {
    (
        latent_height * QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO,
        latent_width * QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO,
    )
}
