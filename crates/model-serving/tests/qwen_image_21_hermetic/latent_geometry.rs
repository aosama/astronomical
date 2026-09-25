//! Hermetic tests for the Qwen-Image-2.1 VAE latent geometry: the channel counts and the
//! latent-grid-to-pixel-grid expansion the decoder contract rests on.
//!
//! These constants are the decode side of the family's geometry: the transformer emits the
//! latent channel count, the VAE reconstructs the output channel count, and every decoded
//! dimension is the latent dimension times the spatial compression ratio. The render pipeline
//! sizes its output from the same arithmetic, so the numbers here are what a caller can predict
//! before any weights load.

use astronomical_model_serving::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT,
    QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO, decoded_pixel_dimensions,
};

#[test]
fn should_expand_the_latent_grid_by_the_spatial_compression_ratio() {
    let (pixel_height, pixel_width) = decoded_pixel_dimensions(64, 64);
    assert_eq!((pixel_height, pixel_width), (1024, 1024));
    // Height and width expand independently, so a non-square latent grid keeps its axes.
    let (pixel_height, pixel_width) = decoded_pixel_dimensions(46, 92);
    assert_eq!((pixel_height, pixel_width), (736, 1472));
}

#[test]
fn should_expose_the_reviewed_artifact_channel_counts() {
    assert_eq!(QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, 64);
    assert_eq!(QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT, 4);
    assert_eq!(QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO, 16);
}
