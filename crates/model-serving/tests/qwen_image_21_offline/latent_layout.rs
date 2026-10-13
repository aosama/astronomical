//! Hermetic tests for the Qwen-Image-2.1 latent and pipeline geometry.
//!
//! These pin the weights-free index math of `qwen_image_21::latent_layout`: how a request's
//! pixel dimensions floor onto the 32-pixel multiple, how the reference's area/aspect-ratio
//! derivation rounds (Python's round-half-to-even), and how pixel extents reduce to the latent
//! grid. Every formula here decides the shape of the tensors the transformer and VAE exchange,
//! so a wrong number surfaces as a broken render rather than an error.

use astronomical_model_serving::{
    VAE_SPATIAL_MULTIPLE, calculate_dimensions, latent_spatial_dimensions, pack_latents_seq_len,
    resolve_generation_dimensions, round_half_to_even, round_to_nearest_multiple,
    unpack_spatial_dims,
};

#[test]
fn should_default_and_floor_the_generation_dimensions_to_a_multiple_of_32() {
    // No overrides -> the default output resolution, floored to a multiple of 32.
    assert_eq!(
        resolve_generation_dimensions(None, None, 1024),
        (1024, 1024)
    );
    // Overrides floored down to the nearest multiple of 32.
    assert_eq!(
        resolve_generation_dimensions(Some(1000), Some(500), 1024),
        (992, 480)
    );
}

#[test]
fn should_floor_each_side_independently() {
    assert_eq!(
        resolve_generation_dimensions(Some(1023), Some(1), 1024),
        (992, 0)
    );
    assert_eq!(
        resolve_generation_dimensions(Some(64), Some(64), 1024),
        (64, 64)
    );
    assert_eq!(
        resolve_generation_dimensions(Some(33), Some(32), 1024),
        (32, 32)
    );
}

#[test]
fn should_calculate_dimensions_from_area_and_aspect_ratio() {
    // Area 1024^2, aspect ratio 1 -> 1024x1024, already a multiple of 32.
    let (width, height) = calculate_dimensions(1024.0 * 1024.0, 1.0);
    assert_eq!(width, 1024);
    assert_eq!(height, 1024);

    // Wide aspect (w/h = 2): width = sqrt(area*2) ~ 1448 -> 1440, height ~ 724 -> 736.
    let (width, height) = calculate_dimensions(1024.0 * 1024.0, 2.0);
    assert_eq!(width, 1440);
    assert_eq!(height, 736);
    assert_eq!(width % VAE_SPATIAL_MULTIPLE, 0);
    assert_eq!(height % VAE_SPATIAL_MULTIPLE, 0);
}

#[test]
fn should_round_half_to_even_not_half_away_from_zero() {
    // 2.5 must round to 2 (even), not 3 — the behavior Python's round() guarantees.
    assert_eq!(round_half_to_even(2.5), 2.0);
    assert_eq!(round_half_to_even(3.5), 4.0);
    assert_eq!(round_half_to_even(0.5), 0.0);
    assert_eq!(round_half_to_even(1.5), 2.0);
    // The nearest-multiple port inherits that.
    assert_eq!(round_to_nearest_multiple(80, 32), 64);
    assert_eq!(round_to_nearest_multiple(96, 32), 96);
}

#[test]
fn should_preserve_the_channel_count_when_flattening_spatially() {
    // Packing collapses the two spatial axes only; the channel count is untouched.
    let seq_len = pack_latents_seq_len(64, 64);
    assert_eq!(seq_len, 4096);
    assert_eq!(seq_len, 64 * 64);
}

#[test]
fn should_round_trip_pack_then_unpack() {
    // Pixel (1024, 768) reduces to the latent grid (64, 48); packing yields the latent
    // sequence, and unpack reduces the pixel dimensions back to that same latent grid.
    let (height, width) = (1024, 768);
    let (latent_height, latent_width) = unpack_spatial_dims(height, width);
    assert_eq!((latent_height, latent_width), (64, 48));
    let seq_len = pack_latents_seq_len(latent_height, latent_width);
    assert_eq!(seq_len, 64 * 48);
}

#[test]
fn should_derive_the_latent_grid_by_dividing_the_scale_factor() {
    // Inputs are multiples of 32 (as the pipeline guarantees), so height // 16 is exact.
    let (latent_height, latent_width) = latent_spatial_dimensions(1024, 1024);
    assert_eq!((latent_height, latent_width), (64, 64));
    let (latent_height, latent_width) = latent_spatial_dimensions(640, 384);
    assert_eq!((latent_height, latent_width), (40, 24));
}
