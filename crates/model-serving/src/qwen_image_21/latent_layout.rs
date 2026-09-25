//! Qwen-Image-2.1 latent & pipeline geometry (weights-free).
//!
//! This module owns the **pure index math** that decides how a request's pixel dimensions map
//! onto the 64-channel latent tensor the transformer consumes and the pixel tensor the VAE
//! decodes back to. It is the image-path counterpart to [`super::text_conditioning`]: both are
//! the deterministic, weights-free scaffolding that the numeric engine (VAE conv, DiT forward,
//! weights, sampler) builds on top of.
//!
//! Every formula here is a faithful port of
//! `QwenImage21Pipeline.calculate_dimensions`, the `vae_scale_factor * 2` floor in
//! `check_inputs`/`prepare_latents`, and the static `_pack_latents`/`_unpack_latents` (the
//! reference comment: *"2.1 consumes latents unpatched, so packing is a plain spatial
//! flatten"*). No tensor values flow through here, so the slice is CPU-only and verifiable
//! without torch, weights, or a GPU.
//!
//! # Conventions
//!
//! - `vae_scale_factor` is 16, so the spatial multiple is 32. The generation height/width must
//!   be a multiple of 32 (the VAE compresses/spreads by 16).
//! - `calculate_dimensions` rounds a derived width/height to the **nearest** multiple of 32
//!   (Python `round()` = round-half-to-even), while the main pipeline floor-rounds an explicit
//!   height/width to a multiple of 32 (floor). Both are replicated separately.
//! - Every function here returns `(height, width)` except `calculate_dimensions`, which keeps the
//!   reference's own `(width, height)` order so the port stays readable against the source. Read
//!   that one function's return type before destructuring it.

/// The width of the text embeddings the denoising transformer consumes.
///
/// This is the one number two components must agree on: the Qwen3-VL encoder emits it
/// (`hidden_size`), the transformer reads it (`context_in_dim`), and the pipeline slices the
/// encoder output to it. Declared once here so no side can restate it independently — the
/// transformer's `CONTEXT_INPUT_WIDTH`, the encoder's `HIDDEN_WIDTH`, and the pipeline's slice
/// all read this constant.
pub const QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH: usize = 4096;

/// The VAE spatial compression factor. The VAE downsamples latents by 16 and upsamples back by 16.
pub const QWEN_IMAGE_21_VAE_SCALE_FACTOR: usize = 16;

/// The spatial multiple every generation height/width must satisfy: `vae_scale_factor * 2`.
pub const VAE_SPATIAL_MULTIPLE: usize = QWEN_IMAGE_21_VAE_SCALE_FACTOR * 2;

/// Python's `round()` is round-half-to-even (banker's rounding), not round-half-away-from-zero.
///
/// `calculate_dimensions` relies on this exact mode; matching it keeps this port bit-faithful to
/// the reference rather than drifting by one multiple of 32 on a halfway input.
#[must_use]
pub fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else {
        // Exactly halfway: round to the even integer.
        if (floor as i64).rem_euclid(2) == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Rounds `value` to the nearest multiple of `multiple` (half rounds to even, matching Python).
#[must_use]
pub fn round_to_nearest_multiple(value: usize, multiple: usize) -> usize {
    let multiple = multiple as f64;
    (round_half_to_even(value as f64 / multiple) * multiple) as usize
}

/// Rounds `value` down to the nearest multiple of `multiple` (`value // multiple * multiple`).
///
/// The `check_inputs`/`prepare_latents` floor that forces the generation size onto a multiple of
/// the VAE spatial multiple. Kept private: it is only the implementation detail of
/// [`resolve_generation_dimensions`], so exposing it adds surface without a consumer.
#[must_use]
fn floor_to_multiple(value: usize, multiple: usize) -> usize {
    value / multiple * multiple
}

/// Derive a width/height pair from a target area and an aspect ratio, each rounded to the
/// nearest multiple of 32.
///
/// Port of `QwenImage21Pipeline.calculate_dimensions` (copied from the edit pipeline):
/// `width = sqrt(area * ratio)`, `height = width / ratio`, then each rounded to the nearest
/// multiple of 32.
///
/// # Returns
///
/// `(width, height)` — width first, matching the reference function's own order rather than this
/// family's usual `(height, width)`.
///
/// The reference derives the width from the area and the height from the width; the tuple keeps
/// that order so this reads line-for-line against `calculate_dimensions` upstream. Every other
/// function in this module returns `(height, width)`.
#[must_use]
pub fn calculate_dimensions(target_area: f64, aspect_ratio: f64) -> (usize, usize) {
    let width = (target_area * aspect_ratio).sqrt();
    let height = width / aspect_ratio;
    (
        round_to_nearest_multiple(width as usize, VAE_SPATIAL_MULTIPLE),
        round_to_nearest_multiple(height as usize, VAE_SPATIAL_MULTIPLE),
    )
}

/// Resolve the final generation height/width from optional caller overrides and the default
/// output resolution, each floored to a multiple of the VAE spatial multiple.
///
/// Port of the `check_inputs` block:
/// ```text
/// height = height or output_resolution
/// width = width or output_resolution
/// width = width // multiple_of * multiple_of      (multiple_of = vae_scale_factor * 2)
/// height = height // multiple_of * multiple_of
/// ```
///
/// # Returns
///
/// `(height, width)` after defaulting and flooring.
#[must_use]
pub fn resolve_generation_dimensions(
    height: Option<usize>,
    width: Option<usize>,
    output_resolution: usize,
) -> (usize, usize) {
    let height = height.unwrap_or(output_resolution);
    let width = width.unwrap_or(output_resolution);
    (
        floor_to_multiple(height, VAE_SPATIAL_MULTIPLE),
        floor_to_multiple(width, VAE_SPATIAL_MULTIPLE),
    )
}

/// The packing that turns a `(channels, H, W)` latent block into a `(H*W, channels)` sequence.
///
/// Port of the static `QwenImage21Pipeline._pack_latents`: the reference consumes latents
/// unpatched, so packing is a plain spatial flatten — `view(batch, channels, H*W).transpose(1, 2)`.
/// The channel count is preserved; only the two spatial axes collapse into the sequence axis.
///
/// # Returns
///
/// `seq_len = H * W`; the per-token channel count is unchanged.
#[must_use]
pub fn pack_latents_seq_len(height: usize, width: usize) -> usize {
    height * width
}

/// Reduce a pixel extent to its latent extent: `2 * (value // 32)`, the formula both
/// `prepare_latents` and `_unpack_latents` apply. For a multiple-of-32 input this equals
/// `value / vae_scale_factor`.
#[must_use]
fn reduce_to_latent_dim(value: usize) -> usize {
    2 * (value / VAE_SPATIAL_MULTIPLE)
}

/// The `(H', W')` a packed `H*W` sequence unpacks back to, before the VAE applies its last stride.
///
/// Port of the shape math in `QwenImage21Pipeline._unpack_latents`:
/// `H' = 2 * (H // (vae_scale_factor * 2))`, `W' = 2 * (W // (vae_scale_factor * 2))`.
/// For an input already reduced to a multiple of 32 this equals `H / vae_scale_factor`, so the
/// packed latent grid is preserved end to end.
#[must_use]
pub fn unpack_spatial_dims(target_height: usize, target_width: usize) -> (usize, usize) {
    (
        reduce_to_latent_dim(target_height),
        reduce_to_latent_dim(target_width),
    )
}

/// The latent spatial shape `(latent_h, latent_w)` for a given pixel `height`/`width`.
///
/// Port of the `img_shapes` construction: each spatial extent divides by `vae_scale_factor`
/// (16) to give the latent grid that the transformer's interleaved sequence overlays.
#[must_use]
pub fn latent_spatial_dimensions(pixel_height: usize, pixel_width: usize) -> (usize, usize) {
    (
        pixel_height / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
        pixel_width / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
    )
}
