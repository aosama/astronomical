//! Layers that more than one model family computes the same way, all in FP32
//! with the result cast back to the value's own dtype, the precision boundary
//! the diffusers reference implementations use: the normalizations and the
//! adjacent-pair rotary rotation of the FLUX.2 and Qwen-Image transformers.
//! One copy here means a numerical fix reaches every family that uses it.
//!
//! They return the runtime's own `MlxCError`, which every family error
//! already converts from, so a caller's `?` works unchanged. Each family keeps
//! its own input checks, with its own error messages, before calling in.

use mlx_bridge::{MlxArray, MlxCError, MlxDtype};
use runtime_integration::MlxRuntime;

/// `nn.LayerNorm` without affine parameters: normalized in FP32, then cast
/// back to the input dtype.
pub fn fp32_layer_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, MlxCError> {
    let normalized_f32 = runtime
        .layer_norm_without_weight_and_bias(&runtime.astype(input, MlxDtype::Float32)?, epsilon)?;
    runtime.astype(&normalized_f32, input.dtype())
}

/// `RMSNorm` over the last axis: FP32 normalization, cast back, then the
/// BF16 learned scale. The diffusers order multiplies by the weight only
/// after the cast, so the scale is applied at the value dtype.
pub fn fp32_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, MlxCError> {
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let normalized_f32 = runtime.rms_norm_without_weight(&input_f32, epsilon)?;
    let normalized = runtime.astype(&normalized_f32, input.dtype())?;
    runtime.multiply(&normalized, weight)
}

/// Rotary position rotation for heads that store each complex component as
/// adjacent real and imaginary values; rotating feature halves instead would
/// couple unrelated frequencies. Rotates `(B, S, H, D)` heads by per-token `(S, D)` cosine and sine tables,
/// in FP32, and casts the result back to the input dtype.
///
/// The caller validates the shapes, with its own family's error: a 4D input,
/// tables of shape `[S, D]`, and an even `D`. MLX reports any shape it still
/// cannot broadcast as an `MlxCError`.
pub fn rotate_adjacent_pairs(
    runtime: &MlxRuntime,
    input: &MlxArray,
    cosines: &MlxArray,
    sines: &MlxArray,
) -> Result<MlxArray, MlxCError> {
    let shape = input.shape();
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let real_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    let imaginary_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 1],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    let negative_imaginary_values = runtime.negative(&imaginary_values)?;
    let rotated_pairs = runtime.concatenate_axis(
        &[
            &runtime.expand_dims(&negative_imaginary_values, 4)?,
            &runtime.expand_dims(&real_values, 4)?,
        ],
        4,
    )?;
    let rotated_pairs = runtime.reshape(&rotated_pairs, &shape)?;
    let broadcast_cosines = runtime.reshape(cosines, &[1, shape[1], 1, shape[3]])?;
    let broadcast_sines = runtime.reshape(sines, &[1, shape[1], 1, shape[3]])?;
    let cosine_component = runtime.multiply(&input_f32, &broadcast_cosines)?;
    let sine_component = runtime.multiply(&rotated_pairs, &broadcast_sines)?;
    let rotated = runtime.add(&cosine_component, &sine_component)?;
    runtime.astype(&rotated, input.dtype())
}

#[cfg(test)]
#[path = "fp32_layers_tests.rs"]
mod fp32_layers_tests;
