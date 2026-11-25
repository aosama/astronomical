//! Flash-Next's RMS norm, applied over groups of the last axis.
//!
//! Its checkpoint stores each scale as a deviation from one; binding adds
//! the one in float32 (see the weight binding's storage conventions), so the
//! weight this norm receives is the true scale. The reference computes the
//! whole norm in float32 and casts back, and the residual lanes this norm
//! reads carry four hidden states side by side, so each lane is normalized
//! over its own `group_width` features, not over the concatenated width.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

/// `input` normalized over every `group_width` features of its last axis,
/// scaled by `scale` (already `1 + deviation`, float32), in `input`'s dtype.
pub fn grouped_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    scale: &MlxArray,
    group_width: i32,
    epsilon: f32,
) -> Result<MlxArray, TextExecutionError> {
    let input_shape = input.shape();
    let width = validated_group_width(&input_shape, group_width)?;
    let mut grouped_shape = input_shape[..input_shape.len() - 1].to_vec();
    grouped_shape.extend([width / group_width, group_width]);
    let widened = runtime.astype(input, MlxDtype::Float32)?;
    let grouped = runtime.reshape(&widened, &grouped_shape)?;
    let mean_square = runtime.mean_axis(&runtime.square(&grouped)?, -1, true)?;
    let epsilon = runtime.full(&[], epsilon, MlxDtype::Float32)?;
    let inverse_root = runtime.rsqrt(&runtime.add(&mean_square, &epsilon)?)?;
    let normalized = runtime.multiply(&grouped, &inverse_root)?;
    let grouped_scale = runtime.reshape(scale, &[width / group_width, group_width])?;
    let scaled = runtime.multiply(&normalized, &grouped_scale)?;
    let flat = runtime.reshape(&scaled, &input_shape)?;
    Ok(runtime.astype(&flat, input.dtype())?)
}

/// The input's last-axis width, refused unless it divides into groups.
fn validated_group_width(input_shape: &[i32], group_width: i32) -> Result<i32, TextExecutionError> {
    let width = *input_shape.last().ok_or(TextExecutionError::InvalidInput {
        description: "a grouped norm needs at least one axis",
    })?;
    if group_width <= 0 || width % group_width != 0 {
        return Err(TextExecutionError::InvalidInput {
            description: "a grouped norm's width must divide into whole groups",
        });
    }
    Ok(width)
}

/// The query or key norm of an attention layer: Qwen 3.5's fused MLX norm,
/// or for Flash-Next, whose zero-centred scale is bound as float32, the
/// reference's float32 norm cast back to the input.
pub fn attention_input_norm(
    runtime: &MlxRuntime,
    config: &crate::qwen3_5_core::configuration::Qwen3_5Config,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, TextExecutionError> {
    if config.qwen4_exp_architecture().is_none() {
        return Ok(runtime.rms_norm(input, weight, epsilon)?);
    }
    let head_dimension = input.shape().last().copied().unwrap_or(0);
    grouped_rms_norm(runtime, input, weight, head_dimension, epsilon)
}

#[cfg(test)]
#[path = "centered_norm_tests.rs"]
mod centered_norm_tests;
