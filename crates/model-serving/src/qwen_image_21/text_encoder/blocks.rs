//! One Qwen3-VL decoder layer: GQA self-attention with QK-norm and half-split rope, then a
//! SwiGLU feed-forward, each behind an RMSNorm with a residual — the reference
//! `Qwen3VLTextDecoderLayer` exactly.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::mlx_math::{apply_rope_half_split, fp32_rms_norm, masked_attention};

use super::weights::{
    HEAD_WIDTH, KEY_VALUE_HEAD_COUNT, QUERY_HEAD_COUNT, QwenImage21TextEncoderLayerWeights,
};

/// The Qwen RMSNorm epsilon (`rms_norm_eps`).
pub(super) const RMS_NORM_EPSILON: f32 = 0.000_001;

/// The reference rope theta (`rope_theta = 5_000_000`). The table builder reads it directly, so a
/// caller cannot hand the encoder tables built for a different base frequency.
const ROPE_THETA: f64 = 5_000_000.0;

/// Per-forward attention context: the rope tables and the shared causal mask.
pub(super) struct EncoderBlockContext<'a> {
    pub(super) rope_cosines: &'a MlxArray,
    pub(super) rope_sines: &'a MlxArray,
    pub(super) causal_mask: &'a MlxArray,
    pub(super) attention_scale: f32,
}

pub(super) fn forward_layer(
    runtime: &MlxRuntime,
    weights: &QwenImage21TextEncoderLayerWeights,
    hidden_states: &MlxArray,
    context: &EncoderBlockContext<'_>,
) -> Result<MlxArray, QwenImage21EngineError> {
    let residual = hidden_states;
    let normalized = fp32_rms_norm(
        runtime,
        hidden_states,
        &weights.input_layernorm,
        RMS_NORM_EPSILON,
    )?;
    let attention_output = forward_attention(runtime, weights, &normalized, context)?;
    let hidden_after_attention = runtime.add(residual, &attention_output)?;

    let residual = &hidden_after_attention;
    let normalized = fp32_rms_norm(
        runtime,
        &hidden_after_attention,
        &weights.post_attention_layernorm,
        RMS_NORM_EPSILON,
    )?;
    let feed_forward_output = forward_feed_forward(runtime, weights, &normalized)?;
    Ok(runtime.add(residual, &feed_forward_output)?)
}

fn forward_attention(
    runtime: &MlxRuntime,
    weights: &QwenImage21TextEncoderLayerWeights,
    normalized: &MlxArray,
    context: &EncoderBlockContext<'_>,
) -> Result<MlxArray, QwenImage21EngineError> {
    let shape = normalized.shape();
    let (batch, sequence_length) = (shape[0], shape[1]);

    let queries = weights.query.forward(runtime, normalized)?;
    let keys = weights.key.forward(runtime, normalized)?;
    let values = weights.value.forward(runtime, normalized)?;

    let query_heads = runtime.reshape(
        &queries,
        &[
            batch,
            sequence_length,
            QUERY_HEAD_COUNT as i32,
            HEAD_WIDTH as i32,
        ],
    )?;
    let key_heads = runtime.reshape(
        &keys,
        &[
            batch,
            sequence_length,
            KEY_VALUE_HEAD_COUNT as i32,
            HEAD_WIDTH as i32,
        ],
    )?;
    let value_heads = runtime.reshape(
        &values,
        &[
            batch,
            sequence_length,
            KEY_VALUE_HEAD_COUNT as i32,
            HEAD_WIDTH as i32,
        ],
    )?;

    // QK-norm over the head axis, then rope — both per token, so they run in the
    // `(batch, tokens, heads, width)` layout before the head-major transpose.
    let query_heads = fp32_rms_norm(runtime, &query_heads, &weights.query_norm, RMS_NORM_EPSILON)?;
    let key_heads = fp32_rms_norm(runtime, &key_heads, &weights.key_norm, RMS_NORM_EPSILON)?;
    let query_heads = apply_rope_half_split(
        runtime,
        &query_heads,
        context.rope_cosines,
        context.rope_sines,
    )?;
    let key_heads = apply_rope_half_split(
        runtime,
        &key_heads,
        context.rope_cosines,
        context.rope_sines,
    )?;

    let query_heads = runtime.transpose_axes(&query_heads, &[0, 2, 1, 3])?;
    let key_heads = runtime.transpose_axes(&key_heads, &[0, 2, 1, 3])?;
    let value_heads = runtime.transpose_axes(&value_heads, &[0, 2, 1, 3])?;

    // GQA: repeat the KV heads so every query head sees its group's keys and values.
    let group_factor = (QUERY_HEAD_COUNT / KEY_VALUE_HEAD_COUNT) as i32;
    let repeated_keys = runtime.repeat_axis(&key_heads, group_factor, 1)?;
    let repeated_values = runtime.repeat_axis(&value_heads, group_factor, 1)?;

    let attended = masked_attention(
        runtime,
        &query_heads,
        &repeated_keys,
        &repeated_values,
        context.causal_mask,
        context.attention_scale,
    )?;
    let attended_tokens = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
    let attended_flat = runtime.reshape(
        &attended_tokens,
        &[
            batch,
            sequence_length,
            QUERY_HEAD_COUNT as i32 * HEAD_WIDTH as i32,
        ],
    )?;
    weights.output.forward(runtime, &attended_flat)
}

fn forward_feed_forward(
    runtime: &MlxRuntime,
    weights: &QwenImage21TextEncoderLayerWeights,
    normalized: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let gated = weights.gate.forward(runtime, normalized)?;
    let activated = runtime.silu(&gated)?;
    let up_projected = weights.up.forward(runtime, normalized)?;
    let combined = runtime.multiply(&activated, &up_projected)?;
    weights.down.forward(runtime, &combined)
}

/// The lower-triangular causal mask as an additive `(1, 1, S, S)` `0 / -inf` table.
pub(super) fn build_causal_mask(
    runtime: &MlxRuntime,
    sequence_length: usize,
) -> Result<MlxArray, QwenImage21EngineError> {
    let mut mask = Vec::with_capacity(sequence_length * sequence_length);
    for query_position in 0..sequence_length {
        for key_position in 0..sequence_length {
            let allowed = key_position <= query_position;
            mask.push(if allowed { 0.0_f32 } else { f32::NEG_INFINITY });
        }
    }
    Ok(runtime.array_from_f32(
        &mask,
        &[1, 1, sequence_length as i32, sequence_length as i32],
    )?)
}

/// The text-only rope tables `(S, head_width / 2)` — one entry per half-split pair.
///
/// Text-only prompts give all three mrope sections the same sequential positions, which
/// collapses the multimodal rope to the standard rotation: `inv_freq[j] = theta^(-2j/head_dim)`.
pub(super) fn build_rope_tables(
    runtime: &MlxRuntime,
    sequence_length: usize,
) -> Result<(MlxArray, MlxArray), QwenImage21EngineError> {
    let pair_count = HEAD_WIDTH / 2;
    let inverse_frequencies: Vec<f32> = (0..pair_count)
        .map(|frequency_index| {
            ROPE_THETA.powf(-(2.0 * frequency_index as f64 / HEAD_WIDTH as f64)) as f32
        })
        .collect();
    let mut cosine_values = Vec::with_capacity(sequence_length * pair_count);
    let mut sine_values = Vec::with_capacity(sequence_length * pair_count);
    for position in 0..sequence_length {
        for &frequency in &inverse_frequencies {
            let angle = position as f32 * frequency;
            cosine_values.push(angle.cos());
            sine_values.push(angle.sin());
        }
    }
    let shape = [sequence_length as i32, pair_count as i32];
    Ok((
        runtime.array_from_f32(&cosine_values, &shape)?,
        runtime.array_from_f32(&sine_values, &shape)?,
    ))
}
