//! The Qwen-Image-2.1 single-stream transformer block.
//!
//! Port of `QwenImage21TransformerBlock`. Modulation is not learned per block: the parent model
//! computes one shared modulation tensor and hands every block the same four selected
//! `(scale, gate)` pairs, so the gates are `tanh`-activated once in preparation and reused by
//! all 32 blocks — mathematically identical to per-block `tanh` on the same values.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::mlx_math::{
    QuantizedLinear, apply_rope, fp32_layer_norm, fp32_rms_norm, fused_attention, masked_attention,
};

pub(super) const LAYER_NORM_EPSILON: f32 = 0.000_001;

#[derive(Debug)]
pub(super) struct QwenImage21AttentionWeights {
    pub(super) to_query: QuantizedLinear,
    pub(super) to_key: QuantizedLinear,
    pub(super) to_value: QuantizedLinear,
    pub(super) to_output: QuantizedLinear,
    pub(super) norm_query: MlxArray,
    pub(super) norm_key: MlxArray,
}

#[derive(Debug)]
pub(super) struct QwenImage21FeedForwardWeights {
    pub(super) gate_layer: QuantizedLinear,
    pub(super) projection: QuantizedLinear,
    pub(super) output: QuantizedLinear,
}

#[derive(Debug)]
pub(super) struct QwenImage21BlockWeights {
    pub(super) attention: QwenImage21AttentionWeights,
    pub(super) feed_forward: QwenImage21FeedForwardWeights,
}

/// One attention call's boundaries: query rows `[start, end)` attending to keys `[0, end)`.
#[derive(Clone, Debug)]
pub(super) struct AttentionSegment {
    pub(super) query_start: usize,
    pub(super) query_end: usize,
    pub(super) key_end: usize,
    pub(super) is_text: bool,
}

/// Everything the 32 blocks share in one forward: the modulation the parent model selected, the
/// attention structure, and the rope tables. Every field is already per token, so blocks read
/// them rather than recomputing anything.
pub(super) struct BlockContext<'a> {
    pub(super) attention_scale: f32,
    pub(super) rope_cosines: &'a MlxArray,
    pub(super) rope_sines: &'a MlxArray,
    pub(super) segments: &'a [AttentionSegment],
    /// Additive `0 / -inf` masks for the text segments, aligned with `segments`.
    pub(super) segment_masks: &'a [MlxArray],
    /// `(1 + scale)` for the attention norm, per token.
    pub(super) attention_one_plus_scale: &'a MlxArray,
    /// `tanh(gate)` for the attention branch, per token.
    pub(super) attention_gate: &'a MlxArray,
    /// `(1 + scale)` for the feed-forward norm, per token.
    pub(super) feed_forward_one_plus_scale: &'a MlxArray,
    /// `tanh(gate)` for the feed-forward branch, per token.
    pub(super) feed_forward_gate: &'a MlxArray,
    pub(super) head_count: usize,
    pub(super) head_width: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_block(
    runtime: &MlxRuntime,
    weights: &QwenImage21BlockWeights,
    hidden_states: &MlxArray,
    context: &BlockContext<'_>,
) -> Result<MlxArray, QwenImage21EngineError> {
    let attention_input =
        modulate_layer_norm(runtime, hidden_states, context.attention_one_plus_scale)?;
    let attention_output =
        forward_attention(runtime, &weights.attention, &attention_input, context)?;
    let gated_attention = runtime.multiply(&attention_output, context.attention_gate)?;
    let hidden_after_attention = runtime.add(hidden_states, &gated_attention)?;

    let feed_forward_input = modulate_layer_norm(
        runtime,
        &hidden_after_attention,
        context.feed_forward_one_plus_scale,
    )?;
    let feed_forward_output =
        forward_feed_forward(runtime, &weights.feed_forward, &feed_forward_input)?;
    let gated_feed_forward = runtime.multiply(&feed_forward_output, context.feed_forward_gate)?;
    Ok(runtime.add(&hidden_after_attention, &gated_feed_forward)?)
}

/// `layer_norm(x) * (1 + scale)` — the reference `_modulate` with no shift.
fn modulate_layer_norm(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    one_plus_scale: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let normalized = fp32_layer_norm(runtime, hidden_states, LAYER_NORM_EPSILON)?;
    Ok(runtime.multiply(&normalized, one_plus_scale)?)
}

fn forward_attention(
    runtime: &MlxRuntime,
    weights: &QwenImage21AttentionWeights,
    modulated: &MlxArray,
    context: &BlockContext<'_>,
) -> Result<MlxArray, QwenImage21EngineError> {
    let queries = weights.to_query.forward(runtime, modulated)?;
    let keys = weights.to_key.forward(runtime, modulated)?;
    let values = weights.to_value.forward(runtime, modulated)?;

    let shape = queries.shape();
    let (batch, sequence_length) = (shape[0], shape[1]);
    let head_shape = |array: &MlxArray| -> Result<MlxArray, QwenImage21EngineError> {
        Ok(runtime.reshape(
            array,
            &[
                batch,
                sequence_length,
                context.head_count as i32,
                context.head_width as i32,
            ],
        )?)
    };
    let queries = head_shape(&queries)?;
    let keys = head_shape(&keys)?;
    let values = head_shape(&values)?;

    let queries = fp32_rms_norm(runtime, &queries, &weights.norm_query, LAYER_NORM_EPSILON)?;
    let keys = fp32_rms_norm(runtime, &keys, &weights.norm_key, LAYER_NORM_EPSILON)?;
    let queries = apply_rope(runtime, &queries, context.rope_cosines, context.rope_sines)?;
    let keys = apply_rope(runtime, &keys, context.rope_cosines, context.rope_sines)?;

    let attended = segment_attention(runtime, &queries, &keys, &values, context)?;
    let attended_flat = runtime.reshape(&attended, &[batch, sequence_length, shape[2]])?;
    weights.to_output.forward(runtime, &attended_flat)
}

/// The prefill decomposition: each segment attends to `[0, end)` keys — causal for text,
/// fully bidirectional for image blocks — and the trailing target segment attends to everything.
fn segment_attention(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    context: &BlockContext<'_>,
) -> Result<MlxArray, QwenImage21EngineError> {
    let (batch, sequence_length, head_count, head_width) = (
        queries.shape()[0],
        queries.shape()[1],
        context.head_count as i32,
        context.head_width as i32,
    );
    let to_heads = |array: &MlxArray| -> Result<MlxArray, QwenImage21EngineError> {
        Ok(runtime.transpose_axes(array, &[0, 2, 1, 3])?)
    };
    let keys_headed = to_heads(keys)?;
    let values_headed = to_heads(values)?;

    let mut outputs = Vec::with_capacity(context.segments.len() + 1);
    for (segment, mask) in context.segments.iter().zip(context.segment_masks.iter()) {
        let (query_start, query_end, key_end) = (
            segment.query_start as i32,
            segment.query_end as i32,
            segment.key_end as i32,
        );
        let segment_queries = to_heads(&runtime.slice(
            queries,
            &[0, query_start, 0, 0],
            &[batch, query_end, head_count, head_width],
            &[1, 1, 1, 1],
        )?)?;
        let segment_keys = runtime.slice(
            &keys_headed,
            &[0, 0, 0, 0],
            &[batch, head_count, key_end, head_width],
            &[1, 1, 1, 1],
        )?;
        let segment_values = runtime.slice(
            &values_headed,
            &[0, 0, 0, 0],
            &[batch, head_count, key_end, head_width],
            &[1, 1, 1, 1],
        )?;
        let attended = if segment.is_text {
            masked_attention(
                runtime,
                &segment_queries,
                &segment_keys,
                &segment_values,
                mask,
                context.attention_scale,
            )?
        } else {
            fused_attention(
                runtime,
                &segment_queries,
                &segment_keys,
                &segment_values,
                context.attention_scale,
            )?
        };
        let attended_tokens = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
        outputs.push(attended_tokens);
    }
    // The trailing target segment attends to every key.
    let target_queries = to_heads(
        &runtime.slice(
            queries,
            &[
                0,
                context
                    .segments
                    .last()
                    .map(|segment| segment.query_end)
                    .unwrap_or_default() as i32,
                0,
                0,
            ],
            &[batch, sequence_length, head_count, head_width],
            &[1, 1, 1, 1],
        )?,
    )?;
    let target_attended = fused_attention(
        runtime,
        &target_queries,
        &keys_headed,
        &values_headed,
        context.attention_scale,
    )?;
    let target_tokens = runtime.transpose_axes(&target_attended, &[0, 2, 1, 3])?;
    outputs.push(target_tokens);

    let output_refs = outputs.iter().collect::<Vec<_>>();
    Ok(runtime.concatenate_axis(&output_refs, 1)?)
}

fn forward_feed_forward(
    runtime: &MlxRuntime,
    weights: &QwenImage21FeedForwardWeights,
    modulated: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let gated = weights.gate_layer.forward(runtime, modulated)?;
    let activated = runtime.silu(&gated)?;
    let projected = weights.projection.forward(runtime, modulated)?;
    let combined = runtime.multiply(&activated, &projected)?;
    weights.output.forward(runtime, &combined)
}
