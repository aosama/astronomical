//! Exact dense Qwen3 decoder layer using native MLX fused attention and RoPE.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::Flux2KleinTextConditioningError;
use super::weights::{
    Flux2KleinDecoderLayerWeights, HEAD_WIDTH, KEY_VALUE_HEAD_COUNT, QUERY_HEAD_COUNT,
};

const RMS_NORM_EPSILON: f32 = 0.000_001;
const ROPE_THETA: f32 = 1_000_000.0;

pub(super) struct Flux2KleinAttentionOutput {
    attention_residual: MlxArray,
    normalized_attention: MlxArray,
}

pub(super) fn forward_decoder_layer_attention(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    combined_attention_mask: &MlxArray,
    batch_size: i32,
    sequence_length: i32,
    weights: &Flux2KleinDecoderLayerWeights,
) -> Result<Flux2KleinAttentionOutput, Flux2KleinTextConditioningError> {
    let normalized_input =
        runtime.rms_norm(hidden_states, &weights.input_norm, RMS_NORM_EPSILON)?;
    let queries = project_heads(
        runtime,
        &normalized_input,
        &weights.query,
        batch_size,
        sequence_length,
        QUERY_HEAD_COUNT,
    )?;
    let keys = project_heads(
        runtime,
        &normalized_input,
        &weights.key,
        batch_size,
        sequence_length,
        KEY_VALUE_HEAD_COUNT,
    )?;
    let values = project_heads(
        runtime,
        &normalized_input,
        &weights.value,
        batch_size,
        sequence_length,
        KEY_VALUE_HEAD_COUNT,
    )?;
    let normalized_queries = runtime.rms_norm(&queries, &weights.query_norm, RMS_NORM_EPSILON)?;
    let normalized_keys = runtime.rms_norm(&keys, &weights.key_norm, RMS_NORM_EPSILON)?;
    let rotated_queries = runtime.rope(&normalized_queries, HEAD_WIDTH, ROPE_THETA, 0)?;
    let rotated_keys = runtime.rope(&normalized_keys, HEAD_WIDTH, ROPE_THETA, 0)?;
    let attention = runtime.masked_scaled_dot_product_attention(
        &rotated_queries,
        &rotated_keys,
        &values,
        (HEAD_WIDTH as f32).sqrt().recip(),
        combined_attention_mask,
    )?;
    let token_major_attention = runtime.transpose_axes(&attention, &[0, 2, 1, 3])?;
    let joined_attention = runtime.reshape(
        &token_major_attention,
        &[batch_size, sequence_length, QUERY_HEAD_COUNT * HEAD_WIDTH],
    )?;
    let attention_output = linear(runtime, &joined_attention, &weights.output)?;
    let attention_residual = runtime.add(hidden_states, &attention_output)?;
    let normalized_attention = runtime.rms_norm(
        &attention_residual,
        &weights.post_attention_norm,
        RMS_NORM_EPSILON,
    )?;
    Ok(Flux2KleinAttentionOutput {
        attention_residual,
        normalized_attention,
    })
}

pub(super) fn forward_decoder_layer_feed_forward(
    runtime: &MlxRuntime,
    attention_output: &Flux2KleinAttentionOutput,
    weights: &Flux2KleinDecoderLayerWeights,
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let gate = linear(
        runtime,
        &attention_output.normalized_attention,
        &weights.gate,
    )?;
    let up = linear(runtime, &attention_output.normalized_attention, &weights.up)?;
    let activated_gate = runtime.silu(&gate)?;
    let gated_up = runtime.multiply(&activated_gate, &up)?;
    let feed_forward_output = linear(runtime, &gated_up, &weights.down)?;
    Ok(runtime.add(&attention_output.attention_residual, &feed_forward_output)?)
}

pub(super) fn build_causal_padding_mask(
    runtime: &MlxRuntime,
    attention_mask: &[u32],
    batch_size: i32,
    sequence_length: i32,
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let batch_size_usize = usize::try_from(batch_size)
        .map_err(|_| Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
    let sequence_length_usize = usize::try_from(sequence_length)
        .map_err(|_| Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
    let expected_attention_mask_length = batch_size_usize
        .checked_mul(sequence_length_usize)
        .ok_or(Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
    if batch_size_usize == 0
        || sequence_length_usize == 0
        || attention_mask.len() != expected_attention_mask_length
    {
        return Err(Flux2KleinTextConditioningError::BatchGeometryOverflow);
    }
    let mask_element_count = batch_size_usize
        .checked_mul(sequence_length_usize)
        .and_then(|count| count.checked_mul(sequence_length_usize))
        .ok_or(Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
    let mut combined_mask_values = Vec::with_capacity(mask_element_count);
    for batch_index in 0..batch_size_usize {
        let row_start = batch_index * sequence_length_usize;
        for query_position in 0..sequence_length_usize {
            for key_position in 0..sequence_length_usize {
                combined_mask_values.push(u32::from(
                    key_position <= query_position && attention_mask[row_start + key_position] != 0,
                ));
            }
        }
    }
    let integer_mask = runtime.array_from_u32(
        &combined_mask_values,
        &[batch_size, 1, sequence_length, sequence_length],
    )?;
    Ok(runtime.astype(&integer_mask, MlxDtype::Bool)?)
}

fn project_heads(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weight: &MlxArray,
    batch_size: i32,
    sequence_length: i32,
    head_count: i32,
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let projected = linear(runtime, hidden_states, weight)?;
    let token_major = runtime.reshape(
        &projected,
        &[batch_size, sequence_length, head_count, HEAD_WIDTH],
    )?;
    Ok(runtime.transpose_axes(&token_major, &[0, 2, 1, 3])?)
}

fn linear(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    weight: &MlxArray,
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let transposed_weight = runtime.transpose_axes(weight, &[1, 0])?;
    Ok(runtime.matmul(activations, &transposed_weight)?)
}
