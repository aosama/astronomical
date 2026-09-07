//! Quantized-attention passes for K2 Horizon MoVA decode.
//!
//! The score and value passes read the packed 4-bit KV slab through the
//! quantized matmul kernel so long-context decode attention reads one bit
//! width instead of bfloat16. Owned by the quantized KV state's views; only
//! reachable at decode because chunked prefill keeps the fused bf16 SDPA.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

use super::error::K2HorizonMoVAExecutionError;

pub(super) fn quantized_scaled_dot_product_attention(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    queries: &MlxArray,
    views: &crate::decoder_cache::QuantizedKeyValueViews,
    group_size: i32,
    bits: i32,
    is_prefill: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let scale = (config.head_dim() as f32).sqrt().recip();
    let scaled_queries = runtime.multiply_scalar(queries, scale)?;
    let query_head_count = config.num_attention_heads();
    let key_value_head_count = config.num_key_value_heads();
    let repeats = query_head_count / key_value_head_count;
    // GQA: group query heads under their key/value head so the batched
    // quantized matmul broadcasts once instead of copying the packed slabs.
    let (scaled_queries, keys, values) = if repeats > 1 {
        let reshaped = runtime.reshape(
            &scaled_queries,
            &[
                1,
                key_value_head_count as i32,
                repeats as i32,
                scaled_queries.shape().get(2).copied().unwrap_or(1),
                config.head_dim() as i32,
            ],
        )?;
        (
            reshaped,
            expanded_quantized_views(runtime, &views.keys)?,
            expanded_quantized_views(runtime, &views.values)?,
        )
    } else {
        (
            scaled_queries,
            retained_quantized_views(&views.keys)?,
            retained_quantized_views(&views.values)?,
        )
    };
    let mut scores = runtime.quantized_matmul_affine(
        &scaled_queries,
        &keys.packed,
        &keys.scales,
        &keys.biases,
        true,
        group_size,
        bits,
    )?;
    if is_prefill {
        let total_sequence_tokens = scores.shape().last().copied().unwrap_or(1);
        let query_axis = scores.shape().len().saturating_sub(2);
        let query_token_count = scores.shape().get(query_axis).copied().unwrap_or(1);
        scores = apply_causal_mask(runtime, &scores, query_token_count, total_sequence_tokens)?;
    }
    let probabilities = runtime.softmax_axis(&scores, -1)?;
    let output = runtime.quantized_matmul_affine(
        &probabilities,
        &values.packed,
        &values.scales,
        &values.biases,
        false,
        group_size,
        bits,
    )?;
    if repeats > 1 {
        let output_axis = output.shape().len().saturating_sub(2);
        let query_token_count = output.shape().get(output_axis).copied().unwrap_or(1);
        return runtime
            .reshape(
                &output,
                &[
                    1,
                    query_head_count as i32,
                    query_token_count,
                    config.head_dim() as i32,
                ],
            )
            .map_err(|error| K2HorizonMoVAExecutionError::InvalidExecution {
                description: format!("K2 Horizon MoVA quantized attention reshape failed: {error}"),
            });
    }
    Ok(output)
}

pub(super) fn expanded_quantized_views(
    runtime: &MlxRuntime,
    views: &crate::decoder_cache::QuantizedTensorViews,
) -> Result<crate::decoder_cache::QuantizedTensorViews, K2HorizonMoVAExecutionError> {
    Ok(crate::decoder_cache::QuantizedTensorViews {
        packed: runtime.expand_dims(&views.packed, -3)?,
        scales: runtime.expand_dims(&views.scales, -3)?,
        biases: runtime.expand_dims(&views.biases, -3)?,
    })
}

pub(super) fn retained_quantized_views(
    views: &crate::decoder_cache::QuantizedTensorViews,
) -> Result<crate::decoder_cache::QuantizedTensorViews, K2HorizonMoVAExecutionError> {
    Ok(crate::decoder_cache::QuantizedTensorViews {
        packed: views.packed.retain()?,
        scales: views.scales.retain()?,
        biases: views.biases.retain()?,
    })
}

fn apply_causal_mask(
    runtime: &MlxRuntime,
    scores: &MlxArray,
    query_token_count: i32,
    total_sequence_tokens: i32,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let query_indices = runtime.arange_i32(
        total_sequence_tokens.saturating_sub(query_token_count),
        total_sequence_tokens,
    )?;
    let key_indices = runtime.arange_i32(0, total_sequence_tokens)?;
    let query_row = runtime.expand_dims(&query_indices, -1)?;
    let key_row = runtime.expand_dims(&key_indices, 0)?;
    let mask = runtime.greater_equal(&query_row, &key_row)?;
    let negative_limit = runtime.full(&[1], -3.4e38, scores.dtype())?;
    runtime
        .where_select(&mask, scores, &negative_limit)
        .map_err(|error| K2HorizonMoVAExecutionError::InvalidExecution {
            description: format!("K2 Horizon MoVA causal mask failed: {error}"),
        })
}
