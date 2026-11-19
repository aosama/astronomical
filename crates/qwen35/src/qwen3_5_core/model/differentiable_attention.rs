//! Full attention in plain MLX operations, for a decision model's training
//! graph.
//!
//! For short runs of queries MLX picks its vector attention kernel, which
//! has no backward pass, so a gradient graph through it fails. The same
//! formula in plain operations differentiates everywhere and matches the
//! fused kernel to float32 rounding (the tests pin it). Serving never takes
//! this route.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::MlxArray;

/// Which kernels compute the attention scores and their weighted values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5AttentionKernel {
    /// MLX's fused attention: what serving runs.
    Fused,
    /// The same formula in plain operations, for a decision model's training
    /// graph: for short query runs MLX picks its vector attention kernel,
    /// which has no backward pass. Scores a training compares with serving
    /// never take this route.
    PlainOperations,
}

/// `softmax(Q·Kᵀ · scale + causal mask) · V` in plain operations, in
/// float32: queries `[batch, heads, Lq, D]` against grouped keys and values
/// `[batch, key value heads, Lk, D]`, each query seeing the keys up to its
/// own position (the last `Lq` keys are the queries' own).
pub fn plain_operation_attention(
    runtime: &MlxRuntime,
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    attention_scale: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    let group_size = rotated_queries.shape()[1] / active_keys.shape()[1];
    let grouped_keys = grouped_float32_heads(runtime, active_keys, group_size)?;
    let grouped_values = grouped_float32_heads(runtime, active_values, group_size)?;
    let float32_queries = runtime.astype(rotated_queries, mlx_bridge::MlxDtype::Float32)?;
    let scores = runtime.multiply_scalar(
        &runtime.matmul(
            &float32_queries,
            &runtime.transpose_axes(&grouped_keys, &[0, 1, 3, 2])?,
        )?,
        attention_scale,
    )?;
    let causal_mask = causal_mask(runtime, rotated_queries.shape()[2], active_keys.shape()[2])?;
    let weights = runtime.softmax_axis(&runtime.add(&scores, &causal_mask)?, -1)?;
    Ok(runtime.astype(
        &runtime.matmul(&weights, &grouped_values)?,
        rotated_queries.dtype(),
    )?)
}

/// Keys or values in float32, each key-value head repeated for the query
/// heads that share it.
fn grouped_float32_heads(
    runtime: &MlxRuntime,
    key_value_heads: &MlxArray,
    group_size: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let float32_heads = runtime.astype(key_value_heads, mlx_bridge::MlxDtype::Float32)?;
    Ok(runtime.repeat_axis(&float32_heads, group_size, 1)?)
}

/// `[Lq, Lk]`: zero where a query may see a key, minus infinity past its
/// own position; the queries are the last `Lq` keys.
fn causal_mask(
    runtime: &MlxRuntime,
    query_token_count: i32,
    key_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let first_query_position = key_token_count - query_token_count;
    let mut mask_values = vec![0.0_f32; (query_token_count * key_token_count) as usize];
    for query_index in 0..query_token_count {
        for key_index in (first_query_position + query_index + 1)..key_token_count {
            mask_values[(query_index * key_token_count + key_index) as usize] = f32::NEG_INFINITY;
        }
    }
    Ok(runtime.array_from_f32(&mask_values, &[query_token_count, key_token_count])?)
}

#[cfg(test)]
#[path = "differentiable_attention_tests.rs"]
mod tests;
