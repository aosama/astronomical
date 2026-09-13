//! Sparse attention over indexer-selected keys for `qwen4_exp`.
//!
//! The production attention gathers only the selected keys and values per
//! query, so the cost follows the budget rather than the context length.
//! This owner takes the projected queries, keys, and values plus the
//! selection owner's indices and computes attention over the gathered
//! subset, restoring the result to token order.
//!
//! Parity is proven against the oracle's explicit full attention restricted
//! to the same selected keys, so a gather that drops a selected key, a
//! softmax over the wrong axis, or a mis-ordered restore fails here.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

/// Computes attention over the selected keys for every query.
///
/// Shapes: queries `[token_count, head_count, head_dim]`, keys and values
/// `[token_count, key_value_head_count, head_dim]`, selected indices
/// `[token_count, selected_count]`. The result is
/// `[token_count, head_count, head_dim]`, in original token order.
///
/// The key-value head count may differ from the query head count; each
/// query head reads its grouped key-value head through a one-hot head map,
/// so the grouped-query contract holds inside the sparse path too.
///
/// # Errors
/// When any MLX operation fails or a shape disagrees with the geometry.
#[allow(clippy::too_many_lines)]
pub fn sparse_attention(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    selected_indices: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution.measure_operation(PerformanceOperation::Qwen4ExpSparseAttention, |_| {
        sparse_attention_inner(runtime, queries, keys, values, selected_indices)
    })
}

fn sparse_attention_inner(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    selected_indices: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let selection_shape = selected_indices.shape();
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "qwen4_exp sparse attention",
            description: format!(
                "queries, keys, and values must be three-dimensional: {query_shape:?}, {key_shape:?}, {value_shape:?}"
            ),
        });
    }
    let token_count = query_shape[0];
    let head_count = query_shape[1];
    let head_dim = query_shape[2];
    let key_value_head_count = key_shape[1];
    let selected_count = selection_shape[1];
    if key_shape[0] != token_count || value_shape[0] != token_count {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "qwen4_exp sparse attention",
            description: format!(
                "keys and values must carry {token_count} tokens: {key_shape:?}, {value_shape:?}"
            ),
        });
    }
    if key_shape[2] != head_dim || value_shape[2] != head_dim {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "qwen4_exp sparse attention",
            description: format!(
                "keys and values must carry head dimension {head_dim}: {key_shape:?}, {value_shape:?}"
            ),
        });
    }
    if selection_shape[0] != token_count {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "qwen4_exp sparse attention",
            description: format!(
                "selected indices must carry {token_count} queries: {selection_shape:?}"
            ),
        });
    }
    // Gather the selected keys and values per query through a one-hot
    // matrix product: [token_count, selected_count, token_count] by
    // [token_count, key_value_head_count * head_dim] yields
    // [token_count, selected_count, key_value_head_count * head_dim]. This
    // is the batched equivalent of gathering rows, uses one matmul per
    // tensor, and avoids per-token loops entirely.
    let flattened_key_count = key_value_head_count * head_dim;
    let keys_flat = runtime.reshape(keys, &[token_count, flattened_key_count as i32])?;
    let values_flat = runtime.reshape(values, &[token_count, flattened_key_count as i32])?;
    let positions = runtime.arange_i32(0, token_count)?;
    let position_axis = runtime.expand_dims(&positions, 0)?;
    let index_axis = runtime.expand_dims(selected_indices, -1)?;
    // Equality is the product of the two directional comparisons the
    // runtime provides, since a dedicated equality op is not on the surface.
    let forward = runtime.greater_equal(&index_axis, &position_axis)?;
    let backward = runtime.greater_equal(&position_axis, &index_axis)?;
    let one_hot = runtime.multiply(&forward, &backward)?;
    let one_hot = runtime.astype(&one_hot, keys.dtype())?;
    let gathered_keys_flat = runtime.matmul(&one_hot, &keys_flat)?;
    let gathered_values_flat = runtime.matmul(&one_hot, &values_flat)?;
    let gathered_keys = runtime.reshape(
        &gathered_keys_flat,
        &[
            token_count,
            selected_count,
            key_value_head_count as i32,
            head_dim as i32,
        ],
    )?;
    let gathered_values = runtime.reshape(
        &gathered_values_flat,
        &[
            token_count,
            selected_count,
            key_value_head_count as i32,
            head_dim as i32,
        ],
    )?;
    // Map grouped key-value heads to query heads through a small one-hot
    // product, then arrange per-head batches for the attention matmuls:
    // gathered [token_count, selected_count, key_value_head_count,
    // head_dim] becomes [token_count, head_count, selected_count, head_dim].
    let group_size = head_count / key_value_head_count;
    let head_indices: Vec<i32> = (0..head_count)
        .map(|head| (head / group_size) as i32)
        .collect();
    let head_index_array = runtime.array_from_i32(&head_indices, &[head_count as i32])?;
    let kv_positions = runtime.arange_i32(0, key_value_head_count as i32)?;
    let forward_map = runtime.greater_equal(
        &runtime.expand_dims(&head_index_array, 1)?,
        &runtime.expand_dims(&kv_positions, 0)?,
    )?;
    let backward_map = runtime.greater_equal(
        &runtime.expand_dims(&kv_positions, 0)?,
        &runtime.expand_dims(&head_index_array, 1)?,
    )?;
    let head_map = runtime.multiply(&forward_map, &backward_map)?;
    let head_map = runtime.astype(&head_map, keys.dtype())?;
    // [1, 1, head_count, key_value_head_count] by
    // [token_count, selected_count, key_value_head_count, head_dim]
    // broadcasts the batch and contracts the key-value-head axis.
    let map_broadcast = runtime.expand_dims(&runtime.expand_dims(&head_map, 0)?, 0)?;
    let mapped_keys = runtime.matmul(&map_broadcast, &gathered_keys)?;
    let mapped_values = runtime.matmul(&map_broadcast, &gathered_values)?;
    // Per-head batches: [token_count, head_count, selected_count, head_dim].
    let keys_by_head = runtime.transpose_axes(&mapped_keys, &[0, 2, 1, 3])?;
    let values_by_head = runtime.transpose_axes(&mapped_values, &[0, 2, 1, 3])?;
    let queries_by_head = runtime.expand_dims(queries, 2)?;
    // Attention per query over its selected keys: scores by matmul against
    // the transposed selected keys, softmax over the selected-key axis, then
    // a weighted sum of the selected values. No causal mask is needed here:
    // the selection owner already enforced causality, so every gathered key
    // is visible to its query by construction.
    let transposed_keys = runtime.transpose_axes(&keys_by_head, &[0, 1, 3, 2])?;
    let scores = runtime.matmul(&queries_by_head, &transposed_keys)?;
    let scale = runtime.array_from_f32(&vec![(1.0 / (head_dim as f64).sqrt()) as f32], &[1])?;
    let scores = runtime.multiply(&scores, &scale)?;
    let probabilities = runtime.softmax_axis(&scores, -1)?;
    let attended = runtime.matmul(&probabilities, &values_by_head)?;
    // Restore token-major order: [token_count, head_count, head_dim].
    runtime.squeeze_axis(&attended, 2)
}
