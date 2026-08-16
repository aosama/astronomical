//! Feature-gated MLX builder for the CPU sliding-window visibility contract.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

/// Builds a `[1, 1, query_tokens, key_tokens]` boolean mask from absolute positions.
pub fn build_causal_sliding_window_mask(
    runtime: &MlxRuntime,
    first_query_absolute_position: i32,
    query_token_count: i32,
    first_key_absolute_position: i32,
    key_token_count: i32,
    window_size: i32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution.measure_operation(
        PerformanceOperation::SlidingWindowMaskConstruction,
        |_| {
            build_causal_sliding_window_mask_inner(
                runtime,
                first_query_absolute_position,
                query_token_count,
                first_key_absolute_position,
                key_token_count,
                window_size,
            )
        },
    )
}

fn build_causal_sliding_window_mask_inner(
    runtime: &MlxRuntime,
    first_query_absolute_position: i32,
    query_token_count: i32,
    first_key_absolute_position: i32,
    key_token_count: i32,
    window_size: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    if window_size <= 0 || query_token_count <= 0 || key_token_count <= 0 {
        return Err(mask_error("window size and token counts must be positive"));
    }
    let last_query_exclusive = first_query_absolute_position
        .checked_add(query_token_count)
        .ok_or_else(|| mask_error("query absolute range overflowed"))?;
    let last_key_exclusive = first_key_absolute_position
        .checked_add(key_token_count)
        .ok_or_else(|| mask_error("key absolute range overflowed"))?;
    let query_absolute_positions =
        runtime.arange_i32(first_query_absolute_position, last_query_exclusive)?;
    let key_absolute_positions =
        runtime.arange_i32(first_key_absolute_position, last_key_exclusive)?;
    let query_rows = runtime.reshape(&query_absolute_positions, &[query_token_count, 1])?;
    let key_columns = runtime.reshape(&key_absolute_positions, &[1, key_token_count])?;
    let causal_mask = runtime.greater_equal(&query_rows, &key_columns)?;
    let window_size_array = runtime.array_from_i32(&[window_size], &[])?;
    let window_limit = runtime.add(&key_columns, &window_size_array)?;
    let window_mask = runtime.less(&query_rows, &window_limit)?;
    // `where(causal, window, false)` is boolean AND without numeric casts.
    let false_mask = runtime.zeros(&[query_token_count, key_token_count], MlxDtype::Bool)?;
    let combined_mask = runtime.where_select(&causal_mask, &window_mask, &false_mask)?;
    runtime.reshape(&combined_mask, &[1, 1, query_token_count, key_token_count])
}

fn mask_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "build a causal sliding-window attention mask",
        description: description.to_owned(),
    }
}
