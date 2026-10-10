use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

const GATED_DELTA_STEP_OPERATION: &str = "apply one Qwen3.5 gated-delta recurrent step";

/// Forces one gated-delta section's graphics-processor work so its GPU time is
/// attributed to its own operation instead of folding into the family wait or
/// the chunk-terminal wait. Multi-token prefill only: one-token decode keeps
/// the latency-sensitive step free of host synchronization.
pub(crate) fn evaluate_linear_attention_section(
    runtime: &MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
    operation: PerformanceOperation,
    arrays: &[&MlxArray],
    token_count: i32,
) -> Result<(), MlxRuntimeError> {
    if !performance_attribution.is_enabled() || token_count <= 1 {
        return Ok(());
    }
    performance_attribution.measure_operation(operation, |_performance_attribution| {
        runtime.evaluate_arrays(arrays)
    })?;
    Ok(())
}

/// Applies one ops-based Qwen3.5 gated-delta recurrence while retaining state in float32.
pub fn qwen3_5_gated_delta_step(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let repeat_factor =
        validate_gated_delta_shapes(queries, keys, values, decays, update_rates, recurrent_state)?;
    let output_dtype = queries.dtype();
    let float32_queries = runtime.astype(queries, MlxDtype::Float32)?;
    let float32_keys = runtime.astype(keys, MlxDtype::Float32)?;
    let float32_values = runtime.astype(values, MlxDtype::Float32)?;
    let float32_decays = runtime.astype(decays, MlxDtype::Float32)?;
    let float32_update_rates = runtime.astype(update_rates, MlxDtype::Float32)?;
    let repeated_queries = runtime.repeat_axis(&float32_queries, repeat_factor, 1)?;
    let repeated_keys = runtime.repeat_axis(&float32_keys, repeat_factor, 1)?;

    let decay_rows = runtime.expand_dims(&float32_decays, -1)?;
    let decay_matrices = runtime.expand_dims(&decay_rows, -1)?;
    let decayed_state = runtime.multiply(recurrent_state, &decay_matrices)?;
    let expanded_keys = runtime.expand_dims(&repeated_keys, 2)?;
    let state_key_products = runtime.multiply(&decayed_state, &expanded_keys)?;
    let remembered_values = runtime.sum_axis(&state_key_products, -1, false)?;
    let value_differences = runtime.subtract(&float32_values, &remembered_values)?;
    let expanded_update_rates = runtime.expand_dims(&float32_update_rates, -1)?;
    let value_updates = runtime.multiply(&value_differences, &expanded_update_rates)?;
    let expanded_value_updates = runtime.expand_dims(&value_updates, -1)?;
    let state_updates = runtime.multiply(&expanded_keys, &expanded_value_updates)?;
    let next_recurrent_state = runtime.add(&decayed_state, &state_updates)?;

    let expanded_queries = runtime.expand_dims(&repeated_queries, 2)?;
    let state_query_products = runtime.multiply(&next_recurrent_state, &expanded_queries)?;
    let float32_output = runtime.sum_axis(&state_query_products, -1, false)?;
    let output = runtime.astype(&float32_output, output_dtype)?;
    Ok((output, next_recurrent_state))
}

fn validate_gated_delta_shapes(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<i32, MlxRuntimeError> {
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let decay_shape = decays.shape();
    let update_rate_shape = update_rates.shape();
    let recurrent_state_shape = recurrent_state.shape();
    if query_shape.len() != 3
        || value_shape.len() != 3
        || decay_shape.len() != 2
        || recurrent_state_shape.len() != 4
    {
        return Err(gated_delta_error(
            "queries, values, decays, and recurrent state must have ranks three, three, two, and four",
        ));
    }
    if key_shape != query_shape {
        return Err(gated_delta_error(
            "gated-delta queries and keys must have identical shapes",
        ));
    }
    if update_rate_shape != decay_shape {
        return Err(gated_delta_error(
            "gated-delta update rates and decays must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let key_head_count = query_shape[1];
    let key_dimension = query_shape[2];
    let value_head_count = value_shape[1];
    let value_dimension = value_shape[2];
    if batch_size <= 0
        || key_head_count <= 0
        || key_dimension <= 0
        || value_head_count <= 0
        || value_dimension <= 0
        || value_head_count % key_head_count != 0
    {
        return Err(gated_delta_error(
            "gated-delta dimensions must be positive and value heads must divide evenly by key heads",
        ));
    }
    if value_shape[0] != batch_size
        || decay_shape != [batch_size, value_head_count]
        || recurrent_state_shape != [batch_size, value_head_count, value_dimension, key_dimension]
    {
        return Err(gated_delta_error(
            "gated-delta value, decay, and recurrent-state dimensions are incompatible",
        ));
    }
    if recurrent_state.dtype() != MlxDtype::Float32 {
        return Err(gated_delta_error(
            "gated-delta recurrent state must use float32",
        ));
    }
    if !is_supported_activation_dtype(queries.dtype())
        || !is_supported_activation_dtype(keys.dtype())
        || !is_supported_activation_dtype(values.dtype())
        || !is_supported_activation_dtype(decays.dtype())
        || !is_supported_activation_dtype(update_rates.dtype())
    {
        return Err(gated_delta_error(
            "gated-delta inputs must use float16, bfloat16, or float32",
        ));
    }
    Ok(value_head_count / key_head_count)
}

fn is_supported_activation_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}

pub(crate) fn gated_delta_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: GATED_DELTA_STEP_OPERATION,
        description: description.to_owned(),
    }
}
