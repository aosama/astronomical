use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{MlxArray, MlxDtype};

use super::gated_delta_input_rules::{
    GatedDeltaDtypeViolation, GatedDeltaHeadGeometry, check_gated_delta_dtypes,
};
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

/// Validates the step's inputs and returns how many value heads share each
/// key head. Shapes are checked before dtypes, as before the shared rules.
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
    let repeat_factor = repeat_factor_from_step_input_shapes(&GatedDeltaStepInputShapes {
        query_shape: &query_shape,
        key_shape: &key_shape,
        value_shape: &value_shape,
        decay_shape: &decay_shape,
        update_rate_shape: &update_rate_shape,
        recurrent_state_shape: &recurrent_state_shape,
    })?;
    validate_step_dtypes(
        recurrent_state.dtype(),
        [
            queries.dtype(),
            keys.dtype(),
            values.dtype(),
            decays.dtype(),
            update_rates.dtype(),
        ],
    )?;
    Ok(repeat_factor)
}

/// The shapes of the step's six inputs, read once from their arrays so the
/// rules below are plain data a unit test can drive.
struct GatedDeltaStepInputShapes<'shape> {
    query_shape: &'shape [i32],
    key_shape: &'shape [i32],
    value_shape: &'shape [i32],
    decay_shape: &'shape [i32],
    update_rate_shape: &'shape [i32],
    recurrent_state_shape: &'shape [i32],
}

/// The step has no token axis: queries and keys are `[batch, key heads, key
/// dim]`, values `[batch, value heads, value dim]`, decays and update rates
/// `[batch, value heads]`.
fn repeat_factor_from_step_input_shapes(
    input_shapes: &GatedDeltaStepInputShapes<'_>,
) -> Result<i32, MlxRuntimeError> {
    let GatedDeltaStepInputShapes {
        query_shape,
        key_shape,
        value_shape,
        decay_shape,
        update_rate_shape,
        recurrent_state_shape,
    } = *input_shapes;
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
    let head_geometry = GatedDeltaHeadGeometry {
        batch_size: query_shape[0],
        key_head_count: query_shape[1],
        key_head_dimension: query_shape[2],
        value_head_count: value_shape[1],
        value_head_dimension: value_shape[2],
    };
    if head_geometry.check().is_err() {
        return Err(gated_delta_error(
            "gated-delta dimensions must be positive and value heads must divide evenly by key heads",
        ));
    }
    if value_shape[0] != head_geometry.batch_size
        || decay_shape != [head_geometry.batch_size, head_geometry.value_head_count]
        || recurrent_state_shape != head_geometry.expected_recurrent_state_shape()
    {
        return Err(gated_delta_error(
            "gated-delta value, decay, and recurrent-state dimensions are incompatible",
        ));
    }
    Ok(head_geometry.value_head_count / head_geometry.key_head_count)
}

/// Applies the shared dtype rules with the step's error text.
fn validate_step_dtypes(
    recurrent_state_dtype: MlxDtype,
    activation_dtypes: [MlxDtype; 5],
) -> Result<(), MlxRuntimeError> {
    match check_gated_delta_dtypes(recurrent_state_dtype, activation_dtypes) {
        Ok(()) => Ok(()),
        Err(GatedDeltaDtypeViolation::RecurrentStateNotFloat32) => Err(gated_delta_error(
            "gated-delta recurrent state must use float32",
        )),
        Err(GatedDeltaDtypeViolation::UnsupportedActivationDtype) => Err(gated_delta_error(
            "gated-delta inputs must use float16, bfloat16, or float32",
        )),
    }
}

pub(crate) fn gated_delta_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: GATED_DELTA_STEP_OPERATION,
        description: description.to_owned(),
    }
}

#[cfg(test)]
#[path = "gated_delta_step_tests.rs"]
mod gated_delta_step_tests;
