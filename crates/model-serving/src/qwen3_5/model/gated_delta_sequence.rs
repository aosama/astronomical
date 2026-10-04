use astronomical_runtime_integration::{
    MlxMetalKernel, MlxMetalKernelOutput, MlxRuntime, MlxRuntimeError,
};

use super::gated_delta_pipelined_kernel::{
    THREADGROUP_THREAD_COUNT, VALUE_ROW_BLOCK_SIZE, gated_delta_pipelined_kernel_source,
};
use super::gated_delta_sequence_contract::{
    GatedDeltaSequenceShape, gated_delta_sequence_error, template_arguments,
    validate_gated_delta_sequence_shapes,
};
use astronomical_mlx_c_rust::MlxArray;

/// Builds the fused Qwen3.5 gated-delta sequence kernel.
pub fn qwen3_5_gated_delta_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    let ordinary_kernel_source = gated_delta_pipelined_kernel_source("", "");
    MlxMetalKernel::new(
        "astronomical_qwen3_5_gated_delta_sequence",
        &[
            "queries",
            "keys",
            "values",
            "decays",
            "update_rates",
            "recurrent_state",
            "token_count",
        ],
        &["outputs", "next_recurrent_state"],
        &ordinary_kernel_source,
    )
}

/// Applies fused Qwen3.5 gated-delta recurrence across one prompt/decode
/// sequence. A retained kernel uses the fused Metal route; a
/// capability-demoted kernel — `None` — uses the ops-based public MLX route.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence(
    runtime: &MlxRuntime,
    gated_delta_kernel: Option<&MlxMetalKernel>,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    match gated_delta_kernel {
        Some(gated_delta_kernel) => apply_qwen3_5_gated_delta_sequence_kernel(
            runtime,
            gated_delta_kernel,
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            sequence_shape,
        ),
        None => ops_gated_delta_sequence_loop(
            runtime,
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            sequence_shape.token_count,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_qwen3_5_gated_delta_sequence_kernel(
    runtime: &MlxRuntime,
    gated_delta_kernel: &MlxMetalKernel,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    sequence_shape: GatedDeltaSequenceShape,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let token_count = runtime.array_from_i32(&[sequence_shape.token_count], &[])?;
    let outputs = runtime.apply_metal_kernel(
        gated_delta_kernel,
        &[
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            &token_count,
        ],
        &[
            MlxMetalKernelOutput::new(
                vec![
                    sequence_shape.batch_size,
                    sequence_shape.token_count,
                    sequence_shape.value_head_count,
                    sequence_shape.value_head_dimension,
                ],
                queries.dtype(),
            ),
            MlxMetalKernelOutput::new(recurrent_state.shape(), recurrent_state.dtype()),
        ],
        [
            THREADGROUP_THREAD_COUNT as i32
                * (sequence_shape.value_head_dimension / VALUE_ROW_BLOCK_SIZE as i32),
            sequence_shape.value_head_count,
            sequence_shape.batch_size,
        ],
        [THREADGROUP_THREAD_COUNT as i32, 1, 1],
        &template_arguments(sequence_shape, queries.dtype(), recurrent_state.dtype()),
    )?;
    let mut output_iterator = outputs.into_iter();
    let sequence_outputs = output_iterator.next().ok_or_else(|| {
        gated_delta_sequence_error("fused gated-delta kernel did not return sequence outputs")
    })?;
    let next_recurrent_state = output_iterator.next().ok_or_else(|| {
        gated_delta_sequence_error("fused gated-delta kernel did not return recurrent state")
    })?;
    Ok((sequence_outputs, next_recurrent_state))
}

/// Applies the gated-delta recurrence through repeated one-token public MLX
/// ops — the documented fallback for a GPU whose capability probe demoted the
/// fused kernel. Arithmetic matches ordinary decode because it uses the same
/// public `qwen3_5_gated_delta_step` reference.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence_ops_fallback(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    ops_gated_delta_sequence_loop(
        runtime,
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
        sequence_shape.token_count,
    )
}

/// Runs the one-token ops reference across a sequence, stacking per-token
/// outputs and carrying the float32 recurrent state forward.
#[allow(clippy::too_many_arguments)]
pub(super) fn ops_gated_delta_sequence_loop(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    token_count: i32,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let query_shape = queries.shape();
    let (key_head_count, key_head_dimension) = (query_shape[2], query_shape[3]);
    let value_head_count = values.shape()[2];
    let value_head_dimension = values.shape()[3];
    let mut token_outputs = Vec::with_capacity(token_count as usize);
    // MLX arrays are immutable, so carrying the state as an owned handle per
    // step is safe; the initial float32 cast is an aliasing no-op for an
    // already-float32 state.
    let mut current_recurrent_state =
        runtime.astype(recurrent_state, astronomical_mlx_c_rust::MlxDtype::Float32)?;
    for token_index in 0..token_count {
        let token_queries = slice_rank_four_token(
            runtime,
            queries,
            token_index,
            key_head_count,
            key_head_dimension,
        )?;
        let token_keys = slice_rank_four_token(
            runtime,
            keys,
            token_index,
            key_head_count,
            key_head_dimension,
        )?;
        let token_values = slice_rank_four_token(
            runtime,
            values,
            token_index,
            value_head_count,
            value_head_dimension,
        )?;
        let token_decays = slice_rank_three_token(runtime, decays, token_index, value_head_count)?;
        let token_update_rates =
            slice_rank_three_token(runtime, update_rates, token_index, value_head_count)?;
        let (token_output, next_recurrent_state) = super::gated_delta::qwen3_5_gated_delta_step(
            runtime,
            &token_queries,
            &token_keys,
            &token_values,
            &token_decays,
            &token_update_rates,
            &current_recurrent_state,
        )?;
        token_outputs.push(token_output);
        current_recurrent_state = next_recurrent_state;
    }
    let token_output_references = token_outputs.iter().collect::<Vec<_>>();
    let sequence_outputs = runtime.stack_axis(&token_output_references, 1)?;
    Ok((sequence_outputs, current_recurrent_state))
}

fn slice_rank_four_token(
    runtime: &MlxRuntime,
    sequence_array: &MlxArray,
    token_index: i32,
    head_count: i32,
    head_dimension: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let sliced_token = runtime.slice(
        sequence_array,
        &[0, token_index, 0, 0],
        &[1, token_index + 1, head_count, head_dimension],
        &[1, 1, 1, 1],
    )?;
    runtime.squeeze_axis(&sliced_token, 1)
}

fn slice_rank_three_token(
    runtime: &MlxRuntime,
    sequence_array: &MlxArray,
    token_index: i32,
    head_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let sliced_token = runtime.slice(
        sequence_array,
        &[0, token_index, 0],
        &[1, token_index + 1, head_count],
        &[1, 1, 1],
    )?;
    runtime.squeeze_axis(&sliced_token, 1)
}
