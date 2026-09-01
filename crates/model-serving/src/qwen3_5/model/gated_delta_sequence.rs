use astronomical_runtime_integration::{
    MlxArray, MlxMetalKernel, MlxMetalKernelOutput, MlxRuntime, MlxRuntimeError,
};

use super::gated_delta_sequence_contract::{
    GatedDeltaSequenceShape, gated_delta_sequence_error, template_arguments,
    validate_gated_delta_sequence_shapes,
};
const CHECKPOINT_SETUP_MARKER: &str = "/* ASTRONOMICAL_CHECKPOINT_SETUP */";
const CHECKPOINT_WRITE_MARKER: &str = "/* ASTRONOMICAL_CHECKPOINT_WRITE */";
const GATED_DELTA_KERNEL_SOURCE: &str = r#"
    constexpr int time_block_size = sizeof(InT) == 4 ? 16 : 32;
    constexpr int value_row_block_size = 32;
    auto thread_index = thread_position_in_threadgroup.x;
    auto value_row_block = threadgroup_position_in_grid.x;
    auto value_head_index = threadgroup_position_in_grid.y;
    auto batch_index = threadgroup_position_in_grid.z;
    auto key_head_index = value_head_index / (Hv / Hk);
    auto first_value_row = value_row_block * value_row_block_size;

    auto value_row_in_block = thread_index / 8;
    auto key_segment_index = thread_index % 8;
    auto first_key_dimension = key_segment_index * 16;

    threadgroup InT staged_keys[time_block_size][Dk + 8];
    threadgroup InT staged_queries[time_block_size][Dk + 8];
    threadgroup InT staged_values[time_block_size][value_row_block_size + 8];
    threadgroup float staged_decays[time_block_size];
    threadgroup float staged_update_rates[time_block_size];

    auto key_base = keys +
        ((size_t)batch_index * token_count * Hk + key_head_index) * Dk;
    auto query_base = queries +
        ((size_t)batch_index * token_count * Hk + key_head_index) * Dk;
    auto value_base = values +
        ((size_t)batch_index * token_count * Hv + value_head_index) * Dv +
        first_value_row;
    auto key_row_stride = (size_t)Hk * Dk;

    float4 state_fragment[4];
    {
        auto input_state = (const device float4*)(recurrent_state +
            (((size_t)batch_index * Hv + value_head_index) * Dv +
             first_value_row + value_row_in_block) * Dk + first_key_dimension);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] = input_state[fragment_index];
        }
    }

    auto output_base = outputs +
        ((size_t)batch_index * token_count * Hv + value_head_index) * Dv +
        first_value_row;

    /* ASTRONOMICAL_CHECKPOINT_SETUP */
    for (int first_token = 0; first_token < token_count;
         first_token += time_block_size) {
        auto tokens_in_block = min(time_block_size, token_count - first_token);
        for (int staged_index = thread_index;
             staged_index < tokens_in_block * Dk;
             staged_index += 256) {
            auto token_in_block = staged_index / Dk;
            auto key_dimension = staged_index % Dk;
            staged_keys[token_in_block][key_dimension] =
                key_base[(size_t)(first_token + token_in_block) * key_row_stride +
                         key_dimension];
            staged_queries[token_in_block][key_dimension] =
                query_base[(size_t)(first_token + token_in_block) * key_row_stride +
                           key_dimension];
        }
        for (int staged_index = thread_index;
             staged_index < tokens_in_block * value_row_block_size;
             staged_index += 256) {
            auto token_in_block = staged_index / value_row_block_size;
            auto value_row = staged_index % value_row_block_size;
            staged_values[token_in_block][value_row] =
                value_base[(size_t)(first_token + token_in_block) * Hv * Dv +
                           value_row];
        }
        for (int token_in_block = thread_index;
             token_in_block < tokens_in_block;
             token_in_block += 256) {
            staged_decays[token_in_block] =
                decays[((size_t)batch_index * token_count + first_token +
                        token_in_block) * Hv + value_head_index];
            staged_update_rates[token_in_block] =
                update_rates[((size_t)batch_index * token_count + first_token +
                              token_in_block) * Hv + value_head_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int token_in_block = 0; token_in_block < tokens_in_block;
             ++token_in_block) {
            auto key_vectors = (const threadgroup vec<InT, 4>*)
                &staged_keys[token_in_block][first_key_dimension];
            auto query_vectors = (const threadgroup vec<InT, 4>*)
                &staged_queries[token_in_block][first_key_dimension];
            float4 float_keys[4];
            for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                float_keys[fragment_index] = float4(key_vectors[fragment_index]);
            }

            float4 remembered_fragments = 0.0f;
            for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                state_fragment[fragment_index] *= staged_decays[token_in_block];
                remembered_fragments +=
                    state_fragment[fragment_index] * float_keys[fragment_index];
            }
            float remembered_value = remembered_fragments.x + remembered_fragments.y +
                remembered_fragments.z + remembered_fragments.w;
            remembered_value += simd_shuffle_down(remembered_value, 4);
            remembered_value += simd_shuffle_down(remembered_value, 2);
            remembered_value += simd_shuffle_down(remembered_value, 1);
            remembered_value = simd_shuffle(
                remembered_value,
                (thread_index % 32) / 8 * 8);
            auto delta =
                ((float)staged_values[token_in_block][value_row_in_block] -
                 remembered_value) * staged_update_rates[token_in_block];

            float4 output_fragments = 0.0f;
            for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                state_fragment[fragment_index] += float_keys[fragment_index] * delta;
                output_fragments += state_fragment[fragment_index] *
                    float4(query_vectors[fragment_index]);
            }
            /* ASTRONOMICAL_CHECKPOINT_WRITE */
            float output_value = output_fragments.x + output_fragments.y +
                output_fragments.z + output_fragments.w;
            output_value += simd_shuffle_down(output_value, 4);
            output_value += simd_shuffle_down(output_value, 2);
            output_value += simd_shuffle_down(output_value, 1);
            if (key_segment_index == 0) {
                output_base[(size_t)(first_token + token_in_block) * Hv * Dv +
                            value_row_in_block] = (InT)output_value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        auto output_state = (device float4*)(next_recurrent_state +
            (((size_t)batch_index * Hv + value_head_index) * Dv +
             first_value_row + value_row_in_block) * Dk + first_key_dimension);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            output_state[fragment_index] = state_fragment[fragment_index];
        }
    }
"#;

/// Builds the fused Qwen3.5 gated-delta sequence kernel.
pub fn qwen3_5_gated_delta_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    let ordinary_kernel_source = gated_delta_kernel_source("", "");
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

pub(super) fn gated_delta_kernel_source(
    checkpoint_setup_source: &str,
    checkpoint_write_source: &str,
) -> String {
    GATED_DELTA_KERNEL_SOURCE
        .replace(CHECKPOINT_SETUP_MARKER, checkpoint_setup_source)
        .replace(CHECKPOINT_WRITE_MARKER, checkpoint_write_source)
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
            256 * (sequence_shape.value_head_dimension / 32),
            sequence_shape.value_head_count,
            sequence_shape.batch_size,
        ],
        [256, 1, 1],
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
    let mut current_recurrent_state = runtime.astype(
        recurrent_state,
        astronomical_runtime_integration::MlxDtype::Float32,
    )?;
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
