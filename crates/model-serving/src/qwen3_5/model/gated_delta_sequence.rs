use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

const GATED_DELTA_SEQUENCE_OPERATION: &str = "apply fused Qwen3.5 gated-delta sequence";
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

/// Applies fused Qwen3.5 gated-delta recurrence across one prompt/decode sequence.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence(
    runtime: &MlxRuntime,
    gated_delta_kernel: &MlxMetalKernel,
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

pub(super) fn template_arguments(
    sequence_shape: GatedDeltaSequenceShape,
    input_dtype: MlxDtype,
    state_dtype: MlxDtype,
) -> [MlxMetalKernelTemplateArgument; 6] {
    [
        MlxMetalKernelTemplateArgument::Dtype {
            name: "InT",
            dtype: input_dtype,
        },
        MlxMetalKernelTemplateArgument::Dtype {
            name: "StT",
            dtype: state_dtype,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Dk",
            integer_template_argument: sequence_shape.key_head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Dv",
            integer_template_argument: sequence_shape.value_head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Hk",
            integer_template_argument: sequence_shape.key_head_count,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Hv",
            integer_template_argument: sequence_shape.value_head_count,
        },
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GatedDeltaSequenceShape {
    pub(super) batch_size: i32,
    pub(super) token_count: i32,
    pub(super) key_head_count: i32,
    pub(super) value_head_count: i32,
    pub(super) key_head_dimension: i32,
    pub(super) value_head_dimension: i32,
}

pub(super) fn validate_gated_delta_sequence_shapes(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<GatedDeltaSequenceShape, MlxRuntimeError> {
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let decay_shape = decays.shape();
    let update_rate_shape = update_rates.shape();
    let recurrent_state_shape = recurrent_state.shape();
    if query_shape.len() != 4
        || key_shape.len() != 4
        || value_shape.len() != 4
        || decay_shape.len() != 3
        || update_rate_shape.len() != 3
        || recurrent_state_shape.len() != 4
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence inputs have invalid ranks",
        ));
    }
    if query_shape != key_shape {
        return Err(gated_delta_sequence_error(
            "fused gated-delta queries and keys must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let token_count = query_shape[1];
    let key_head_count = query_shape[2];
    let key_head_dimension = query_shape[3];
    let value_head_count = value_shape[2];
    let value_head_dimension = value_shape[3];
    if batch_size <= 0
        || token_count <= 0
        || key_head_count <= 0
        || value_head_count <= 0
        || key_head_dimension <= 0
        || value_head_dimension <= 0
        || value_head_count % key_head_count != 0
        || key_head_dimension != 128
        || value_head_dimension % 32 != 0
    {
        return Err(gated_delta_sequence_error(
            "blocked gated-delta dimensions must be positive, value heads must be a multiple of key heads, key dimension must be 128, and value dimension must divide by 32",
        ));
    }
    if value_shape[0] != batch_size
        || value_shape[1] != token_count
        || decay_shape != [batch_size, token_count, value_head_count]
        || update_rate_shape != decay_shape
        || recurrent_state_shape
            != [
                batch_size,
                value_head_count,
                value_head_dimension,
                key_head_dimension,
            ]
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence shapes are incompatible",
        ));
    }
    if recurrent_state.dtype() != MlxDtype::Float32 {
        return Err(gated_delta_sequence_error(
            "fused gated-delta recurrent state must use float32",
        ));
    }
    if !is_supported_activation_dtype(queries.dtype())
        || !is_supported_activation_dtype(keys.dtype())
        || !is_supported_activation_dtype(values.dtype())
        || !is_supported_activation_dtype(decays.dtype())
        || !is_supported_activation_dtype(update_rates.dtype())
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta inputs must use float16, bfloat16, or float32",
        ));
    }
    Ok(GatedDeltaSequenceShape {
        batch_size,
        token_count,
        key_head_count,
        value_head_count,
        key_head_dimension,
        value_head_dimension,
    })
}

fn is_supported_activation_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}

pub(super) fn gated_delta_sequence_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: GATED_DELTA_SEQUENCE_OPERATION,
        description: description.to_owned(),
    }
}
