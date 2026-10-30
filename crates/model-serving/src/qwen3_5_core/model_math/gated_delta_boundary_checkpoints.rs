use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::gated_delta_pipelined_kernel::{
    THREADGROUP_THREAD_COUNT, VALUE_ROW_BLOCK_SIZE, gated_delta_pipelined_kernel_source,
};
use super::gated_delta_sequence_contract::{
    GatedDeltaSequenceShape, gated_delta_sequence_error, template_arguments,
    validate_gated_delta_sequence_shapes,
};
use mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

const GATED_DELTA_CHECKPOINT_SETUP_SOURCE: &str = r#"
    int checkpoint_index = 0;
    int next_checkpoint_token_count = first_checkpoint_token_count;
"#;
const GATED_DELTA_CHECKPOINT_WRITE_SOURCE: &str = r#"
                const int completed_token_count = first_token +
                    /* ASTRONOMICAL_CHECKPOINT_TOKEN */ + 1;
                if (checkpoint_index < checkpoint_count &&
                    completed_token_count == next_checkpoint_token_count) {
                    auto checkpoint_state = (device float4*)(
                        recurrent_boundary_states +
                        (((((size_t)checkpoint_index * B + batch_index) * Hv +
                           value_head_index) * Dv + first_value_row +
                           value_row_in_block) * Dk));
                    for (int fragment_index = 0; fragment_index < 4;
                         ++fragment_index) {
                        checkpoint_state[key_segment_index +
                            8 * fragment_index] =
                            state_fragment[fragment_index];
                    }
                    ++checkpoint_index;
                    next_checkpoint_token_count +=
                        checkpoint_interval_token_count;
                }
"#;

/// Boundary checkpoint outputs from one fused gated-delta sequence.
pub struct Qwen3_5GatedDeltaBoundaryCheckpointResult {
    pub sequence_outputs: MlxArray,
    pub next_recurrent_state: MlxArray,
    pub recurrent_boundary_states: Vec<MlxArray>,
}

/// Builds the fused Qwen3.5 gated-delta boundary-checkpoint kernel.
pub fn qwen3_5_gated_delta_checkpoint_kernel()
-> Result<MlxMetalKernel, astronomical_runtime_integration::MlxRuntimeError> {
    let checkpoint_kernel_source = gated_delta_pipelined_kernel_source(
        GATED_DELTA_CHECKPOINT_SETUP_SOURCE,
        GATED_DELTA_CHECKPOINT_WRITE_SOURCE,
    );
    MlxMetalKernel::new(
        "astronomical_qwen3_5_gated_delta_sequence_with_boundary_checkpoints",
        &[
            "queries",
            "keys",
            "values",
            "decays",
            "update_rates",
            "recurrent_state",
            "token_count",
            "first_checkpoint_token_count",
            "checkpoint_interval_token_count",
            "checkpoint_count",
        ],
        &[
            "outputs",
            "next_recurrent_state",
            "recurrent_boundary_states",
        ],
        &checkpoint_kernel_source,
    )
    .map_err(astronomical_runtime_integration::MlxRuntimeError::from)
}

/// Applies fused gated-delta recurrence while retaining requested boundary
/// states. A retained kernel uses the fused Metal route; a capability-demoted
/// kernel — `None` — uses the ops-based fallback, which snapshots boundaries
/// at exactly the same completed-token positions.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
    runtime: &MlxRuntime,
    gated_delta_checkpoint_kernel: Option<&MlxMetalKernel>,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    completed_prefill_chunk_tokens: &[i32],
    checkpoint_interval_token_count: i32,
) -> Result<Qwen3_5GatedDeltaBoundaryCheckpointResult, MlxRuntimeError> {
    let Some(gated_delta_checkpoint_kernel) = gated_delta_checkpoint_kernel else {
        return qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback(
            runtime,
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            completed_prefill_chunk_tokens,
            checkpoint_interval_token_count,
        );
    };
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    validate_boundary_checkpoint_positions(
        completed_prefill_chunk_tokens,
        checkpoint_interval_token_count,
        sequence_shape.token_count,
    )?;
    let checkpoint_count = i32::try_from(completed_prefill_chunk_tokens.len()).map_err(|_| {
        gated_delta_sequence_error("gated-delta checkpoint count exceeds the Int32 range")
    })?;
    let checkpoint_plan = CheckpointSequencePlan {
        sequence_shape,
        checkpoint_count,
    };
    let scalar_inputs = checkpoint_scalar_kernel_inputs(
        runtime,
        &checkpoint_plan,
        completed_prefill_chunk_tokens[0],
        checkpoint_interval_token_count,
    )?;
    let checkpoint_template_arguments =
        checkpoint_template_arguments(sequence_shape, queries.dtype(), recurrent_state.dtype());
    let dispatch = CheckpointKernelDispatch {
        sequence_buffers: CheckpointSequenceBuffers {
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
        },
        scalar_inputs,
        checkpoint_plan,
    };
    let outputs = apply_gated_delta_checkpoint_kernel(
        runtime,
        gated_delta_checkpoint_kernel,
        &dispatch,
        &checkpoint_template_arguments,
    )?;
    let kernel_outputs = unpack_checkpoint_kernel_outputs(outputs)?;
    let recurrent_state_shape = recurrent_state.shape();
    let recurrent_boundary_states = sliced_recurrent_boundary_states(
        runtime,
        &kernel_outputs.packed_recurrent_boundary_states,
        &dispatch.checkpoint_plan,
        &recurrent_state_shape,
    )?;
    Ok(Qwen3_5GatedDeltaBoundaryCheckpointResult {
        sequence_outputs: kernel_outputs.sequence_outputs,
        next_recurrent_state: kernel_outputs.next_recurrent_state,
        recurrent_boundary_states,
    })
}

/// Validated sequence geometry plus the checkpoint count, grouped so the
/// dispatch and its output unpacking observe identical immutable facts.
struct CheckpointSequencePlan {
    sequence_shape: GatedDeltaSequenceShape,
    checkpoint_count: i32,
}

/// The six sequence tensors the fused checkpoint kernel reads, in launch order.
struct CheckpointSequenceBuffers<'a> {
    queries: &'a MlxArray,
    keys: &'a MlxArray,
    values: &'a MlxArray,
    decays: &'a MlxArray,
    update_rates: &'a MlxArray,
    recurrent_state: &'a MlxArray,
}

/// Scalar control arrays built once per forward and consumed by the launch.
struct CheckpointScalarKernelInputs {
    token_count: MlxArray,
    first_checkpoint_token_count: MlxArray,
    checkpoint_interval_token_count: MlxArray,
    checkpoint_count_input: MlxArray,
}

/// Everything one fused checkpoint kernel launch consumes.
struct CheckpointKernelDispatch<'a> {
    sequence_buffers: CheckpointSequenceBuffers<'a>,
    scalar_inputs: CheckpointScalarKernelInputs,
    checkpoint_plan: CheckpointSequencePlan,
}

/// The three arrays one fused checkpoint kernel launch returns, in the order
/// the kernel declares them.
#[derive(Debug)]
pub(super) struct CheckpointKernelOutputs {
    pub(super) sequence_outputs: MlxArray,
    pub(super) next_recurrent_state: MlxArray,
    pub(super) packed_recurrent_boundary_states: MlxArray,
}

fn checkpoint_scalar_kernel_inputs(
    runtime: &MlxRuntime,
    checkpoint_plan: &CheckpointSequencePlan,
    first_checkpoint_token_count: i32,
    checkpoint_interval_token_count: i32,
) -> Result<CheckpointScalarKernelInputs, MlxRuntimeError> {
    Ok(CheckpointScalarKernelInputs {
        token_count: runtime.array_from_i32(&[checkpoint_plan.sequence_shape.token_count], &[])?,
        first_checkpoint_token_count: runtime
            .array_from_i32(&[first_checkpoint_token_count], &[])?,
        checkpoint_interval_token_count: runtime
            .array_from_i32(&[checkpoint_interval_token_count], &[])?,
        checkpoint_count_input: runtime.array_from_i32(&[checkpoint_plan.checkpoint_count], &[])?,
    })
}

/// Extends the shape-invariant sequence template arguments with the batch
/// size `B`: the checkpoint-write source indexes boundary states through it,
/// so it is baked per launch while token count stays a runtime scalar.
pub(super) fn checkpoint_template_arguments(
    sequence_shape: GatedDeltaSequenceShape,
    input_dtype: MlxDtype,
    state_dtype: MlxDtype,
) -> Vec<MlxMetalKernelTemplateArgument> {
    let mut checkpoint_template_arguments =
        Vec::from(template_arguments(sequence_shape, input_dtype, state_dtype));
    checkpoint_template_arguments.push(MlxMetalKernelTemplateArgument::Integer {
        name: "B",
        integer_template_argument: sequence_shape.batch_size,
    });
    checkpoint_template_arguments
}

/// Submits one fused checkpoint kernel launch with its buffers, scalar
/// controls, output shapes, and grid geometry, exactly as declared.
fn apply_gated_delta_checkpoint_kernel(
    runtime: &MlxRuntime,
    gated_delta_checkpoint_kernel: &MlxMetalKernel,
    dispatch: &CheckpointKernelDispatch<'_>,
    checkpoint_template_arguments: &[MlxMetalKernelTemplateArgument],
) -> Result<Vec<MlxArray>, MlxRuntimeError> {
    let sequence_shape = dispatch.checkpoint_plan.sequence_shape;
    let scalar_inputs = &dispatch.scalar_inputs;
    let outputs = runtime.apply_metal_kernel(
        gated_delta_checkpoint_kernel,
        &[
            dispatch.sequence_buffers.queries,
            dispatch.sequence_buffers.keys,
            dispatch.sequence_buffers.values,
            dispatch.sequence_buffers.decays,
            dispatch.sequence_buffers.update_rates,
            dispatch.sequence_buffers.recurrent_state,
            &scalar_inputs.token_count,
            &scalar_inputs.first_checkpoint_token_count,
            &scalar_inputs.checkpoint_interval_token_count,
            &scalar_inputs.checkpoint_count_input,
        ],
        &[
            MlxMetalKernelOutput::new(
                vec![
                    sequence_shape.batch_size,
                    sequence_shape.token_count,
                    sequence_shape.value_head_count,
                    sequence_shape.value_head_dimension,
                ],
                dispatch.sequence_buffers.queries.dtype(),
            ),
            MlxMetalKernelOutput::new(
                dispatch.sequence_buffers.recurrent_state.shape(),
                dispatch.sequence_buffers.recurrent_state.dtype(),
            ),
            MlxMetalKernelOutput::new(
                vec![
                    dispatch.checkpoint_plan.checkpoint_count,
                    sequence_shape.batch_size,
                    sequence_shape.value_head_count,
                    sequence_shape.value_head_dimension,
                    sequence_shape.key_head_dimension,
                ],
                MlxDtype::Float32,
            ),
        ],
        [
            THREADGROUP_THREAD_COUNT as i32
                * (sequence_shape.value_head_dimension / VALUE_ROW_BLOCK_SIZE as i32),
            sequence_shape.value_head_count,
            sequence_shape.batch_size,
        ],
        [THREADGROUP_THREAD_COUNT as i32, 1, 1],
        checkpoint_template_arguments,
    )?;
    Ok(outputs)
}

/// Unpacks the launch results in kernel-declared order, rejecting short
/// returns before any boundary slicing can index a missing array.
pub(super) fn unpack_checkpoint_kernel_outputs(
    outputs: Vec<MlxArray>,
) -> Result<CheckpointKernelOutputs, MlxRuntimeError> {
    let mut output_iterator = outputs.into_iter();
    let sequence_outputs = output_iterator.next().ok_or_else(|| {
        gated_delta_sequence_error("checkpoint kernel did not return sequence outputs")
    })?;
    let next_recurrent_state = output_iterator.next().ok_or_else(|| {
        gated_delta_sequence_error("checkpoint kernel did not return final recurrent state")
    })?;
    let packed_recurrent_boundary_states = output_iterator.next().ok_or_else(|| {
        gated_delta_sequence_error("checkpoint kernel did not return boundary recurrent states")
    })?;
    Ok(CheckpointKernelOutputs {
        sequence_outputs,
        next_recurrent_state,
        packed_recurrent_boundary_states,
    })
}

/// Slices each checkpoint axis entry out of the packed boundary tensor and
/// reshapes it back to the recurrent-state layout the prompt cache stores.
fn sliced_recurrent_boundary_states(
    runtime: &MlxRuntime,
    packed_recurrent_boundary_states: &MlxArray,
    checkpoint_plan: &CheckpointSequencePlan,
    recurrent_state_shape: &[i32],
) -> Result<Vec<MlxArray>, MlxRuntimeError> {
    let sequence_shape = checkpoint_plan.sequence_shape;
    let mut recurrent_boundary_states =
        Vec::with_capacity(usize::try_from(checkpoint_plan.checkpoint_count).unwrap_or(0));
    for checkpoint_index in 0..checkpoint_plan.checkpoint_count {
        let recurrent_boundary_state_with_checkpoint_axis = runtime.slice(
            packed_recurrent_boundary_states,
            &[checkpoint_index, 0, 0, 0, 0],
            &[
                checkpoint_index + 1,
                sequence_shape.batch_size,
                sequence_shape.value_head_count,
                sequence_shape.value_head_dimension,
                sequence_shape.key_head_dimension,
            ],
            &[1, 1, 1, 1, 1],
        )?;
        recurrent_boundary_states.push(runtime.reshape(
            &recurrent_boundary_state_with_checkpoint_axis,
            recurrent_state_shape,
        )?);
    }
    Ok(recurrent_boundary_states)
}

fn validate_boundary_checkpoint_positions(
    completed_prefill_chunk_tokens: &[i32],
    checkpoint_interval_token_count: i32,
    token_count: i32,
) -> Result<(), MlxRuntimeError> {
    if completed_prefill_chunk_tokens.is_empty() {
        return Err(gated_delta_sequence_error(
            "gated-delta boundary checkpoint positions must not be empty",
        ));
    }
    if checkpoint_interval_token_count <= 0 {
        return Err(gated_delta_sequence_error(
            "gated-delta checkpoint interval must be positive",
        ));
    }
    let mut previous_completed_prefill_chunk_tokens = 0;
    for current_completed_prefill_chunk_tokens in completed_prefill_chunk_tokens {
        if *current_completed_prefill_chunk_tokens <= previous_completed_prefill_chunk_tokens
            || *current_completed_prefill_chunk_tokens >= token_count
        {
            return Err(gated_delta_sequence_error(
                "gated-delta boundary checkpoints must be positive, strictly increasing, and less than token_count",
            ));
        }
        if previous_completed_prefill_chunk_tokens > 0
            && *current_completed_prefill_chunk_tokens - previous_completed_prefill_chunk_tokens
                != checkpoint_interval_token_count
        {
            return Err(gated_delta_sequence_error(
                "gated-delta boundary checkpoint spacing must match the supplied interval",
            ));
        }
        previous_completed_prefill_chunk_tokens = *current_completed_prefill_chunk_tokens;
    }
    Ok(())
}

/// Applies the gated-delta recurrence with boundary snapshots through repeated
/// one-token public MLX ops — the documented fallback for a GPU whose
/// capability probe demoted the fused checkpoint kernel. Boundary states are
/// captured at exactly the same completed-token positions the fused kernel
/// writes, so the persistent prompt cache boundary contract is preserved.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    completed_prefill_chunk_tokens: &[i32],
    checkpoint_interval_token_count: i32,
) -> Result<Qwen3_5GatedDeltaBoundaryCheckpointResult, MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    validate_boundary_checkpoint_positions(
        completed_prefill_chunk_tokens,
        checkpoint_interval_token_count,
        sequence_shape.token_count,
    )?;
    let query_shape = queries.shape();
    let (key_head_count, key_head_dimension) = (query_shape[2], query_shape[3]);
    let value_head_count = values.shape()[2];
    let value_head_dimension = values.shape()[3];
    let mut token_outputs = Vec::with_capacity(sequence_shape.token_count as usize);
    let mut recurrent_boundary_states = Vec::with_capacity(completed_prefill_chunk_tokens.len());
    let mut next_checkpoint_position_index = 0_usize;
    let mut current_recurrent_state =
        runtime.astype(recurrent_state, mlx_c_rust::MlxDtype::Float32)?;
    for token_index in 0..sequence_shape.token_count {
        let token_queries = runtime
            .slice(
                queries,
                &[0, token_index, 0, 0],
                &[1, token_index + 1, key_head_count, key_head_dimension],
                &[1, 1, 1, 1],
            )
            .and_then(|sliced| runtime.squeeze_axis(&sliced, 1))?;
        let token_keys = runtime
            .slice(
                keys,
                &[0, token_index, 0, 0],
                &[1, token_index + 1, key_head_count, key_head_dimension],
                &[1, 1, 1, 1],
            )
            .and_then(|sliced| runtime.squeeze_axis(&sliced, 1))?;
        let token_values = runtime
            .slice(
                values,
                &[0, token_index, 0, 0],
                &[1, token_index + 1, value_head_count, value_head_dimension],
                &[1, 1, 1, 1],
            )
            .and_then(|sliced| runtime.squeeze_axis(&sliced, 1))?;
        let token_decays = runtime
            .slice(
                decays,
                &[0, token_index, 0],
                &[1, token_index + 1, value_head_count],
                &[1, 1, 1],
            )
            .and_then(|sliced| runtime.squeeze_axis(&sliced, 1))?;
        let token_update_rates = runtime
            .slice(
                update_rates,
                &[0, token_index, 0],
                &[1, token_index + 1, value_head_count],
                &[1, 1, 1],
            )
            .and_then(|sliced| runtime.squeeze_axis(&sliced, 1))?;
        let (token_output, next_recurrent_state) =
            super::gated_delta_step::qwen3_5_gated_delta_step(
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
        let completed_token_count = token_index + 1;
        if next_checkpoint_position_index < completed_prefill_chunk_tokens.len()
            && completed_prefill_chunk_tokens[next_checkpoint_position_index]
                == completed_token_count
        {
            // The fused kernel publishes float32 boundary states; the aliasing
            // float32 cast yields an owned handle without a data copy.
            recurrent_boundary_states
                .push(runtime.astype(&current_recurrent_state, mlx_c_rust::MlxDtype::Float32)?);
            next_checkpoint_position_index += 1;
        }
    }
    let token_output_references = token_outputs.iter().collect::<Vec<_>>();
    let sequence_outputs = runtime.stack_axis(&token_output_references, 1)?;
    Ok(Qwen3_5GatedDeltaBoundaryCheckpointResult {
        sequence_outputs,
        next_recurrent_state: current_recurrent_state,
        recurrent_boundary_states,
    })
}
