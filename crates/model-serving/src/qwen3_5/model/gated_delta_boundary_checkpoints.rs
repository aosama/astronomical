use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

use super::gated_delta_sequence::gated_delta_kernel_source;
use super::gated_delta_sequence_contract::{
    gated_delta_sequence_error, template_arguments, validate_gated_delta_sequence_shapes,
};

const GATED_DELTA_CHECKPOINT_SETUP_SOURCE: &str = r#"
    int checkpoint_index = 0;
    int next_checkpoint_token_count = first_checkpoint_token_count;
"#;
const GATED_DELTA_CHECKPOINT_WRITE_SOURCE: &str = r#"
            auto completed_token_count = first_token + token_in_block + 1;
            if (checkpoint_index < checkpoint_count &&
                completed_token_count == next_checkpoint_token_count) {
                auto checkpoint_state = (device float4*)(recurrent_boundary_states +
                    (((((size_t)checkpoint_index * B + batch_index) * Hv +
                       value_head_index) * Dv + first_value_row + value_row_in_block) *
                     Dk + first_key_dimension));
                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                    checkpoint_state[fragment_index] = state_fragment[fragment_index];
                }
                ++checkpoint_index;
                next_checkpoint_token_count += checkpoint_interval_token_count;
            }
"#;

/// Boundary checkpoint outputs from one fused gated-delta sequence.
pub struct Qwen3_5GatedDeltaBoundaryCheckpointResult {
    pub sequence_outputs: MlxArray,
    pub next_recurrent_state: MlxArray,
    pub recurrent_boundary_states: Vec<MlxArray>,
}

/// Builds the fused Qwen3.5 gated-delta boundary-checkpoint kernel.
pub fn qwen3_5_gated_delta_checkpoint_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    let checkpoint_kernel_source = gated_delta_kernel_source(
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
}

/// Applies fused gated-delta recurrence while retaining requested boundary states.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
    runtime: &MlxRuntime,
    gated_delta_checkpoint_kernel: &MlxMetalKernel,
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
    let checkpoint_count = i32::try_from(completed_prefill_chunk_tokens.len()).map_err(|_| {
        gated_delta_sequence_error("gated-delta checkpoint count exceeds the Int32 range")
    })?;
    let token_count = runtime.array_from_i32(&[sequence_shape.token_count], &[])?;
    let first_checkpoint_token_count =
        runtime.array_from_i32(&[completed_prefill_chunk_tokens[0]], &[])?;
    let checkpoint_interval_token_count =
        runtime.array_from_i32(&[checkpoint_interval_token_count], &[])?;
    let checkpoint_count_input = runtime.array_from_i32(&[checkpoint_count], &[])?;
    let mut checkpoint_template_arguments = Vec::from(template_arguments(
        sequence_shape,
        queries.dtype(),
        recurrent_state.dtype(),
    ));
    checkpoint_template_arguments.push(MlxMetalKernelTemplateArgument::Integer {
        name: "B",
        integer_template_argument: sequence_shape.batch_size,
    });
    let outputs = runtime.apply_metal_kernel(
        gated_delta_checkpoint_kernel,
        &[
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            &token_count,
            &first_checkpoint_token_count,
            &checkpoint_interval_token_count,
            &checkpoint_count_input,
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
            MlxMetalKernelOutput::new(
                vec![
                    checkpoint_count,
                    sequence_shape.batch_size,
                    sequence_shape.value_head_count,
                    sequence_shape.value_head_dimension,
                    sequence_shape.key_head_dimension,
                ],
                MlxDtype::Float32,
            ),
        ],
        [
            256 * (sequence_shape.value_head_dimension / 32),
            sequence_shape.value_head_count,
            sequence_shape.batch_size,
        ],
        [256, 1, 1],
        &checkpoint_template_arguments,
    )?;
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
    let recurrent_state_shape = recurrent_state.shape();
    let mut recurrent_boundary_states = Vec::with_capacity(completed_prefill_chunk_tokens.len());
    for checkpoint_index in 0..checkpoint_count {
        let recurrent_boundary_state_with_checkpoint_axis = runtime.slice(
            &packed_recurrent_boundary_states,
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
            &recurrent_state_shape,
        )?);
    }
    Ok(Qwen3_5GatedDeltaBoundaryCheckpointResult {
        sequence_outputs,
        next_recurrent_state,
        recurrent_boundary_states,
    })
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
