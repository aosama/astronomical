//! Model-visible correction tokens extend the active folded-ternary context.

use astronomical_ipc_protocol::RequestId;

use super::engine_trait_impl::gpu_error;
use super::sampling::sample_token;
use super::{PrismHadamardQwen35InferenceExecution, successor_stays_pending};
use crate::{InferenceEngineError, PerformanceOperation};

pub(super) fn inject_input_tokens(
    execution: &mut PrismHadamardQwen35InferenceExecution,
    request_id: RequestId,
    input_token_ids: Vec<u32>,
) -> Result<(), InferenceEngineError> {
    if input_token_ids.is_empty() {
        return Ok(());
    }
    let token_count = validated_feedback_token_count(execution, request_id, &input_token_ids)?;
    let model = execution
        .model
        .as_ref()
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "folded-ternary model is not loaded".to_owned(),
        })?;
    let feedback_token_indices: Vec<i32> = input_token_ids
        .iter()
        .map(|token_id| *token_id as i32)
        .collect();
    let token_indices = model
        .runtime
        .array_from_i32(&feedback_token_indices, &[1, token_count])
        .map_err(gpu_error)?;
    run_feedback_forward_and_sample(execution, &token_indices)
}

/// Validates one feedback injection before any tensor work: the active
/// generation must belong to this request, every token must be inside the
/// vocabulary, and the projected context (processed context plus this
/// feedback plus the remaining output reserve) must fit the model's window.
/// Returns the feedback's token count in the model's i32 shape range.
fn validated_feedback_token_count(
    execution: &mut PrismHadamardQwen35InferenceExecution,
    request_id: RequestId,
    input_token_ids: &[u32],
) -> Result<i32, InferenceEngineError> {
    let maximum_context_tokens = execution.maximum_context_tokens;
    let model = execution
        .model
        .as_ref()
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "folded-ternary model is not loaded".to_owned(),
        })?;
    let active_generation =
        execution
            .active
            .as_mut()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary has no active generation".to_owned(),
            })?;
    if active_generation.request_id != request_id {
        return Err(InferenceEngineError::Fatal {
            reason: "folded-ternary feedback request correlation mismatch".to_owned(),
        });
    }
    let vocabulary_size = model.pack().config().text().vocab_size;
    if input_token_ids
        .iter()
        .any(|token_id| *token_id >= vocabulary_size)
    {
        return Err(InferenceEngineError::Fatal {
            reason: "folded-ternary model feedback contains a token outside the vocabulary"
                .to_owned(),
        });
    }
    let processed_context_token_count = usize::try_from(
        active_generation
            .generation_state
            .sequence_position_offset(),
    )
    .map_err(|_| InferenceEngineError::Fatal {
        reason: "folded-ternary context position is negative".to_owned(),
    })?;
    let projected_context_token_count = processed_context_token_count
        .checked_add(input_token_ids.len())
        .and_then(|context_token_count| {
            context_token_count.checked_add(active_generation.remaining_output_tokens as usize)
        })
        .ok_or_else(|| InferenceEngineError::InvalidRequest {
            reason: "folded-ternary feedback context token count overflowed".to_owned(),
        })?;
    if projected_context_token_count > maximum_context_tokens as usize {
        return Err(InferenceEngineError::InvalidRequest {
            reason: "folded-ternary model-visible feedback exceeds the context window".to_owned(),
        });
    }
    i32::try_from(input_token_ids.len()).map_err(|_| InferenceEngineError::InvalidRequest {
        reason: "folded-ternary feedback token count exceeds the model shape range".to_owned(),
    })
}

/// Runs the measured feedback forward and samples its next token: prefill
/// advance over the feedback tokens, final-logits construction, synchronous
/// evaluation of both graphs, decode sampling, the decode evaluation
/// submission, and the pending-token bookkeeping. Every stage stays inside
/// its own attribution span in the original order.
fn run_feedback_forward_and_sample(
    execution: &mut PrismHadamardQwen35InferenceExecution,
    token_indices: &mlx_c_rust::MlxArray,
) -> Result<(), InferenceEngineError> {
    let model = execution
        .model
        .as_ref()
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "folded-ternary model is not loaded".to_owned(),
        })?;
    let active_generation =
        execution
            .active
            .as_mut()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary has no active generation".to_owned(),
            })?;
    let hidden_states = active_generation
        .measure_operation(
            PerformanceOperation::PromptPrefillAdvanceSpan,
            |active_generation| {
                model.forward_hidden_states(token_indices, &mut active_generation.generation_state)
            },
        )
        .map_err(gpu_error)?;
    let final_logits = active_generation
        .measure_operation(PerformanceOperation::FinalLogitsGraphConstruction, |_| {
            model.logits_for_last_hidden_state(&hidden_states)
        })
        .map_err(gpu_error)?;
    active_generation
        .measure_operation(
            PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
            |_| {
                model
                    .runtime
                    .evaluate_arrays(&[&hidden_states, &final_logits])
            },
        )
        .map_err(gpu_error)?;
    let next_token = active_generation.measure_operation(
        PerformanceOperation::DecodeSamplingSpan,
        |active_generation| sample_token(model, &final_logits, active_generation),
    )?;
    active_generation
        .measure_operation(
            PerformanceOperation::DecodeAsyncEvaluationSubmission,
            |_| model.runtime.async_eval_arrays(&[&next_token]),
        )
        .map_err(gpu_error)?;
    active_generation.pending_sampled_token =
        successor_stays_pending(active_generation.remaining_output_tokens).then_some(next_token);
    Ok(())
}
