use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::{InferenceEngineError, PerformanceCounter};

use super::target_verification::forward_target_verification_window_with_performance_attribution;
use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;
use crate::qwen3_5::multi_token_prediction::AcceptedMultiTokenPredictionDraftRollback;

const DEPTH_ONE_TARGET_VERIFY_TOKEN_COUNT: usize = 2;

/// Returns whether a depth-one prediction window could cross the forced thinking boundary.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_mtp_verification_may_cross_thinking_budget(
    is_inside_thinking: bool,
    thinking_token_count: u16,
    thinking_budget: Option<u16>,
    possible_emitted_token_count: u16,
) -> bool {
    is_inside_thinking
        && thinking_budget.is_some_and(|thinking_budget| {
            thinking_token_count.saturating_add(possible_emitted_token_count) >= thinking_budget
        })
}

/// Returns whether a depth-one prediction window fits output and context boundaries.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_depth_one_mtp_window_fits(
    generated_token_count: u16,
    maximum_output_tokens: u16,
    next_position_tokens: u32,
    maximum_position_count: usize,
) -> bool {
    maximum_output_tokens.saturating_sub(generated_token_count) >= 2
        && maximum_position_count.saturating_sub(next_position_tokens as usize) >= 2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::qwen3_5) enum Qwen3_5PredictionAcceptanceOutcome {
    Accepted,
    Rejected,
    OperationalFallback,
}

pub(in crate::qwen3_5) fn propose_depth_one_mtp_draft(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
) -> Option<u32> {
    let Some(mut multi_token_prediction_request) =
        active_request.take_optional_prediction_session()
    else {
        return None;
    };
    let Some(mtp_target_hidden_states) = multi_token_prediction_request.take_target_hidden_states()
    else {
        active_request.restore_optional_prediction_session(multi_token_prediction_request);
        return None;
    };

    let draft_token_id = match model.forward_mtp_draft_with_performance_attribution(
        &mtp_target_hidden_states,
        current_generated_token,
        multi_token_prediction_request.request_state_mut(),
        active_request.performance_attribution_mut(),
    ) {
        Ok((_mtp_forward_output, draft_token_id)) => {
            active_request.restore_optional_prediction_session(multi_token_prediction_request);
            draft_token_id
        }
        Err(mtp_forward_error) => {
            tracing::warn!(
                request_id = request_id.value(),
                error = %mtp_forward_error,
                "MTP draft forward failed; continuing this request with target-only decode"
            );
            return None;
        }
    };
    Some(draft_token_id)
}

pub(in crate::qwen3_5) fn projected_verification_window_memory_growth_bytes(
    model: &Qwen3_5Model,
    active_request: &Qwen3_5EngineRequest,
) -> Result<usize, InferenceEngineError> {
    let full_attention_bytes_per_layer_token = model
        .config()
        .full_attention_key_value_state_bytes_per_layer_token()
        .ok_or_else(|| {
            fatal_engine_error("prediction full-attention bytes per layer token overflowed")
        })?;
    active_request
        .optional_prediction_session()
        .ok_or_else(|| fatal_engine_error("active prediction request state disappeared"))?
        .projected_sequential_full_attention_growth_bytes(
            full_attention_bytes_per_layer_token,
            &[1, 1],
        )
        .map_err(qwen3_5_runtime_error)
}

pub(in crate::qwen3_5) fn verification_window_workspace_bytes(
    model: &Qwen3_5Model,
) -> Result<usize, InferenceEngineError> {
    model
        .decoder_cache_layout()
        .boundary_snapshot_payload_byte_count()
        .map_err(|decoder_cache_layout_error| {
            fatal_engine_error(format!(
                "failed to project target verification-window workspace: {decoder_cache_layout_error}"
            ))
        })
}

pub(in crate::qwen3_5) fn verify_depth_one_mtp_prefix_acceptance(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token_id: u32,
    draft_token_id: u32,
) -> Result<Qwen3_5PredictionAcceptanceOutcome, InferenceEngineError> {
    let target_state_checkpoint = active_request
        .request_decoder_state()
        .checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let target_verify_start_position_tokens = active_request.next_position_tokens();
    let (target_forward_output, target_verify_token_ids, verified_prefix_boundary_checkpoint) =
        match active_request.with_decoder_state_and_performance_attribution(
            |request_decoder_state, performance_attribution| {
                forward_target_verification_window_with_performance_attribution(
                    model,
                    &[current_generated_token_id, draft_token_id],
                    target_verify_start_position_tokens,
                    request_decoder_state,
                    performance_attribution,
                )
            },
        ) {
            Ok(target_verification_output) => target_verification_output,
            Err(target_verify_error) => {
                restore_target_state_after_mtp_failure(
                    active_request,
                    target_state_checkpoint,
                    target_verify_start_position_tokens,
                )?;
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %target_verify_error,
                    "MTP target verification failed; continuing this request with target-only decode"
                );
                active_request.clear_optional_prediction_session();
                return Ok(Qwen3_5PredictionAcceptanceOutcome::OperationalFallback);
            }
        };
    active_request.advance_position(DEPTH_ONE_TARGET_VERIFY_TOKEN_COUNT)?;

    if target_verify_token_ids.len() != DEPTH_ONE_TARGET_VERIFY_TOKEN_COUNT {
        restore_target_state_after_mtp_failure(
            active_request,
            target_state_checkpoint,
            target_verify_start_position_tokens,
        )?;
        tracing::warn!(
            request_id = request_id.value(),
            actual_token_count = target_verify_token_ids.len(),
            "MTP target verification returned an unexpected greedy-token count"
        );
        active_request.clear_optional_prediction_session();
        return Ok(Qwen3_5PredictionAcceptanceOutcome::OperationalFallback);
    }

    let force_mtp_draft_rejection = active_request
        .optional_prediction_session_mut()
        .is_some_and(|multi_token_prediction_request| {
            multi_token_prediction_request.take_forced_draft_rejection_for_tests()
        });
    let accepted_draft = !force_mtp_draft_rejection && target_verify_token_ids[0] == draft_token_id;
    let verified_prefix_position_tokens = target_verify_start_position_tokens
        .checked_add(1)
        .ok_or_else(|| fatal_engine_error("MTP verified-prefix position overflowed"))?;
    if accepted_draft {
        active_request
            .optional_prediction_session_mut()
            .ok_or_else(|| fatal_engine_error("MTP request session disappeared"))?
            .set_accepted_draft_rollback(AcceptedMultiTokenPredictionDraftRollback {
                verified_prefix_boundary_checkpoint,
                verified_prefix_position_tokens,
            });
        active_request
            .optional_prediction_session_mut()
            .ok_or_else(|| fatal_engine_error("MTP request session disappeared"))?
            .queue_verified_generated_token_id(draft_token_id);
        active_request.set_pending_generated_token(
            model
                .runtime()
                .array_from_u32(&[target_verify_token_ids[1]], &[1, 1])
                .map_err(qwen3_5_runtime_error)?,
        );
        let retained_hidden_state = match target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), 1)
        {
            Ok(hidden_state) => Some(hidden_state),
            Err(hidden_state_error) => {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %qwen3_5_runtime_error(hidden_state_error),
                    "MTP accepted a draft but could not retain the next hidden-state seed; disabling MTP for this request"
                );
                active_request.clear_optional_prediction_session();
                None
            }
        };
        if let Some(multi_token_prediction_request) =
            active_request.optional_prediction_session_mut()
        {
            multi_token_prediction_request.set_target_hidden_states(retained_hidden_state);
        }
    } else {
        active_request.measure_operation_with_request(
            crate::PerformanceOperation::MtpRejectedDraftStateRestoration,
            |active_request| {
                active_request
                    .request_decoder_state_mut()
                    .restore_verified_prefix(
                        verified_prefix_position_tokens,
                        verified_prefix_boundary_checkpoint,
                    )
                    .map_err(qwen3_5_runtime_error)
            },
        )?;
        active_request.set_next_position_tokens(verified_prefix_position_tokens);
        active_request.set_pending_generated_token(
            model
                .runtime()
                .array_from_u32(&[target_verify_token_ids[0]], &[1, 1])
                .map_err(qwen3_5_runtime_error)?,
        );
        let retained_hidden_state = match target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), 0)
        {
            Ok(hidden_state) => Some(hidden_state),
            Err(hidden_state_error) => {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %qwen3_5_runtime_error(hidden_state_error),
                    "MTP rejected a draft but could not retain the correction hidden-state seed; disabling MTP for this request"
                );
                active_request.clear_optional_prediction_session();
                None
            }
        };
        if let Some(multi_token_prediction_request) =
            active_request.optional_prediction_session_mut()
        {
            multi_token_prediction_request.set_target_hidden_states(retained_hidden_state);
        }
    }

    tracing::trace!(
        request_id = request_id.value(),
        draft_token_id,
        target_verified_token_id = target_verify_token_ids[0],
        bonus_token_id = target_verify_token_ids[1],
        accepted = accepted_draft,
        "completed depth-one MTP prefix acceptance"
    );
    Ok(if accepted_draft {
        Qwen3_5PredictionAcceptanceOutcome::Accepted
    } else {
        Qwen3_5PredictionAcceptanceOutcome::Rejected
    })
}

pub(in crate::qwen3_5) fn prediction_verification_is_eligible(
    active_request: &Qwen3_5EngineRequest,
    maximum_position_count: usize,
) -> bool {
    let Some(multi_token_prediction_request) = active_request.optional_prediction_session() else {
        return false;
    };
    multi_token_prediction_request
        .target_hidden_states()
        .is_some()
        && qwen3_5_depth_one_mtp_window_fits(
            active_request.generated_token_count(),
            active_request.maximum_output_tokens(),
            active_request.next_position_tokens(),
            maximum_position_count,
        )
        && !qwen3_5_mtp_verification_may_cross_thinking_budget(
            active_request.is_inside_thinking(),
            active_request.thinking_token_count(),
            active_request.thinking_budget(),
            2,
        )
}

pub(in crate::qwen3_5) fn take_queued_prediction_token(
    active_request: &mut Qwen3_5EngineRequest,
) -> Option<u32> {
    let multi_token_prediction_request = active_request.optional_prediction_session_mut()?;
    let queued_prediction_token_id =
        multi_token_prediction_request.take_verified_generated_token_id();
    if queued_prediction_token_id.is_some() {
        multi_token_prediction_request.clear_accepted_draft_rollback();
    }
    queued_prediction_token_id
}

pub(in crate::qwen3_5) fn disable_prediction_after_memory_admission_failure(
    active_request: &mut Qwen3_5EngineRequest,
) {
    active_request.clear_optional_prediction_session();
    active_request
        .performance_attribution_mut()
        .record_counter(PerformanceCounter::MtpMemoryAdmissionFallbackCount, 1);
}

pub(in crate::qwen3_5) fn forward_initial_target_token_with_prediction_state(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    final_prompt_token_id: u32,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    if !active_request.has_optional_prediction_session() {
        return Ok(None);
    }
    let starting_position_tokens = active_request.next_position_tokens();
    let target_forward_output = active_request
        .with_decoder_state_and_performance_attribution(
            |request_decoder_state, performance_attribution| {
                model
                    .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                        &[final_prompt_token_id],
                        starting_position_tokens,
                        request_decoder_state,
                        performance_attribution,
                    )
            },
        )?;
    active_request.advance_position(1)?;
    let first_generated_token =
        active_request.build_generated_token(model, target_forward_output.final_logits())?;
    if let Some(multi_token_prediction_request) = active_request.optional_prediction_session_mut() {
        multi_token_prediction_request.set_target_hidden_states(Some(
            target_forward_output.into_pre_final_normalization_hidden_states(),
        ));
    }
    Ok(Some(first_generated_token))
}

pub(in crate::qwen3_5) fn forward_next_target_token_with_prediction_state(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    current_generated_token: &MlxArray,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    if !active_request.has_optional_prediction_session() {
        return Ok(None);
    }
    let starting_position_tokens = active_request.next_position_tokens();
    let target_forward_output = active_request
        .with_decoder_state_and_performance_attribution(
            |request_decoder_state, performance_attribution| {
                model
                    .generated_token_forward_with_pre_final_normalization_hidden_states_and_performance_attribution(
                        current_generated_token,
                        starting_position_tokens,
                        request_decoder_state,
                        performance_attribution,
                    )
            },
        )?;
    active_request.advance_position(1)?;
    let next_generated_token =
        active_request.build_generated_token(model, target_forward_output.final_logits())?;
    if let Some(multi_token_prediction_request) = active_request.optional_prediction_session_mut() {
        multi_token_prediction_request.set_target_hidden_states(Some(
            target_forward_output.into_pre_final_normalization_hidden_states(),
        ));
    }
    Ok(Some(next_generated_token))
}

pub(in crate::qwen3_5) fn attempt_prediction_proposal_and_verification(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
    current_generated_token_id: u32,
) -> Result<(Qwen3_5PredictionAcceptanceOutcome, bool), InferenceEngineError> {
    active_request
        .performance_attribution_mut()
        .record_counter(PerformanceCounter::MtpAdmittedAttemptCount, 1);
    let proposed_mtp_draft_token_id =
        propose_depth_one_mtp_draft(model, active_request, request_id, current_generated_token);
    let target_verification_was_attempted = proposed_mtp_draft_token_id.is_some();
    let prediction_acceptance_outcome = match proposed_mtp_draft_token_id {
        Some(mtp_draft_token_id) => match verify_depth_one_mtp_prefix_acceptance(
            model,
            active_request,
            request_id,
            current_generated_token_id,
            mtp_draft_token_id,
        ) {
            Ok(prediction_acceptance_outcome) => prediction_acceptance_outcome,
            Err(target_verification_error) => {
                active_request
                    .performance_attribution_mut()
                    .record_counter(PerformanceCounter::MtpOperationalFallbackCount, 1);
                return Err(target_verification_error);
            }
        },
        None => Qwen3_5PredictionAcceptanceOutcome::OperationalFallback,
    };
    record_mtp_outcome(active_request, prediction_acceptance_outcome);
    Ok((
        prediction_acceptance_outcome,
        target_verification_was_attempted,
    ))
}

pub(in crate::qwen3_5) fn record_mtp_outcome(
    active_request: &mut Qwen3_5EngineRequest,
    prediction_acceptance_outcome: Qwen3_5PredictionAcceptanceOutcome,
) {
    let prediction_outcome_counter = match prediction_acceptance_outcome {
        Qwen3_5PredictionAcceptanceOutcome::Accepted => PerformanceCounter::MtpAcceptedDraftCount,
        Qwen3_5PredictionAcceptanceOutcome::Rejected => PerformanceCounter::MtpRejectedDraftCount,
        Qwen3_5PredictionAcceptanceOutcome::OperationalFallback => {
            PerformanceCounter::MtpOperationalFallbackCount
        }
    };
    active_request
        .performance_attribution_mut()
        .record_counter(prediction_outcome_counter, 1);
}

fn restore_target_state_after_mtp_failure(
    active_request: &mut Qwen3_5EngineRequest,
    target_state_checkpoint: crate::RequestDecoderStateStackCheckpoint,
    target_verify_start_position_tokens: u32,
) -> Result<(), InferenceEngineError> {
    active_request
        .request_decoder_state_mut()
        .restore_checkpoint(target_state_checkpoint)
        .map_err(qwen3_5_runtime_error)?;
    active_request.set_next_position_tokens(target_verify_start_position_tokens);
    Ok(())
}
