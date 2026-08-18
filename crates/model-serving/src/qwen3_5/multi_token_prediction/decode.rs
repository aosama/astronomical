use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;
use crate::{InferenceEngineError, PerformanceCounter};

use super::accepted_prefix_commit::commit_accepted_mtp_prefix;
use super::target_verification::forward_target_verification_window_with_performance_attribution;
use super::{
    MtpDepthDowngradeReason, MtpDraftDepth, MtpVerificationDecision,
    qwen3_5_mtp_effective_depth_and_reason_for_windows, qwen3_5_mtp_effective_depth_for_windows,
    qwen3_5_mtp_verification_decision,
};

/// Returns whether a prediction window could cross the forced thinking boundary.
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

/// Preserves the phase-one depth-one contract for direct runtime callers.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_depth_one_mtp_window_fits(
    generated_token_count: u16,
    maximum_output_tokens: u16,
    next_position_tokens: u32,
    maximum_position_count: usize,
) -> bool {
    qwen3_5_mtp_effective_depth_for_windows(
        MtpDraftDepth::DEPTH_ONE,
        generated_token_count,
        maximum_output_tokens,
        next_position_tokens,
        maximum_position_count,
        false,
        0,
        None,
    )
    .is_some()
}

pub(in crate::qwen3_5) fn effective_prediction_depth(
    active_request: &Qwen3_5EngineRequest,
    maximum_position_count: usize,
) -> (Option<MtpDraftDepth>, Option<MtpDepthDowngradeReason>) {
    let Some(prediction_request) = active_request.optional_prediction_session() else {
        return (None, None);
    };
    if prediction_request.target_hidden_states().is_none() {
        return (None, None);
    }
    qwen3_5_mtp_effective_depth_and_reason_for_windows(
        prediction_request.requested_depth(),
        active_request.generated_token_count(),
        active_request.maximum_output_tokens(),
        active_request.next_position_tokens(),
        maximum_position_count,
        active_request.is_inside_thinking(),
        active_request.thinking_token_count(),
        active_request.thinking_budget(),
    )
}

pub(in crate::qwen3_5) fn projected_verification_window_memory_growth_bytes(
    model: &Qwen3_5Model,
    active_request: &Qwen3_5EngineRequest,
    effective_depth: MtpDraftDepth,
) -> Result<usize, InferenceEngineError> {
    let full_attention_bytes_per_layer_token = model
        .config()
        .full_attention_key_value_state_bytes_per_layer_token()
        .ok_or_else(|| {
            fatal_engine_error("prediction full-attention bytes per layer token overflowed")
        })?;
    let sequential_update_token_counts = vec![1; usize::from(effective_depth.get())];
    active_request
        .optional_prediction_session()
        .ok_or_else(|| fatal_engine_error("active prediction request state disappeared"))?
        .projected_sequential_full_attention_growth_bytes(
            full_attention_bytes_per_layer_token,
            &sequential_update_token_counts,
        )
        .map_err(qwen3_5_runtime_error)
}

/// Operation-local logits, hidden rows, and decision vectors. Snapshots are excluded.
pub(in crate::qwen3_5) fn verification_transient_array_bytes(
    model: &Qwen3_5Model,
    effective_depth: MtpDraftDepth,
) -> Result<usize, InferenceEngineError> {
    super::qwen3_5_mtp_verification_transient_array_bytes(
        effective_depth,
        model.config().vocabulary_size() as usize,
        model.config().hidden_size() as usize,
    )
    .map_err(|_| fatal_engine_error("MTP verification transient arrays overflowed"))
}

pub(in crate::qwen3_5) fn verification_boundary_snapshot_bytes(
    model: &Qwen3_5Model,
    effective_depth: MtpDraftDepth,
) -> Result<usize, InferenceEngineError> {
    model
        .decoder_cache_layout()
        .boundary_snapshot_payload_byte_count()
        .map_err(|decoder_cache_layout_error| {
            fatal_engine_error(format!(
                "failed to project target verification-window workspace: {decoder_cache_layout_error}"
            ))
        })?
        .checked_mul(usize::from(effective_depth.get()))
        .ok_or_else(|| fatal_engine_error("target verification boundary workspace overflowed"))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::qwen3_5) fn attempt_prediction_proposal_and_verification(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
    current_generated_token_id: u32,
    effective_depth: MtpDraftDepth,
    end_of_sequence_token_ids: &[u32],
) -> Result<MtpVerificationDecision, InferenceEngineError> {
    active_request
        .performance_attribution_mut()
        .record_counter(PerformanceCounter::MtpAdmittedAttemptCount, 1);
    let requested_depth = active_request
        .optional_prediction_session()
        .ok_or_else(|| fatal_engine_error("MTP request session disappeared before proposal"))?
        .requested_depth();
    active_request.performance_attribution_mut().record_counter(
        PerformanceCounter::MtpRequestedDepthTotal,
        u64::from(requested_depth.get()),
    );
    active_request.performance_attribution_mut().record_counter(
        PerformanceCounter::MtpEffectiveDepthTotal,
        u64::from(effective_depth.get()),
    );
    let mut prediction_request = active_request
        .take_optional_prediction_session()
        .ok_or_else(|| fatal_engine_error("MTP request session disappeared before proposal"))?;
    let target_hidden_seed = prediction_request
        .take_target_hidden_states()
        .ok_or_else(|| fatal_engine_error("MTP proposal lost its target hidden seed"))?;
    // Operational rollback retains the pre-proposal frontier. Successful
    // commits reuse the exact target-authoritative base produced by the proposal itself.
    let predictor_rollback_checkpoint = prediction_request
        .allocation_checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let proposal = model.propose_mtp_chain_with_performance_attribution(
        &target_hidden_seed,
        current_generated_token,
        effective_depth,
        prediction_request.request_state_mut(),
        active_request.performance_attribution_mut(),
    );
    let proposal = match proposal {
        Ok(proposal) => proposal,
        Err(proposal_error) => {
            // Proposal mutates only private predictor state. Put the session back so the
            // request can drop that mutated predictor atomically, then quarantine MTP for
            // the rest of this request. Target decoder state is still the pre-attempt frontier.
            active_request.restore_optional_prediction_session(prediction_request);
            active_request.clear_optional_prediction_session();
            tracing::warn!(
                request_id = request_id.value(),
                error = %proposal_error,
                "MTP proposal chain failed; continuing this request with target-only decode"
            );
            active_request
                .performance_attribution_mut()
                .record_counter(PerformanceCounter::MtpOperationalFallbackCount, 1);
            return Ok(MtpVerificationDecision::operational_fallback(
                effective_depth,
            ));
        }
    };
    let (draft_token_ids, predictor_base_checkpoint) = proposal.into_parts();
    active_request.performance_attribution_mut().record_counter(
        PerformanceCounter::MtpProposedDraftCount,
        u64::try_from(draft_token_ids.len()).unwrap_or(u64::MAX),
    );
    active_request.restore_optional_prediction_session(prediction_request);

    let target_state_checkpoint = active_request
        .request_decoder_state()
        .checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let target_verify_start_position_tokens = active_request.next_position_tokens();
    let mut verifier_input_token_ids = Vec::with_capacity(draft_token_ids.len() + 1);
    verifier_input_token_ids.push(current_generated_token_id);
    verifier_input_token_ids.extend_from_slice(&draft_token_ids);
    let verification_output = active_request.with_decoder_state_and_performance_attribution(
        |request_decoder_state, performance_attribution| {
            forward_target_verification_window_with_performance_attribution(
                model,
                &verifier_input_token_ids,
                target_verify_start_position_tokens,
                request_decoder_state,
                performance_attribution,
            )
        },
    );
    let mut verification_output = match verification_output {
        Ok(verification_output) => verification_output,
        Err(verification_error) => {
            restore_complete_attempt_state(
                active_request,
                target_state_checkpoint,
                predictor_rollback_checkpoint,
                target_verify_start_position_tokens,
            )?;
            tracing::warn!(
                request_id = request_id.value(),
                error = %verification_error,
                "MTP target verification failed; continuing this request with target-only decode"
            );
            return Ok(MtpVerificationDecision::operational_fallback(
                effective_depth,
            ));
        }
    };
    active_request.advance_position(verifier_input_token_ids.len())?;
    if active_request
        .optional_prediction_session_mut()
        .is_some_and(|prediction_request| {
            prediction_request.take_forced_draft_rejection_for_tests()
        })
        && !verification_output.target_token_ids.is_empty()
    {
        verification_output.target_token_ids[0] = draft_token_ids[0].wrapping_add(1);
    }
    let decision = qwen3_5_mtp_verification_decision(
        effective_depth,
        &draft_token_ids,
        &verification_output.target_token_ids,
        end_of_sequence_token_ids,
    )
    .map_err(|_| fatal_engine_error("MTP verification returned inconsistent bounded vectors"))?;
    if decision.was_eos_truncated() {
        active_request
            .performance_attribution_mut()
            .record_counter(PerformanceCounter::MtpEosTruncatedPrefixCount, 1);
    }
    let commit_outcome = commit_accepted_mtp_prefix(
        model,
        active_request,
        target_verify_start_position_tokens,
        predictor_base_checkpoint,
        &draft_token_ids,
        &decision,
        verification_output,
    );
    if let Err(commit_error) = commit_outcome {
        restore_complete_attempt_state(
            active_request,
            target_state_checkpoint,
            predictor_rollback_checkpoint,
            target_verify_start_position_tokens,
        )?;
        tracing::warn!(
            request_id = request_id.value(),
            error = %commit_error,
            "MTP state repair failed; continuing this request with target-only decode"
        );
        return Ok(MtpVerificationDecision::operational_fallback(
            effective_depth,
        ));
    }
    drop(target_state_checkpoint);
    record_mtp_outcome(active_request, &decision);
    Ok(decision)
}

pub(in crate::qwen3_5) fn take_queued_prediction_token(
    active_request: &mut Qwen3_5EngineRequest,
) -> Option<u32> {
    active_request
        .optional_prediction_session_mut()?
        .take_verified_generated_token_id()
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
    let target_forward_output = active_request.with_decoder_state_and_performance_attribution(
        |request_decoder_state, performance_attribution| {
            model.forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
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
    if let Some(prediction_request) = active_request.optional_prediction_session_mut() {
        prediction_request.set_target_hidden_states(Some(
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
    let target_forward_output = active_request.with_decoder_state_and_performance_attribution(
        |request_decoder_state, performance_attribution| {
            model.generated_token_forward_with_pre_final_normalization_hidden_states_and_performance_attribution(
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
    if let Some(prediction_request) = active_request.optional_prediction_session_mut() {
        prediction_request.set_target_hidden_states(Some(
            target_forward_output.into_pre_final_normalization_hidden_states(),
        ));
    }
    Ok(Some(next_generated_token))
}

fn record_mtp_outcome(
    active_request: &mut Qwen3_5EngineRequest,
    decision: &MtpVerificationDecision,
) {
    let accepted_count = u64::from(decision.accepted_count());
    let proposed_count = u64::from(decision.proposed_count());
    active_request
        .performance_attribution_mut()
        .record_counter(PerformanceCounter::MtpAcceptedDraftCount, accepted_count);
    active_request.performance_attribution_mut().record_counter(
        PerformanceCounter::MtpRejectedDraftCount,
        proposed_count.saturating_sub(accepted_count),
    );
    let proposed_position_counters = [
        PerformanceCounter::MtpProposedDraftPositionOneCount,
        PerformanceCounter::MtpProposedDraftPositionTwoCount,
        PerformanceCounter::MtpProposedDraftPositionThreeCount,
    ];
    let accepted_position_counters = [
        PerformanceCounter::MtpAcceptedDraftPositionOneCount,
        PerformanceCounter::MtpAcceptedDraftPositionTwoCount,
        PerformanceCounter::MtpAcceptedDraftPositionThreeCount,
    ];
    let rejected_position_counters = [
        PerformanceCounter::MtpRejectedDraftPositionOneCount,
        PerformanceCounter::MtpRejectedDraftPositionTwoCount,
        PerformanceCounter::MtpRejectedDraftPositionThreeCount,
    ];
    for (draft_position, proposed_position_counter) in proposed_position_counters
        .into_iter()
        .enumerate()
        .take(usize::from(decision.proposed_count()))
    {
        active_request
            .performance_attribution_mut()
            .record_counter(proposed_position_counter, 1);
        let outcome_counter = if draft_position < usize::from(decision.accepted_count()) {
            accepted_position_counters[draft_position]
        } else {
            rejected_position_counters[draft_position]
        };
        active_request
            .performance_attribution_mut()
            .record_counter(outcome_counter, 1);
    }
}

fn restore_complete_attempt_state(
    active_request: &mut Qwen3_5EngineRequest,
    target_state_checkpoint: crate::RequestDecoderStateStackCheckpoint,
    predictor_state_checkpoint: super::Qwen3_5MtpRequestStateAllocationCheckpoint,
    target_verify_start_position_tokens: u32,
) -> Result<(), InferenceEngineError> {
    active_request
        .request_decoder_state_mut()
        .restore_checkpoint(target_state_checkpoint)
        .map_err(qwen3_5_runtime_error)?;
    if let Some(prediction_request) = active_request.optional_prediction_session_mut() {
        prediction_request
            .restore_allocation_checkpoint(predictor_state_checkpoint)
            .map_err(qwen3_5_runtime_error)?;
    }
    active_request.set_next_position_tokens(target_verify_start_position_tokens);
    active_request.clear_pending_generated_token();
    active_request.clear_optional_prediction_session();
    active_request
        .performance_attribution_mut()
        .record_counter(PerformanceCounter::MtpOperationalFallbackCount, 1);
    Ok(())
}
