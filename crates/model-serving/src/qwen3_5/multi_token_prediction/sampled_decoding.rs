//! Decoding-side scaffolding for sampled multi-token-prediction attempts.
//!
//! Checkpoints, rollback, operational fallback, and prefix commit are shared
//! with the greedy attempt in `decode.rs`; this module only owns the sampled
//! proposal/verify wiring and the request's keyed random stream handling.

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;
use crate::{InferenceEngineError, PerformanceCounter};

use super::accepted_prefix_commit::commit_accepted_mtp_prefix;
use super::decode::{record_mtp_outcome, restore_complete_attempt_state};
use super::sampled_verification::MtpSampledSamplingSettings;
use super::{MtpDraftDepth, MtpVerificationDecision, qwen3_5_mtp_sampled_verification_decision};

#[allow(clippy::too_many_arguments)]
pub(in crate::qwen3_5) fn attempt_sampled_prediction_proposal_and_verification(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
    current_generated_token_id: u32,
    effective_depth: MtpDraftDepth,
    end_of_sequence_token_ids: &[u32],
    sampling_settings: MtpSampledSamplingSettings,
) -> Result<MtpVerificationDecision, InferenceEngineError> {
    let mut sampling_random_state = active_request.take_sampling_random_state()?;
    let attempt_outcome = run_sampled_prediction_attempt(
        model,
        active_request,
        request_id,
        current_generated_token,
        current_generated_token_id,
        effective_depth,
        end_of_sequence_token_ids,
        sampling_settings,
        &mut sampling_random_state,
    );
    active_request.restore_sampling_random_state(sampling_random_state);
    attempt_outcome
}

#[allow(clippy::too_many_arguments)]
fn run_sampled_prediction_attempt(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
    current_generated_token_id: u32,
    effective_depth: MtpDraftDepth,
    end_of_sequence_token_ids: &[u32],
    sampling_settings: MtpSampledSamplingSettings,
    sampling_random_state: &mut MlxArray,
) -> Result<MtpVerificationDecision, InferenceEngineError> {
    let mut prediction_request = active_request
        .take_optional_prediction_session()
        .ok_or_else(|| fatal_engine_error("MTP request session disappeared before proposal"))?;
    let target_hidden_seed = prediction_request
        .take_target_hidden_states()
        .ok_or_else(|| fatal_engine_error("MTP proposal lost its target hidden seed"))?;
    // Two retained frontiers keep commit repair and operational rollback independent. Commit
    // repair consumes its frontier only after a rejection leaves proposal state too far ahead.
    let predictor_commit_checkpoint = prediction_request
        .allocation_checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let predictor_rollback_checkpoint = prediction_request
        .allocation_checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let proposal = model.propose_sampled_mtp_chain_with_performance_attribution(
        &target_hidden_seed,
        current_generated_token,
        effective_depth,
        prediction_request.request_state_mut(),
        sampling_settings,
        sampling_random_state,
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
                "sampled MTP proposal chain failed; continuing this request with target-only decode"
            );
            active_request
                .performance_attribution_mut()
                .record_counter(PerformanceCounter::MtpOperationalFallbackCount, 1);
            return Ok(MtpVerificationDecision::operational_fallback(
                effective_depth,
            ));
        }
    };
    let draft_token_ids = proposal.draft_token_ids.clone();
    active_request.performance_attribution_mut().record_counter(
        PerformanceCounter::MtpProposedDraftCount,
        u64::try_from(draft_token_ids.len()).unwrap_or(u64::MAX),
    );
    active_request.restore_optional_prediction_session(prediction_request);
    let forced_rejection_flag = active_request
        .optional_prediction_session_mut()
        .is_some_and(|prediction_request| {
            prediction_request.take_forced_draft_rejection_for_tests()
        });

    let target_state_checkpoint = active_request
        .request_decoder_state()
        .checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let target_verify_start_position_tokens = active_request.next_position_tokens();
    let mut verifier_input_token_ids = Vec::with_capacity(draft_token_ids.len() + 1);
    verifier_input_token_ids.push(current_generated_token_id);
    verifier_input_token_ids.extend_from_slice(&draft_token_ids);
    let verification = active_request.with_decoder_state_and_performance_attribution(
        |request_decoder_state, performance_attribution| {
            model.verify_sampled_mtp_window_with_performance_attribution(
                &verifier_input_token_ids,
                target_verify_start_position_tokens,
                request_decoder_state,
                proposal,
                sampling_settings,
                forced_rejection_flag,
                sampling_random_state,
                performance_attribution,
            )
        },
    );
    let verified_sampled_output = match verification {
        Ok(verified_sampled_output) => verified_sampled_output,
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
                "sampled MTP verification failed; continuing this request with target-only decode"
            );
            return Ok(MtpVerificationDecision::operational_fallback(
                effective_depth,
            ));
        }
    };
    active_request.advance_position(verifier_input_token_ids.len())?;
    let decision = qwen3_5_mtp_sampled_verification_decision(
        effective_depth,
        &draft_token_ids,
        &verified_sampled_output.accepted_coin_flags,
        Some(verified_sampled_output.post_prefix_token_id),
        end_of_sequence_token_ids,
    )
    .map_err(|_| fatal_engine_error("MTP sampled verification returned inconsistent vectors"))?;
    if decision.was_eos_truncated() {
        active_request
            .performance_attribution_mut()
            .record_counter(PerformanceCounter::MtpEosTruncatedPrefixCount, 1);
    }
    let commit_outcome = commit_accepted_mtp_prefix(
        model,
        active_request,
        target_verify_start_position_tokens,
        current_generated_token,
        &target_hidden_seed,
        predictor_commit_checkpoint,
        &draft_token_ids,
        &decision,
        verified_sampled_output.prefix_boundaries,
        verified_sampled_output.target_forward_output,
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
            "sampled MTP state repair failed; continuing this request with target-only decode"
        );
        return Ok(MtpVerificationDecision::operational_fallback(
            effective_depth,
        ));
    }
    record_mtp_outcome(active_request, &decision);
    Ok(decision)
}
