use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::{InferenceEngineError, PerformanceCounter};

use super::engine_request::AcceptedMtpDraftRollback;
use super::engine_request::Qwen3_5EngineRequest;
use super::qwen3_5_runtime_error;
use crate::qwen3_5::Qwen3_5Model;

const DEPTH_ONE_TARGET_VERIFY_TOKEN_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpPrefixAcceptanceOutcome {
    Accepted,
    Rejected,
    OperationalFallback,
}

pub(super) fn propose_depth_one_mtp_draft(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token: &MlxArray,
) -> Option<u32> {
    let Some(mut mtp_request_state) = active_request.mtp_request_state.take() else {
        return None;
    };
    let Some(mtp_target_hidden_states) = active_request.mtp_target_hidden_states.take() else {
        active_request.mtp_request_state = Some(mtp_request_state);
        return None;
    };

    let draft_token_id = match model.forward_mtp_draft_with_performance_attribution(
        &mtp_target_hidden_states,
        current_generated_token,
        &mut mtp_request_state,
        &mut active_request.performance_attribution,
    ) {
        Ok((_mtp_forward_output, draft_token_id)) => draft_token_id,
        Err(mtp_forward_error) => {
            tracing::warn!(
                request_id = request_id.value(),
                error = %mtp_forward_error,
                "MTP draft forward failed; continuing this request with target-only decode"
            );
            active_request.mtp_request_state = None;
            return None;
        }
    };
    active_request.mtp_request_state = Some(mtp_request_state);
    Some(draft_token_id)
}

pub(super) fn verify_depth_one_mtp_prefix_acceptance(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    request_id: RequestId,
    current_generated_token_id: u32,
    draft_token_id: u32,
) -> Result<MtpPrefixAcceptanceOutcome, InferenceEngineError> {
    let target_state_checkpoint = active_request
        .request_decoder_state
        .checkpoint()
        .map_err(qwen3_5_runtime_error)?;
    let target_verify_start_position_tokens = active_request.next_position_tokens;
    let (target_forward_output, target_verify_token_ids, verified_prefix_boundary_checkpoint) =
        match model.forward_depth_one_mtp_verification_with_performance_attribution(
            &[current_generated_token_id, draft_token_id],
            active_request.next_position_tokens,
            &mut active_request.request_decoder_state,
            &mut active_request.performance_attribution,
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
                active_request.mtp_request_state = None;
                return Ok(MtpPrefixAcceptanceOutcome::OperationalFallback);
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
        active_request.mtp_request_state = None;
        return Ok(MtpPrefixAcceptanceOutcome::OperationalFallback);
    }

    let force_mtp_draft_rejection =
        std::mem::take(&mut active_request.force_next_mtp_draft_rejection_for_tests);
    let accepted_draft = !force_mtp_draft_rejection && target_verify_token_ids[0] == draft_token_id;
    let verified_prefix_position_tokens = target_verify_start_position_tokens
        .checked_add(1)
        .ok_or_else(|| super::fatal_engine_error("MTP verified-prefix position overflowed"))?;
    if accepted_draft {
        active_request.accepted_mtp_draft_rollback = Some(AcceptedMtpDraftRollback {
            verified_prefix_boundary_checkpoint,
            verified_prefix_position_tokens,
        });
        active_request
            .verified_mtp_generated_token_ids
            .push_back(draft_token_id);
        active_request.pending_generated_token = Some(
            model
                .runtime()
                .array_from_u32(&[target_verify_token_ids[1]], &[1, 1])
                .map_err(qwen3_5_runtime_error)?,
        );
        active_request.mtp_target_hidden_states = match target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), 1)
        {
            Ok(hidden_state) => Some(hidden_state),
            Err(hidden_state_error) => {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %qwen3_5_runtime_error(hidden_state_error),
                    "MTP accepted a draft but could not retain the next hidden-state seed; disabling MTP for this request"
                );
                active_request.mtp_request_state = None;
                None
            }
        };
    } else {
        active_request.measure_operation_with_request(
            crate::PerformanceOperation::MtpRejectedDraftStateRestoration,
            |active_request| {
                active_request
                    .request_decoder_state
                    .restore_mtp_verified_prefix(
                        verified_prefix_position_tokens,
                        verified_prefix_boundary_checkpoint,
                    )
                    .map_err(qwen3_5_runtime_error)
            },
        )?;
        active_request.next_position_tokens = verified_prefix_position_tokens;
        active_request.pending_generated_token = Some(
            model
                .runtime()
                .array_from_u32(&[target_verify_token_ids[0]], &[1, 1])
                .map_err(qwen3_5_runtime_error)?,
        );
        active_request.mtp_target_hidden_states = match target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), 0)
        {
            Ok(hidden_state) => Some(hidden_state),
            Err(hidden_state_error) => {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %qwen3_5_runtime_error(hidden_state_error),
                    "MTP rejected a draft but could not retain the correction hidden-state seed; disabling MTP for this request"
                );
                active_request.mtp_request_state = None;
                None
            }
        };
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
        MtpPrefixAcceptanceOutcome::Accepted
    } else {
        MtpPrefixAcceptanceOutcome::Rejected
    })
}

pub(super) fn record_mtp_outcome(
    active_request: &mut Qwen3_5EngineRequest,
    mtp_prefix_acceptance_outcome: MtpPrefixAcceptanceOutcome,
) {
    let mtp_outcome_counter = match mtp_prefix_acceptance_outcome {
        MtpPrefixAcceptanceOutcome::Accepted => PerformanceCounter::MtpAcceptedDraftCount,
        MtpPrefixAcceptanceOutcome::Rejected => PerformanceCounter::MtpRejectedDraftCount,
        MtpPrefixAcceptanceOutcome::OperationalFallback => {
            PerformanceCounter::MtpOperationalFallbackCount
        }
    };
    active_request
        .performance_attribution
        .record_counter(mtp_outcome_counter, 1);
}

fn restore_target_state_after_mtp_failure(
    active_request: &mut Qwen3_5EngineRequest,
    target_state_checkpoint: crate::RequestDecoderStateStackCheckpoint,
    target_verify_start_position_tokens: u32,
) -> Result<(), InferenceEngineError> {
    active_request
        .request_decoder_state
        .restore_checkpoint(target_state_checkpoint)
        .map_err(qwen3_5_runtime_error)?;
    active_request.next_position_tokens = target_verify_start_position_tokens;
    Ok(())
}
