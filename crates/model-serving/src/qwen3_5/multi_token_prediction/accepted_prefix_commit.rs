use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;
use crate::{InferenceEngineError, PerformanceOperation};

use super::MtpVerificationDecision;
use super::request_state::Qwen3_5MtpRequestStateAllocationCheckpoint;
use super::verified_emission_queue::{VerifiedEmissionQueue, VerifiedTargetFrontier};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;
use crate::qwen3_5::model::Qwen3_5TargetForwardOutput;

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_accepted_mtp_prefix(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    target_verify_start_position_tokens: u32,
    current_generated_token: &MlxArray,
    target_hidden_seed: &MlxArray,
    predictor_checkpoint: Qwen3_5MtpRequestStateAllocationCheckpoint,
    draft_token_ids: &[u32],
    decision: &MtpVerificationDecision,
    mut prefix_boundaries: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    target_forward_output: Qwen3_5TargetForwardOutput,
) -> Result<(), InferenceEngineError> {
    let accepted_count = usize::from(decision.accepted_count());
    let retained_target_input_count = accepted_count + 1;
    let verifier_input_count = draft_token_ids.len() + 1;
    let retained_target_position_tokens =
        target_verify_start_position_tokens
            .checked_add(u32::try_from(retained_target_input_count).map_err(|_| {
                fatal_engine_error("retained MTP target prefix exceeds the u32 range")
            })?)
            .ok_or_else(|| fatal_engine_error("retained MTP target position overflowed"))?;

    let mut prefix_boundaries = prefix_boundaries.drain(..).map(Some).collect::<Vec<_>>();
    if retained_target_input_count < verifier_input_count {
        let retained_boundary = prefix_boundaries
            .get_mut(accepted_count)
            .and_then(Option::take)
            .ok_or_else(|| fatal_engine_error("MTP target repair lost its retained boundary"))?;
        active_request.measure_operation_with_request(
            PerformanceOperation::MtpTargetRepair,
            |active_request| {
                active_request
                    .request_decoder_state_mut()
                    .restore_verified_prefix(retained_target_position_tokens, retained_boundary)
                    .map_err(qwen3_5_runtime_error)
            },
        )?;
        active_request
            .performance_attribution_mut()
            .record_counter(crate::PerformanceCounter::MtpTargetRepairCount, 1);
    }
    active_request.set_next_position_tokens(retained_target_position_tokens);

    if accepted_count > 0 {
        let public_boundary = prefix_boundaries
            .get_mut(0)
            .and_then(Option::take)
            .ok_or_else(|| fatal_engine_error("MTP queue lost its initial public frontier"))?;
        let public_position_tokens = target_verify_start_position_tokens
            .checked_add(1)
            .ok_or_else(|| fatal_engine_error("MTP public frontier position overflowed"))?;
        let mut emission_queue = VerifiedEmissionQueue::new(VerifiedTargetFrontier {
            position_tokens: public_position_tokens,
            boundary: public_boundary,
        });
        for (accepted_position, token_id) in draft_token_ids[..accepted_count].iter().enumerate() {
            let is_last_accepted_token = accepted_position + 1 == accepted_count;
            let frontier_after_emission = if is_last_accepted_token {
                None
            } else {
                let boundary_index = accepted_position + 1;
                let boundary = prefix_boundaries
                    .get_mut(boundary_index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        fatal_engine_error("MTP queue lost an intermediate public frontier")
                    })?;
                Some(VerifiedTargetFrontier {
                    position_tokens: target_verify_start_position_tokens
                        .checked_add(u32::try_from(boundary_index + 1).map_err(|_| {
                            fatal_engine_error("MTP queue frontier exceeds the u32 range")
                        })?)
                        .ok_or_else(|| fatal_engine_error("MTP queue frontier overflowed"))?,
                    boundary,
                })
            };
            emission_queue.push(*token_id, frontier_after_emission);
        }
        active_request
            .optional_prediction_session_mut()
            .ok_or_else(|| fatal_engine_error("MTP request session disappeared during commit"))?
            .set_verified_emission_queue(emission_queue);
    }

    if let Some(pending_target_token_id) = decision.pending_target_token_id() {
        active_request.set_pending_generated_token(
            model
                .runtime()
                .array_from_u32(&[pending_target_token_id], &[1, 1])
                .map_err(qwen3_5_runtime_error)?,
        );
    } else {
        active_request.clear_pending_generated_token();
    }

    commit_confirmed_predictor_history(
        model,
        active_request,
        current_generated_token,
        target_hidden_seed,
        predictor_checkpoint,
        draft_token_ids.len(),
        &draft_token_ids[..accepted_count],
        &target_forward_output,
    )?;
    let retained_hidden_row_index = i32::try_from(accepted_count)
        .map_err(|_| fatal_engine_error("MTP retained hidden row exceeds Int32"))?;
    let retained_target_hidden = target_forward_output
        .pre_final_normalization_hidden_state_at(model.runtime(), retained_hidden_row_index)
        .map_err(qwen3_5_runtime_error)?;
    if let Some(prediction_request) = active_request.optional_prediction_session_mut() {
        prediction_request.set_target_hidden_states(Some(retained_target_hidden));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn commit_confirmed_predictor_history(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    current_generated_token: &MlxArray,
    target_hidden_seed: &MlxArray,
    predictor_checkpoint: Qwen3_5MtpRequestStateAllocationCheckpoint,
    proposed_draft_count: usize,
    accepted_draft_token_ids: &[u32],
    target_forward_output: &Qwen3_5TargetForwardOutput,
) -> Result<(), InferenceEngineError> {
    active_request.measure_operation_with_request(
        PerformanceOperation::MtpPredictorCommitReplay,
        |active_request| {
            active_request
                .with_optional_prediction_session_and_performance_attribution(
                    |prediction_request, performance_attribution| {
                        let mut replay_hidden_rows =
                            Vec::with_capacity(accepted_draft_token_ids.len() + 1);
                        let mut replay_token_array_owners =
                            Vec::with_capacity(accepted_draft_token_ids.len());
                        let proposal_committed_draft_count =
                            proposed_draft_count.checked_sub(1).ok_or_else(|| {
                                fatal_engine_error("MTP proposal contained no drafts")
                            })?;

                        // The proposal chain has already committed current + every draft except
                        // its final token. Reuse that exact prefix so depth-one MTP does not repeat
                        // its head forward on every attempt; restore only when rejection leaves the
                        // proposal state ahead of the accepted target frontier.
                        let accepted_draft_replay_start = if accepted_draft_token_ids.len()
                            >= proposal_committed_draft_count
                        {
                            proposal_committed_draft_count
                        } else {
                            prediction_request
                                .restore_allocation_checkpoint(predictor_checkpoint)
                                .map_err(qwen3_5_runtime_error)?;
                            let current_output = model
                                .build_mtp_draft_graph(
                                    target_hidden_seed,
                                    current_generated_token,
                                    prediction_request.request_state_mut(),
                                    performance_attribution,
                                )
                                .map_err(InferenceEngineError::from)?;
                            let (_, current_post_normalized_hidden) = current_output.into_arrays();
                            replay_hidden_rows.push(current_post_normalized_hidden);
                            0
                        };
                        for (accepted_position, accepted_token_id) in accepted_draft_token_ids
                            .iter()
                            .enumerate()
                            .skip(accepted_draft_replay_start)
                        {
                            let token_array = model
                                .runtime()
                                .array_from_u32(&[*accepted_token_id], &[1, 1])
                                .map_err(qwen3_5_runtime_error)?;
                            let target_hidden_row = target_forward_output
                                .pre_final_normalization_hidden_state_at(
                                    model.runtime(),
                                    i32::try_from(accepted_position).map_err(|_| {
                                        fatal_engine_error("MTP predictor replay row exceeds Int32")
                                    })?,
                                )
                                .map_err(qwen3_5_runtime_error)?;
                            let replay_output = model
                                .build_mtp_draft_graph(
                                    &target_hidden_row,
                                    &token_array,
                                    prediction_request.request_state_mut(),
                                    performance_attribution,
                                )
                                .map_err(InferenceEngineError::from)?;
                            let (_, replay_post_normalized_hidden) = replay_output.into_arrays();
                            // MLX graphs retain these token inputs lazily until the combined state
                            // evaluation commits every repaired predictor row.
                            replay_token_array_owners.push(token_array);
                            replay_hidden_rows.push(replay_post_normalized_hidden);
                        }
                        if !replay_hidden_rows.is_empty() {
                            let replay_roots = replay_hidden_rows.iter().collect::<Vec<_>>();
                            model
                                .evaluate_mtp_updated_state(
                                    prediction_request.request_state_mut(),
                                    &replay_roots,
                                    performance_attribution,
                                )
                                .map_err(InferenceEngineError::from)?;
                        }
                        performance_attribution.record_counter(
                            crate::PerformanceCounter::MtpPredictorReplayTokenCount,
                            u64::try_from(replay_hidden_rows.len()).unwrap_or(u64::MAX),
                        );
                        Ok::<(), InferenceEngineError>(())
                    },
                )
                .ok_or_else(|| {
                    fatal_engine_error("MTP request session disappeared during predictor repair")
                })?
        },
    )
}
