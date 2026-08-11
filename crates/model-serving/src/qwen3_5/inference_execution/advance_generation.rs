//! Serial request advancement with one-token-ahead graphics-processor submission.
//!
//! The token returned to the user is normally the predecessor of the token
//! already being evaluated.

use astronomical_ipc_protocol::RequestId;

use crate::{
    AdaptiveRamGrowthContext, GeneratedToken, InferenceEngineError, PerformanceAttributionOutcome,
    PerformanceOperation,
};

use super::generated_token_emission::synchronize_generated_token_id;
use super::memory_admission::{
    AdaptiveRamGrowthMemoryAdmissionError, collect_completed_forward_memory_snapshot,
    record_completed_adaptive_ram_growth,
};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::multi_token_prediction::{
    Qwen3_5PredictionAcceptanceOutcome, attempt_prediction_proposal_and_verification,
    disable_prediction_after_memory_admission_failure,
    forward_initial_target_token_with_prediction_state,
    forward_next_target_token_with_prediction_state, prediction_verification_is_eligible,
    projected_verification_window_memory_growth_bytes, take_queued_prediction_token,
    verification_window_workspace_bytes,
};
impl Qwen3_5EngineState {
    pub(super) fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let mut active_request = self.active_request.take().ok_or_else(|| {
            fatal_engine_error("Qwen3.5 generation advance requested without an active request")
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(fatal_engine_error(
                "Qwen3.5 generation request correlation mismatch",
            ));
        }

        let advance_span = if active_request.prefill_cursor
            < active_request.input_token_ids.len().saturating_sub(1)
        {
            PerformanceOperation::PromptPrefillAdvanceSpan
        } else {
            PerformanceOperation::DecodeAdvanceSpan
        };
        let advance_span_started_at = active_request
            .performance_attribution
            .begin_operation_span();
        let active_request_advance = self.advance_active_request(request_id, &mut active_request);
        active_request
            .performance_attribution
            .complete_operation_span(advance_span, advance_span_started_at);
        match active_request_advance {
            Ok(ActiveRequestAdvance::Continue(generated_token)) => {
                self.active_request = Some(active_request);
                Ok(generated_token)
            }
            Ok(ActiveRequestAdvance::Complete(generated_token)) => {
                let generation_finalization = self.finalize_generation_request(
                    active_request,
                    PerformanceAttributionOutcome::Success,
                    None,
                );
                Ok(generated_token.with_generation_finalization(generation_finalization))
            }
            Err(generation_error) => {
                self.finalize_generation_request_after_error(
                    active_request,
                    &generation_error,
                    "generation advance rejected",
                    "generation advance failed",
                );
                Err(generation_error)
            }
        }
    }

    fn advance_active_request(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        if let Some(prefill_progress) =
            self.advance_prompt_prefill_if_pending(request_id, active_request)?
        {
            return Ok(ActiveRequestAdvance::Continue(prefill_progress));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let final_prompt_index = active_request.input_token_ids.len() - 1;

        if let Some(queued_prediction_token_id) = take_queued_prediction_token(active_request) {
            let mlx_memory_snapshot = if !self.adaptive_ram_growth_guard_enabled {
                None
            } else {
                Some(
                    model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_runtime_error)?,
                )
            };
            let generated_token_emission = self.build_generated_token_emission(
                model,
                active_request,
                queued_prediction_token_id,
                mlx_memory_snapshot.as_ref(),
            )?;
            return if generated_token_emission.is_terminal {
                Ok(ActiveRequestAdvance::Complete(
                    generated_token_emission.generated_token,
                ))
            } else {
                Ok(ActiveRequestAdvance::Continue(
                    generated_token_emission.generated_token,
                ))
            };
        }

        let current_generated_token = match active_request.pending_generated_token.take() {
            Some(pending_generated_token) => pending_generated_token,
            None => {
                let final_prompt_token_id = active_request.input_token_ids[final_prompt_index];
                let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
                    1,
                    active_request.has_optional_prediction_session(),
                    model.sparse_experts_are_paged(),
                );
                let active_memory_bytes_before_growth = self
                    .measure_adaptive_ram_growth_memory_admission(
                        adaptive_ram_growth_context,
                        &mut active_request.performance_attribution,
                        &active_request.request_decoder_state,
                        0,
                        0,
                    )?;
                self.save_speculative_prefill_target_prefix(active_request)?;
                let first_generated_token = if let Some(prediction_token) =
                    forward_initial_target_token_with_prediction_state(
                        model,
                        active_request,
                        final_prompt_token_id,
                    )? {
                    prediction_token
                } else {
                    let final_prompt_logits = model
                        .build_forward_chunk_with_performance_attribution(
                            &[final_prompt_token_id],
                            active_request.next_position_tokens,
                            &mut active_request.request_decoder_state,
                            &mut active_request.performance_attribution,
                        )
                        .map_err(InferenceEngineError::from)?;
                    active_request.advance_position(1)?;
                    active_request.build_generated_token(model, &final_prompt_logits)?
                };
                active_request
                    .performance_attribution
                    .measure_operation(
                        PerformanceOperation::DecodeAsyncEvaluationSubmission,
                        |_performance_attribution| {
                            model.async_evaluate_generation(
                                &first_generated_token,
                                &active_request.request_decoder_state,
                            )
                        },
                    )
                    .map_err(InferenceEngineError::from)?;
                record_completed_adaptive_ram_growth(
                    &mut self.adaptive_ram_growth_guard,
                    adaptive_ram_growth_context
                        .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                    true,
                    model,
                    active_memory_bytes_before_growth,
                    0,
                    &mut active_request.performance_attribution,
                )?;
                first_generated_token
            }
        };

        let current_generated_token_id =
            synchronize_generated_token_id(active_request, &current_generated_token)?;
        if self.generated_token_will_be_terminal(active_request, current_generated_token_id) {
            let mlx_memory_snapshot = if self.adaptive_ram_growth_guard_enabled {
                Some(
                    model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_runtime_error)?,
                )
            } else {
                None
            };
            let generated_token_emission = self.build_generated_token_emission(
                model,
                active_request,
                current_generated_token_id,
                mlx_memory_snapshot.as_ref(),
            )?;
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }

        if prediction_verification_is_eligible(active_request, self.maximum_position_count) {
            let prediction_history_growth_bytes =
                projected_verification_window_memory_growth_bytes(model, active_request)?;
            let target_verification_boundary_workspace_bytes =
                verification_window_workspace_bytes(model)?;
            let adaptive_ram_growth_context =
                AdaptiveRamGrowthContext::decode(2, true, model.sparse_experts_are_paged());
            let memory_admission_outcome = self.measure_adaptive_ram_growth_memory_admission(
                adaptive_ram_growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                prediction_history_growth_bytes,
                target_verification_boundary_workspace_bytes,
            );
            match memory_admission_outcome {
                Err(AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason }) => {
                    tracing::warn!(
                        request_id = request_id.value(),
                        reason,
                        "prediction verification memory admission fell back to target-only decode"
                    );
                    disable_prediction_after_memory_admission_failure(active_request);
                }
                Err(AdaptiveRamGrowthMemoryAdmissionError::Engine(memory_admission_error)) => {
                    return Err(memory_admission_error);
                }
                Ok(active_memory_bytes_before_growth) => {
                    let prediction_attempt_outcome = attempt_prediction_proposal_and_verification(
                        model,
                        active_request,
                        request_id,
                        &current_generated_token,
                        current_generated_token_id,
                    );
                    match prediction_attempt_outcome {
                        Ok((prediction_acceptance_outcome, _target_verification_was_attempted))
                            if prediction_acceptance_outcome
                                != Qwen3_5PredictionAcceptanceOutcome::OperationalFallback =>
                        {
                            let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
                                &mut self.adaptive_ram_growth_guard,
                                adaptive_ram_growth_context.with_sparse_experts_are_paged(
                                    model.sparse_experts_are_paged(),
                                ),
                                true,
                                model,
                                active_memory_bytes_before_growth,
                                target_verification_boundary_workspace_bytes,
                                &mut active_request.performance_attribution,
                            )?;
                            let generated_token_emission = self.build_generated_token_emission(
                                model,
                                active_request,
                                current_generated_token_id,
                                mlx_memory_snapshot.as_ref(),
                            )?;
                            return if generated_token_emission.is_terminal {
                                Ok(ActiveRequestAdvance::Complete(
                                    generated_token_emission.generated_token,
                                ))
                            } else {
                                Ok(ActiveRequestAdvance::Continue(
                                    generated_token_emission.generated_token,
                                ))
                            };
                        }
                        Ok((_prediction_acceptance_outcome, target_verification_was_attempted)) => {
                            record_completed_adaptive_ram_growth(
                                &mut self.adaptive_ram_growth_guard,
                                adaptive_ram_growth_context.with_sparse_experts_are_paged(
                                    model.sparse_experts_are_paged(),
                                ),
                                true,
                                model,
                                active_memory_bytes_before_growth,
                                if target_verification_was_attempted {
                                    target_verification_boundary_workspace_bytes
                                } else {
                                    0
                                },
                                &mut active_request.performance_attribution,
                            )?;
                        }
                        Err(target_verification_error) => {
                            record_completed_adaptive_ram_growth(
                                &mut self.adaptive_ram_growth_guard,
                                adaptive_ram_growth_context.with_sparse_experts_are_paged(
                                    model.sparse_experts_are_paged(),
                                ),
                                true,
                                model,
                                active_memory_bytes_before_growth,
                                target_verification_boundary_workspace_bytes,
                                &mut active_request.performance_attribution,
                            )?;
                            return Err(target_verification_error);
                        }
                    }
                }
            }
        }

        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
            1,
            active_request.has_optional_prediction_session(),
            model.sparse_experts_are_paged(),
        );
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            0,
            0,
        )?;
        let next_generated_token = if let Some(prediction_token) =
            forward_next_target_token_with_prediction_state(
                model,
                active_request,
                &current_generated_token,
            )? {
            prediction_token
        } else {
            let next_logits = model
                .build_generated_token_forward_with_performance_attribution(
                    &current_generated_token,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            let next_generated_token = active_request.build_generated_token(model, &next_logits)?;
            next_generated_token
        };
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::DecodeAsyncEvaluationSubmission,
                |_performance_attribution| {
                    model.async_evaluate_generation(
                        &next_generated_token,
                        &active_request.request_decoder_state,
                    )
                },
            )
            .map_err(InferenceEngineError::from)?;
        let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            true,
            model,
            active_memory_bytes_before_growth,
            0,
            &mut active_request.performance_attribution,
        )?;

        let generated_token_emission = self.build_generated_token_emission(
            model,
            active_request,
            current_generated_token_id,
            mlx_memory_snapshot.as_ref(),
        )?;
        if generated_token_emission.is_terminal {
            Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ))
        } else {
            // Keep the asynchronously submitted successor private until the
            // next call synchronizes it. The current token alone is observable.
            active_request.pending_generated_token = Some(next_generated_token);
            Ok(ActiveRequestAdvance::Continue(
                generated_token_emission.generated_token,
            ))
        }
    }
}

enum ActiveRequestAdvance {
    Continue(GeneratedToken),
    Complete(GeneratedToken),
}
