use astronomical_ipc_protocol::RequestId;

use crate::{
    AdaptiveRamGrowthContext, GeneratedToken, InferenceEngineError, PerformanceAttributionOutcome,
    PerformanceCounter, PerformanceOperation,
};

use super::generated_token_emission::{
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_mtp_verification_may_cross_thinking_budget,
    synchronize_generated_token_id,
};
use super::memory_admission::{
    collect_completed_forward_memory_snapshot, record_completed_adaptive_ram_growth,
};
use super::mtp_prefix_acceptance::{
    MtpPrefixAcceptanceOutcome, propose_depth_one_mtp_draft, record_mtp_outcome,
    verify_depth_one_mtp_prefix_acceptance,
};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
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

        if let Some(verified_mtp_generated_token_id) =
            active_request.verified_mtp_generated_token_ids.pop_front()
        {
            active_request.accepted_mtp_draft_rollback = None;
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
                verified_mtp_generated_token_id,
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
                    active_request.mtp_request_state.is_some(),
                    model.sparse_experts_are_paged(),
                );
                let active_memory_bytes_before_growth = self
                    .measure_adaptive_ram_growth_memory_admission(
                        adaptive_ram_growth_context,
                        &mut active_request.performance_attribution,
                        &active_request.request_decoder_state,
                        0,
                    )?;
                let first_generated_token = if active_request.mtp_request_state.is_some() {
                    let target_forward_output = model
                        .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                            &[final_prompt_token_id],
                            active_request.next_position_tokens,
                            &mut active_request.request_decoder_state,
                            &mut active_request.performance_attribution,
                        )
                        .map_err(InferenceEngineError::from)?;
                    active_request.advance_position(1)?;
                    let first_generated_token = active_request
                        .build_generated_token(model, target_forward_output.final_logits())?;
                    active_request.mtp_target_hidden_states =
                        Some(target_forward_output.into_pre_final_normalization_hidden_states());
                    first_generated_token
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

        if active_request.mtp_request_state.is_some()
            && active_request.mtp_target_hidden_states.is_some()
            && qwen3_5_depth_one_mtp_window_fits(
                active_request.generated_token_count,
                active_request.maximum_output_tokens,
                active_request.next_position_tokens,
                self.maximum_position_count,
            )
            && !qwen3_5_mtp_verification_may_cross_thinking_budget(
                active_request.is_inside_thinking,
                active_request.thinking_token_count,
                active_request.thinking_budget,
                2,
            )
        {
            let mtp_full_attention_bytes_per_layer_token = model
                .config()
                .full_attention_key_value_state_bytes_per_layer_token()
                .ok_or_else(|| {
                    fatal_engine_error("MTP full-attention bytes per layer token overflowed")
                })?;
            let mtp_full_attention_growth_bytes = active_request
                .mtp_request_state
                .as_ref()
                .ok_or_else(|| fatal_engine_error("active MTP request state disappeared"))?
                .projected_sequential_capacity_growth_bytes(
                    mtp_full_attention_bytes_per_layer_token,
                    &[1, 1],
                )
                .map_err(qwen3_5_runtime_error)?;
            let adaptive_ram_growth_context =
                AdaptiveRamGrowthContext::decode(2, true, model.sparse_experts_are_paged());
            let active_memory_bytes_before_growth = self
                .measure_adaptive_ram_growth_memory_admission(
                    adaptive_ram_growth_context,
                    &mut active_request.performance_attribution,
                    &active_request.request_decoder_state,
                    mtp_full_attention_growth_bytes,
                )?;
            active_request
                .performance_attribution
                .record_counter(PerformanceCounter::MtpAdmittedAttemptCount, 1);
            let proposed_mtp_draft_token_id = propose_depth_one_mtp_draft(
                model,
                active_request,
                request_id,
                &current_generated_token,
            );
            let mtp_prefix_acceptance_outcome = match proposed_mtp_draft_token_id {
                Some(mtp_draft_token_id) => {
                    match verify_depth_one_mtp_prefix_acceptance(
                        model,
                        active_request,
                        request_id,
                        current_generated_token_id,
                        mtp_draft_token_id,
                    ) {
                        Ok(mtp_prefix_acceptance_outcome) => mtp_prefix_acceptance_outcome,
                        Err(target_verification_error) => {
                            active_request
                                .performance_attribution
                                .record_counter(PerformanceCounter::MtpOperationalFallbackCount, 1);
                            record_completed_adaptive_ram_growth(
                                &mut self.adaptive_ram_growth_guard,
                                adaptive_ram_growth_context.with_sparse_experts_are_paged(
                                    model.sparse_experts_are_paged(),
                                ),
                                true,
                                model,
                                active_memory_bytes_before_growth,
                                &mut active_request.performance_attribution,
                            )?;
                            return Err(target_verification_error);
                        }
                    }
                }
                None => MtpPrefixAcceptanceOutcome::OperationalFallback,
            };
            record_mtp_outcome(active_request, mtp_prefix_acceptance_outcome);
            if mtp_prefix_acceptance_outcome != MtpPrefixAcceptanceOutcome::OperationalFallback {
                let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
                    &mut self.adaptive_ram_growth_guard,
                    adaptive_ram_growth_context
                        .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                    true,
                    model,
                    active_memory_bytes_before_growth,
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
            record_completed_adaptive_ram_growth(
                &mut self.adaptive_ram_growth_guard,
                adaptive_ram_growth_context
                    .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                true,
                model,
                active_memory_bytes_before_growth,
                &mut active_request.performance_attribution,
            )?;
        }

        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
            1,
            active_request.mtp_request_state.is_some(),
            model.sparse_experts_are_paged(),
        );
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            0,
        )?;
        let next_generated_token = if active_request.mtp_request_state.is_some() {
            let target_forward_output = model
                .generated_token_forward_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    &current_generated_token,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            let next_generated_token = active_request
                .build_generated_token(model, target_forward_output.final_logits())?;
            if active_request.mtp_request_state.is_some() {
                active_request.mtp_target_hidden_states =
                    Some(target_forward_output.into_pre_final_normalization_hidden_states());
            }
            next_generated_token
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
            active_request.build_generated_token(model, &next_logits)?
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
