use astronomical_ipc_protocol::RequestId;

use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, PerformanceCounter, PerformanceOperation,
};

use super::super::model::memory_admission::{
    invalid_request_error, validate_context_memory_admission,
};
use super::memory_admission::record_completed_adaptive_ram_growth;
use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    pub(super) fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        if input_token_ids.is_empty() {
            return Ok(());
        }
        if input_token_ids
            .iter()
            .any(|token_id| *token_id >= self.vocabulary_size)
        {
            return Err(fatal_engine_error(
                "injected model feedback contains a token outside the certified vocabulary",
            ));
        }

        let mut active_request = self.active_request.take().ok_or_else(|| {
            fatal_engine_error(
                "Qwen3.5 model feedback injection requested without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(fatal_engine_error(
                "Qwen3.5 model feedback injection request correlation mismatch",
            ));
        }

        match self.inject_into_active_request(&mut active_request, &input_token_ids) {
            Ok(()) => {
                self.active_request = Some(active_request);
                Ok(())
            }
            Err(generation_error) => {
                self.finalize_generation_request_after_error(
                    active_request,
                    &generation_error,
                    "model feedback injection rejected",
                    "model feedback injection failed",
                );
                Err(generation_error)
            }
        }
    }

    fn inject_into_active_request(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        input_token_ids: &[u32],
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        if !active_request.verified_mtp_generated_token_ids.is_empty() {
            let accepted_mtp_draft_rollback = active_request
                .accepted_mtp_draft_rollback
                .take()
                .ok_or_else(|| {
                    fatal_engine_error("queued MTP draft lost its target rollback checkpoint")
                })?;
            active_request
                .request_decoder_state
                .restore_mtp_verified_prefix(
                    accepted_mtp_draft_rollback.verified_prefix_position_tokens,
                    accepted_mtp_draft_rollback.verified_prefix_boundary_checkpoint,
                )
                .map_err(super::qwen3_5_runtime_error)?;
            active_request.next_position_tokens =
                accepted_mtp_draft_rollback.verified_prefix_position_tokens;
        }
        let remaining_output_tokens = active_request
            .maximum_output_tokens
            .saturating_sub(active_request.generated_token_count)
            as usize;
        let projected_context_tokens = (active_request.next_position_tokens as usize)
            .checked_add(input_token_ids.len())
            .and_then(|context_tokens| context_tokens.checked_add(remaining_output_tokens))
            .ok_or_else(|| invalid_request_error("generation context token count overflowed"))?;
        if projected_context_tokens > self.maximum_position_count {
            return Err(invalid_request_error(
                "generation context exceeds the certified maximum position count",
            ));
        }

        validate_context_memory_admission(
            model,
            self.memory_limits,
            self.context_memory_reservation_bytes_per_token,
            projected_context_tokens,
            0,
        )?;

        active_request.pending_generated_token = None;
        active_request.verified_mtp_generated_token_ids.clear();
        active_request.mtp_target_hidden_states = None;
        let mut should_reseed_mtp_after_injection = active_request.mtp_request_state.is_some();
        if let Some(mtp_request_state) = active_request.mtp_request_state.as_mut() {
            mtp_request_state
                .reset_with_growth_tokens(self.full_attention_kv_state_growth_tokens)
                .map_err(super::qwen3_5_runtime_error)?;
        }
        let final_input_token_position = input_token_ids.len() - 1;
        if final_input_token_position > 0 {
            let feedback_prefix_token_ids = &input_token_ids[..final_input_token_position];
            let mtp_full_attention_growth_bytes = match active_request.mtp_request_state.as_ref() {
                Some(mtp_request_state) => {
                    let mtp_full_attention_bytes_per_layer_token = model
                        .config()
                        .full_attention_key_value_state_bytes_per_layer_token()
                        .ok_or_else(|| {
                            fatal_engine_error(
                                "MTP full-attention bytes per layer token overflowed",
                            )
                        })?;
                    mtp_request_state
                        .projected_capacity_growth_bytes(
                            mtp_full_attention_bytes_per_layer_token,
                            feedback_prefix_token_ids.len(),
                        )
                        .map_err(super::qwen3_5_runtime_error)?
                }
                None => 0,
            };
            let injected_prefill_execution_context =
                super::prefill_execution_context::Qwen3_5PrefillExecutionContext::new(
                    false,
                    active_request.mtp_request_state.is_some(),
                    model.sparse_experts_are_paged(),
                    self.persistent_prompt_cache.is_some()
                        && active_request.can_use_persistent_prompt_cache
                        && !active_request.persistent_prompt_cache_capture_has_stopped
                        && active_request.mtp_request_state.is_none(),
                );
            let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
                feedback_prefix_token_ids.len(),
                self.prefill_chunck_sizer
                    .prompt_processing_context_identifier(
                        active_request.next_position_tokens as usize,
                        injected_prefill_execution_context,
                    ),
                false,
                active_request.mtp_request_state.is_some(),
                model.sparse_experts_are_paged(),
            );
            let active_memory_bytes_before_growth = self
                .measure_adaptive_ram_growth_memory_admission(
                    adaptive_ram_growth_context,
                    &mut active_request.performance_attribution,
                    &active_request.request_decoder_state,
                    mtp_full_attention_growth_bytes,
                    0,
                )?;
            if should_reseed_mtp_after_injection {
                let target_prefill_output = model
                    .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                        feedback_prefix_token_ids,
                        active_request.next_position_tokens,
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map_err(InferenceEngineError::from)?;
                let shifted_feedback_token_ids = &input_token_ids[1..];
                if let Err(mtp_prefill_error) = model
                    .prefill_mtp_history_from_token_ids_with_performance_attribution(
                        target_prefill_output.pre_final_normalization_hidden_states(),
                        shifted_feedback_token_ids,
                        active_request
                            .mtp_request_state
                            .as_mut()
                            .ok_or_else(|| fatal_engine_error("MTP request state disappeared"))?,
                        &mut active_request.performance_attribution,
                    )
                {
                    tracing::warn!(
                        request_id = active_request.request_id.value(),
                        error = %mtp_prefill_error,
                        "MTP feedback-history prefill failed; continuing target-only"
                    );
                    active_request.mtp_request_state = None;
                    should_reseed_mtp_after_injection = false;
                }
            } else {
                model
                    .prefill_chunck_with_performance_attribution(
                        feedback_prefix_token_ids,
                        active_request.next_position_tokens,
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map_err(InferenceEngineError::from)?;
            }
            active_request.advance_position(feedback_prefix_token_ids.len())?;
            record_completed_adaptive_ram_growth(
                &mut self.adaptive_ram_growth_guard,
                adaptive_ram_growth_context
                    .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                false,
                model,
                active_memory_bytes_before_growth,
                0,
                &mut active_request.performance_attribution,
            )?;
        }
        let final_input_token_id = input_token_ids[final_input_token_position];
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
            1,
            should_reseed_mtp_after_injection,
            model.sparse_experts_are_paged(),
        );
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            0,
            0,
        )?;
        let (feedback_logits, mtp_target_hidden_states) = if should_reseed_mtp_after_injection {
            let target_forward_output = model
                .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    &[final_input_token_id],
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            (
                target_forward_output
                    .final_logits()
                    .retain()
                    .map_err(super::qwen3_5_runtime_error)?,
                Some(target_forward_output.into_pre_final_normalization_hidden_states()),
            )
        } else {
            (
                model
                    .build_forward_chunk_with_performance_attribution(
                        &[final_input_token_id],
                        active_request.next_position_tokens,
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map_err(InferenceEngineError::from)?,
                None,
            )
        };
        active_request.advance_position(1)?;
        let next_generated_token = active_request.build_generated_token(model, &feedback_logits)?;
        active_request.mtp_target_hidden_states = mtp_target_hidden_states;
        if should_reseed_mtp_after_injection {
            active_request
                .performance_attribution
                .record_counter(PerformanceCounter::MtpFeedbackHistoryReseedCount, 1);
        }
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
        active_request.pending_generated_token = Some(next_generated_token);
        Ok(())
    }
}
