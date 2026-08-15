use astronomical_ipc_protocol::RequestId;

use crate::{AdaptiveRamGrowthContext, InferenceEngineError, PerformanceOperation};

use super::super::model::memory_admission::invalid_request_error;
use super::completed_forward_memory::record_completed_adaptive_ram_growth;
use super::{Qwen3_5EngineState, fatal_engine_error};
use crate::qwen3_5::multi_token_prediction::{
    disable_prediction_after_optional_injection_failure, forward_final_injected_prediction_token,
    projected_injected_prediction_growth_bytes, reseed_prediction_after_injected_prefix,
    reset_prediction_after_injection, restore_queued_prediction_prefix_before_injection,
};

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
        restore_queued_prediction_prefix_before_injection(active_request)?;
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

        // Injection extends the same live context as ordinary request admission.
        // Re-run binary residency admission before mutating decoder state so a
        // rejection leaves the continuation frontier unchanged.
        let additional_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        let target_expert_payload_bytes_reclaimed_during_injection = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                projected_context_tokens,
                0,
                additional_maximum_expert_page_reservation_bytes,
                &mut active_request.performance_attribution,
            )?;
        if target_expert_payload_bytes_reclaimed_during_injection > 0 {
            tracing::info!(
                request_id = active_request.request_id.value(),
                target_expert_payload_bytes_reclaimed_during_injection,
                "admitted injected model feedback after reclaiming target experts"
            );
        }

        // External feedback changes the continuation frontier. Discard both the
        // one-token-ahead successor and its rollback verdict before forwarding
        // feedback; neither belongs to the newly injected token sequence.
        active_request.pending_generated_token = None;
        let mut should_reseed_prediction_after_injection = reset_prediction_after_injection(
            active_request,
            self.full_attention_kv_state_growth_tokens,
        )?;
        let final_input_token_position = input_token_ids.len() - 1;
        if final_input_token_position > 0 {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let feedback_prefix_token_ids = &input_token_ids[..final_input_token_position];
            let additional_persistent_state_growth_bytes =
                projected_injected_prediction_growth_bytes(
                    model,
                    active_request,
                    feedback_prefix_token_ids.len(),
                )?;
            let injected_prefill_execution_context =
                super::prefill_execution_context::Qwen3_5PrefillExecutionContext::new(
                    false,
                    active_request.has_optional_prediction_session(),
                    model.sparse_experts_are_paged(),
                    self.persistent_prompt_cache.is_some()
                        && active_request.can_use_persistent_prompt_cache
                        && !active_request.has_optional_prediction_session(),
                );
            let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
                feedback_prefix_token_ids.len(),
                self.prompt_processing_chunk_sizer
                    .exact_measurement_context_identifier(
                        active_request.next_position_tokens as usize,
                        injected_prefill_execution_context,
                    ),
                false,
                active_request.has_optional_prediction_session(),
                model.sparse_experts_are_paged(),
            );
            let (active_memory_bytes_before_growth, retained_expert_payload_bytes_before_growth) =
                self.measure_adaptive_ram_growth_memory_admission(
                    adaptive_ram_growth_context,
                    &mut active_request.performance_attribution,
                    &active_request.request_decoder_state,
                    additional_persistent_state_growth_bytes,
                    0,
                )?;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            if should_reseed_prediction_after_injection {
                let shifted_feedback_token_ids = &input_token_ids[1..];
                if let Err(prediction_history_error) = reseed_prediction_after_injected_prefix(
                    model,
                    active_request,
                    feedback_prefix_token_ids,
                    shifted_feedback_token_ids,
                ) {
                    tracing::warn!(
                        request_id = active_request.request_id.value(),
                        error = %prediction_history_error,
                        "optional prediction feedback-history prefill failed; continuing target-only"
                    );
                    disable_prediction_after_optional_injection_failure(active_request);
                    should_reseed_prediction_after_injection = false;
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
                retained_expert_payload_bytes_before_growth,
                0,
                &mut active_request.performance_attribution,
            )?;
        }
        let final_input_token_id = input_token_ids[final_input_token_position];
        let sparse_experts_are_paged = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .sparse_experts_are_paged();
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
            1,
            should_reseed_prediction_after_injection,
            sparse_experts_are_paged,
        );
        let (active_memory_bytes_before_growth, retained_expert_payload_bytes_before_growth) = self
            .measure_adaptive_ram_growth_memory_admission(
                adaptive_ram_growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                0,
                0,
            )?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let next_generated_token = if should_reseed_prediction_after_injection {
            forward_final_injected_prediction_token(model, active_request, final_input_token_id)?
                .ok_or_else(|| {
                    fatal_engine_error("prediction request disappeared before injected final token")
                })?
        } else {
            let feedback_logits = model
                .build_forward_chunk_with_performance_attribution(
                    &[final_input_token_id],
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            active_request.build_generated_token(model, &feedback_logits)?
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
        record_completed_adaptive_ram_growth(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            true,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            &mut active_request.performance_attribution,
        )?;
        active_request.pending_generated_token = Some(next_generated_token);
        Ok(())
    }
}
