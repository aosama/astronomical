use astronomical_ipc_protocol::RequestId;

use crate::{AdaptiveRamGrowthContext, InferenceEngineError, PerformanceOperation};

use super::super::model::memory_admission;
use super::Qwen3_5EngineState;
use super::completed_forward_memory;

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
            return Err(super::fatal_engine_error(
                "injected model feedback contains a token outside the model vocabulary",
            ));
        }

        let mut active_request = self.active_request.take().ok_or_else(|| {
            super::fatal_engine_error(
                "Qwen3.5 model feedback injection requested without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(super::fatal_engine_error(
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
        let remaining_output_tokens = active_request
            .maximum_output_tokens
            .saturating_sub(active_request.generated_token_count)
            as usize;
        let projected_context_tokens = (active_request.next_position_tokens as usize)
            .checked_add(input_token_ids.len())
            .and_then(|context_tokens| context_tokens.checked_add(remaining_output_tokens))
            .ok_or_else(|| {
                memory_admission::invalid_request_error("generation context token count overflowed")
            })?;
        if projected_context_tokens > self.hard_maximum_position_count {
            return Err(memory_admission::invalid_request_error(
                "generation context exceeds the model maximum position count",
            ));
        }
        if projected_context_tokens > self.maximum_position_count {
            tracing::warn!(
                projected_context_tokens,
                advertised_context_tokens = self.maximum_position_count,
                "continuation exceeds the configured context limit; serving anyway because it fits the model artifact context window"
            );
        }

        // Injection extends the same live context as ordinary request admission.
        // Re-run binary residency admission before mutating decoder state so a
        // rejection leaves the continuation frontier unchanged.
        let target_expert_payload_bytes_reclaimed_during_injection = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                projected_context_tokens,
                0,
                0,
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
        let final_input_token_position = input_token_ids.len() - 1;
        if final_input_token_position > 0 {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let feedback_prefix_token_ids = &input_token_ids[..final_input_token_position];
            let injected_prefill_execution_context =
                super::prefill_execution_context::Qwen3_5PrefillExecutionContext::new(
                    false,
                    model.sparse_experts_are_paged(),
                    self.persistent_prompt_cache.is_some()
                        && active_request.can_use_persistent_prompt_cache,
                );
            let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
                feedback_prefix_token_ids.len(),
                injected_prefill_execution_context.context_identifier_flags(),
                false,
                model.sparse_experts_are_paged(),
            );
            let admitted_baseline = self.measure_adaptive_ram_growth_memory_admission(
                adaptive_ram_growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                0,
                0,
            )?;
            let active_memory_bytes_before_growth = admitted_baseline.active_memory_bytes;
            let retained_expert_payload_bytes_before_growth =
                admitted_baseline.retained_expert_payload_bytes;
            let streamed_expert_page_bytes_before_growth =
                admitted_baseline.streamed_expert_page_bytes;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            model
                .prefill_chunk_with_performance_attribution(
                    feedback_prefix_token_ids,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(feedback_prefix_token_ids.len())?;
            completed_forward_memory::record_completed_adaptive_ram_growth(
                &mut self.adaptive_ram_growth_guard,
                adaptive_ram_growth_context
                    .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                false,
                model,
                active_memory_bytes_before_growth,
                retained_expert_payload_bytes_before_growth,
                0,
                streamed_expert_page_bytes_before_growth,
                &mut active_request.performance_attribution,
            )?;
        }
        let final_input_token_id = input_token_ids[final_input_token_position];
        let sparse_experts_are_paged = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .sparse_experts_are_paged();
        let adaptive_ram_growth_context =
            AdaptiveRamGrowthContext::decode(1, sparse_experts_are_paged);
        let admitted_baseline = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            0,
            0,
        )?;
        let active_memory_bytes_before_growth = admitted_baseline.active_memory_bytes;
        let retained_expert_payload_bytes_before_growth =
            admitted_baseline.retained_expert_payload_bytes;
        let streamed_expert_page_bytes_before_growth = admitted_baseline.streamed_expert_page_bytes;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let feedback_logits = model
            .build_forward_chunk_with_performance_attribution(
                &[final_input_token_id],
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            )
            .map_err(InferenceEngineError::from)?;
        active_request.advance_position(1)?;
        let next_generated_token = active_request.build_generated_token(model, &feedback_logits)?;
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
        completed_forward_memory::record_completed_adaptive_ram_growth(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            true,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
        active_request.pending_generated_token = Some(next_generated_token);
        Ok(())
    }
}
