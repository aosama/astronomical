use ipc_protocol::RequestId;

use crate::{InferenceEngineError, PerformanceOperation};

use super::Qwen3_5EngineState;
use super::completed_forward_memory::{self, MeasuredForward};
use crate::qwen3_5_streaming::model::request_refusal;

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
        let projected_context_tokens =
            project_injection_context_token_count(InjectionContextTokenProjection {
                next_position_tokens: active_request.next_position_tokens as usize,
                injected_input_token_count: input_token_ids.len(),
                remaining_output_tokens: (active_request
                    .maximum_output_tokens
                    .saturating_sub(active_request.generated_token_count))
                    as usize,
                hard_maximum_position_count: self.hard_maximum_position_count,
                advertised_maximum_position_count: self.maximum_position_count,
            })?;
        self.make_room_for_injected_context(projected_context_tokens)?;

        // External feedback changes the continuation frontier. Discard both the
        // one-token-ahead successor and its rollback verdict before forwarding
        // feedback; neither belongs to the newly injected token sequence.
        active_request.pending_generated_token = None;
        let final_input_token_position = input_token_ids.len() - 1;
        if final_input_token_position > 0 {
            self.prefill_injected_feedback_prefix(
                active_request,
                input_token_ids,
                final_input_token_position,
            )?;
        }
        self.decode_injected_final_token(
            active_request,
            input_token_ids,
            final_input_token_position,
        )
    }

    /// Injection extends the request's stored state between measured
    /// steps. The memory governor grows the context account to the
    /// projected extent first, so the kept weights have made room before
    /// the injected tokens run.
    fn make_room_for_injected_context(
        &self,
        projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let projected_context_bytes = projected_context_tokens
            .saturating_mul(self.context_memory_reservation_bytes_per_token);
        model.govern_context_extent(u64::try_from(projected_context_bytes).unwrap_or(u64::MAX));
        Ok(())
    }

    fn prefill_injected_feedback_prefix(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        input_token_ids: &[u32],
        final_input_token_position: usize,
    ) -> Result<(), InferenceEngineError> {
        let feedback_prefix_token_ids = &input_token_ids[..final_input_token_position];
        let memory_baseline = self.begin_forward_memory_measurement()?;
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
        completed_forward_memory::measure_completed_forward(
            model,
            MeasuredForward::prompt(feedback_prefix_token_ids.len()),
            active_request.request_decoder_state.payload_byte_count(),
            memory_baseline.active_memory_bytes,
            &mut active_request.performance_attribution,
        )?;
        Ok(())
    }

    fn decode_injected_final_token(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        input_token_ids: &[u32],
        final_input_token_position: usize,
    ) -> Result<(), InferenceEngineError> {
        let final_input_token_id = input_token_ids[final_input_token_position];
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        // The final feedback token runs the lazy forward and is submitted
        // ahead, unlike a reply token. It is still exact: outside optimistic
        // serving every sparse layer either serves from a table that holds
        // all its experts or brings its route to the host and reads what is
        // missing. What it skips is bookkeeping for this one token: the
        // decode-phase bracket of the `astro_expert_cache` and the expert
        // demand and route sample that `forward_resolving_expert_routes`
        // records, which the next forward clears.
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
        completed_forward_memory::measure_completed_forward(
            model,
            MeasuredForward::reply_token(),
            active_request.request_decoder_state.payload_byte_count(),
            memory_baseline.active_memory_bytes,
            &mut active_request.performance_attribution,
        )?;
        active_request.pending_generated_token = Some(next_generated_token);
        Ok(())
    }
}

/// Context arithmetic inputs for projecting whether an injection fits the
/// context windows. The budget fields are pre-converted to `usize` so the
/// checked additions run in one width — this keeps the overflow guard
/// reachable and unit-testable without dragging an engine state into the
/// arithmetic.
struct InjectionContextTokenProjection {
    next_position_tokens: usize,
    injected_input_token_count: usize,
    remaining_output_tokens: usize,
    hard_maximum_position_count: usize,
    advertised_maximum_position_count: usize,
}

fn project_injection_context_token_count(
    projection: InjectionContextTokenProjection,
) -> Result<usize, InferenceEngineError> {
    let projected_context_tokens = projection
        .next_position_tokens
        .checked_add(projection.injected_input_token_count)
        .and_then(|context_tokens| context_tokens.checked_add(projection.remaining_output_tokens))
        .ok_or_else(|| {
            request_refusal::invalid_request_error("generation context token count overflowed")
        })?;
    if projected_context_tokens > projection.hard_maximum_position_count {
        return Err(request_refusal::invalid_request_error(
            "generation context exceeds the model maximum position count",
        ));
    }
    if projected_context_tokens > projection.advertised_maximum_position_count {
        tracing::warn!(
            projected_context_tokens,
            advertised_context_tokens = projection.advertised_maximum_position_count,
            "continuation exceeds the configured context limit; serving anyway because it fits the model artifact context window"
        );
    }
    Ok(projected_context_tokens)
}

#[cfg(test)]
#[path = "inject_input_tokens_tests.rs"]
mod inject_input_tokens_tests;
