//! Request-shape validation performed before generation allocates decoder state.
//!
//! Keeping these boundary checks separate from request-state construction makes
//! `start_generation` read as a lifecycle while this owner explains why malformed
//! or impossible token ranges are rejected before touching MLX state.

use crate::{InferenceEngineError, Qwen3_5InferenceRequest};

use super::super::model::memory_admission::invalid_request_error;
use super::super::text::minimum_bounded_output_token_count;
use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    pub(super) fn validate_generation_request_and_resolve_total_context(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
    ) -> Result<usize, InferenceEngineError> {
        if self.model.is_none() {
            return Err(fatal_engine_error("Qwen3.5 engine is not loaded"));
        }
        if inference_request.input_token_ids().is_empty() {
            return Err(fatal_engine_error("generation prompt must not be empty"));
        }
        if inference_request.max_output_tokens() == 0 {
            return Err(fatal_engine_error(
                "generation output-token budget must be positive",
            ));
        }
        if let Some(thinking_budget) = inference_request.thinking_budget() {
            let forced_transition_token_ids =
                inference_request.forced_thinking_transition_token_ids();
            if forced_transition_token_ids.last().copied() != Some(self.think_end_token_id) {
                return Err(invalid_request_error(
                    "thinking budget requires a model-owned transition ending with the model thinking marker",
                ));
            }
            let minimum_bounded_output_tokens = minimum_bounded_output_token_count(
                thinking_budget,
                forced_transition_token_ids.len(),
            )
            .ok_or_else(|| {
                invalid_request_error("thinking-budget output reservation overflowed")
            })?;
            if usize::from(inference_request.max_output_tokens()) < minimum_bounded_output_tokens {
                return Err(invalid_request_error(format!(
                    "output-token budget must reserve {minimum_bounded_output_tokens} positions for the thinking allowance, forced transition, and final answer"
                )));
            }
        }
        let ordinary_target_prefill_control_span_token_count =
            inference_request.ordinary_target_prefill_control_span_token_count();
        if ordinary_target_prefill_control_span_token_count
            > inference_request.input_token_ids().len().saturating_sub(1)
        {
            return Err(invalid_request_error(
                "system-and-tool control span reaches beyond selectable prompt content",
            ));
        }
        if inference_request
            .input_token_ids()
            .iter()
            .any(|token_id| *token_id >= self.vocabulary_size)
        {
            return Err(fatal_engine_error(
                "generation prompt contains a token outside the model vocabulary",
            ));
        }
        if inference_request
            .forced_thinking_transition_token_ids()
            .iter()
            .chain(inference_request.natural_reasoning_end_token_ids())
            .any(|token_id| *token_id >= self.vocabulary_size)
        {
            return Err(invalid_request_error(
                "thinking-budget configuration contains a token outside the model vocabulary",
            ));
        }
        let prompt_token_count = inference_request.input_token_ids().len();
        let maximum_output_token_count = usize::from(inference_request.max_output_tokens());
        let thinking_budget_token_count = inference_request.thinking_budget().map(usize::from);
        let total_context_tokens = prompt_token_count
            .checked_add(maximum_output_token_count)
            .ok_or_else(|| invalid_request_error("generation context token count overflowed"))?;
        if total_context_tokens > self.maximum_position_count {
            return Err(invalid_request_error(
                "generation context exceeds the model maximum position count",
            ));
        }
        tracing::info!(
            prompt_token_count,
            maximum_output_token_count,
            thinking_budget_token_count,
            total_context_token_count = total_context_tokens,
            maximum_position_count = self.maximum_position_count,
            "resolved generation context token reservation"
        );
        Ok(total_context_tokens)
    }
}
