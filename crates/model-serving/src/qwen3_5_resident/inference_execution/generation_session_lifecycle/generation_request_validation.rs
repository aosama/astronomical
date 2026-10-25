//! Request-shape validation performed before generation allocates decoder state.
//!
//! Keeping these boundary checks separate from request-state construction makes
//! `start_generation` read as a lifecycle while this owner explains why malformed
//! or impossible token ranges are rejected before touching MLX state.

use crate::{
    InferenceEngineError, Qwen3_5InferenceRequest, bounded_output_reservation_token_count,
};

use super::Qwen3_5EngineState;
use crate::qwen3_5_core::text;
use crate::qwen3_5_resident::model::memory_admission;

impl Qwen3_5EngineState {
    pub(super) fn validate_generation_request_and_resolve_reserved_context(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
    ) -> Result<usize, InferenceEngineError> {
        if self.model.is_none() {
            return Err(super::fatal_engine_error("Qwen3.5 engine is not loaded"));
        }
        if inference_request.input_token_ids().is_empty() {
            return Err(super::fatal_engine_error(
                "generation prompt must not be empty",
            ));
        }
        if inference_request.max_output_tokens() == 0 {
            return Err(super::fatal_engine_error(
                "generation output-token budget must be positive",
            ));
        }
        if let Some(thinking_budget) = inference_request.thinking_budget() {
            let forced_transition_token_ids =
                inference_request.forced_thinking_transition_token_ids();
            if !super::generation_thinking_end_markers::forced_thinking_transition_ends_with_model_marker(
                forced_transition_token_ids,
                self.think_end_token_id,
            ) {
                return Err(memory_admission::invalid_request_error(
                    "thinking budget requires a model-owned transition ending with the model thinking marker",
                ));
            }
            let minimum_bounded_output_tokens = text::minimum_bounded_output_token_count(
                thinking_budget,
                forced_transition_token_ids.len(),
            )
            .ok_or_else(
                super::generation_error_constructors::thinking_budget_output_reservation_overflow_error,
            )?;
            if usize::from(inference_request.max_output_tokens()) < minimum_bounded_output_tokens {
                return Err(memory_admission::invalid_request_error(format!(
                    "output-token budget must reserve {minimum_bounded_output_tokens} positions for the thinking allowance, forced transition, and final answer"
                )));
            }
        }
        if super::generation_vocabulary_boundaries::contains_token_id_outside_vocabulary(
            inference_request.input_token_ids(),
            self.vocabulary_size,
        ) {
            return Err(super::fatal_engine_error(
                "generation prompt contains a token outside the model vocabulary",
            ));
        }
        if super::generation_vocabulary_boundaries::contains_token_id_outside_vocabulary(
            inference_request.forced_thinking_transition_token_ids(),
            self.vocabulary_size,
        ) || super::generation_vocabulary_boundaries::contains_token_id_outside_vocabulary(
            inference_request.natural_reasoning_end_token_ids(),
            self.vocabulary_size,
        ) {
            return Err(memory_admission::invalid_request_error(
                "thinking-budget configuration contains a token outside the model vocabulary",
            ));
        }
        let prompt_token_count = inference_request.input_token_ids().len();
        let maximum_output_token_count = usize::from(inference_request.max_output_tokens());
        let thinking_budget_token_count = inference_request.thinking_budget().map(usize::from);
        let total_context_tokens = prompt_token_count
            .checked_add(maximum_output_token_count)
            .ok_or_else(
                super::generation_error_constructors::generation_context_token_count_overflow_error,
            )?;
        if total_context_tokens > self.hard_maximum_position_count {
            return Err(memory_admission::invalid_request_error(
                "generation context exceeds the model maximum position count",
            ));
        }
        if total_context_tokens > self.maximum_position_count {
            tracing::warn!(
                total_context_token_count = total_context_tokens,
                advertised_context_tokens = self.maximum_position_count,
                "request exceeds the configured context limit; serving anyway because it fits the model artifact context window"
            );
        }
        // The memory reservation covers the thinking allowance plus a visible-answer
        // reserve, never the caller's full output budget: a 65,535-token budget with a
        // 16,384-token thinking allowance pinned ~1.3 GB of KV that never materialized
        // and forced a per-turn warm-expert reclaim in long agent conversations
        // (2026-10-16 incident). Beyond-reserve output grows lazily under the
        // per-step growth guard. Position limits above stay on the true total.
        let output_reservation_token_count = bounded_output_reservation_token_count(
            maximum_output_token_count,
            thinking_budget_token_count,
        );
        let reserved_context_tokens = prompt_token_count + output_reservation_token_count;
        tracing::info!(
            prompt_token_count,
            maximum_output_token_count,
            thinking_budget_token_count,
            total_context_token_count = total_context_tokens,
            reserved_context_token_count = reserved_context_tokens,
            maximum_position_count = self.maximum_position_count,
            "resolved generation context token reservation"
        );
        Ok(reserved_context_tokens)
    }
}
