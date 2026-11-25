//! Request-shape validation performed before generation allocates decoder state.
//!
//! Keeping these boundary checks separate from request-state construction makes
//! `start_generation` read as a lifecycle while this owner explains why malformed
//! or impossible token ranges are rejected before touching MLX state.

use crate::engine_errors::{
    fatal_engine_error, generation_context_token_count_overflow_error, invalid_request_error,
    thinking_budget_output_reservation_overflow_error,
};
use crate::request::minimum_bounded_output_token_count;
use crate::{InferenceEngineError, TextInferenceRequest, bounded_output_reservation_token_count};

use super::generation_thinking_end_markers::forced_thinking_transition_ends_with_model_marker;
use super::generation_vocabulary_boundaries::contains_token_id_outside_vocabulary;

/// The loaded model's limits a request is checked against.
pub struct TextGenerationRequestLimits {
    pub is_model_loaded: bool,
    /// Token ID that closes the thinking block.
    pub think_end_token_id: u32,
    pub vocabulary_size: u32,
    /// The configured context limit, advisory: crossing it is logged.
    pub maximum_position_count: usize,
    /// The model artifact's native position range: the only hard boundary.
    pub hard_maximum_position_count: usize,
}

/// Validates one request and returns the context it reserves: the prompt plus
/// the thinking allowance and a visible-answer reserve.
pub fn validate_generation_request_and_resolve_reserved_context(
    request_limits: &TextGenerationRequestLimits,
    inference_request: &TextInferenceRequest,
) -> Result<usize, InferenceEngineError> {
    if !request_limits.is_model_loaded {
        return Err(fatal_engine_error("engine is not loaded"));
    }
    if inference_request.input_token_ids().is_empty() {
        return Err(fatal_engine_error("generation prompt must not be empty"));
    }
    if inference_request.max_output_tokens() == 0 {
        return Err(fatal_engine_error(
            "generation output-token budget must be positive",
        ));
    }
    validate_thinking_budget(request_limits, inference_request)?;
    validate_vocabulary(request_limits, inference_request)?;
    resolve_reserved_context(request_limits, inference_request)
}

fn validate_thinking_budget(
    request_limits: &TextGenerationRequestLimits,
    inference_request: &TextInferenceRequest,
) -> Result<(), InferenceEngineError> {
    let Some(thinking_budget) = inference_request.thinking_budget() else {
        return Ok(());
    };
    let forced_transition_token_ids = inference_request.forced_thinking_transition_token_ids();
    if !forced_thinking_transition_ends_with_model_marker(
        forced_transition_token_ids,
        request_limits.think_end_token_id,
    ) {
        return Err(invalid_request_error(
            "thinking budget requires a model-owned transition ending with the model thinking marker",
        ));
    }
    let minimum_bounded_output_tokens =
        minimum_bounded_output_token_count(thinking_budget, forced_transition_token_ids.len())
            .ok_or_else(thinking_budget_output_reservation_overflow_error)?;
    if usize::from(inference_request.max_output_tokens()) < minimum_bounded_output_tokens {
        return Err(invalid_request_error(format!(
            "output-token budget must reserve {minimum_bounded_output_tokens} positions for the thinking allowance, forced transition, and final answer"
        )));
    }
    Ok(())
}

fn validate_vocabulary(
    request_limits: &TextGenerationRequestLimits,
    inference_request: &TextInferenceRequest,
) -> Result<(), InferenceEngineError> {
    if contains_token_id_outside_vocabulary(
        inference_request.input_token_ids(),
        request_limits.vocabulary_size,
    ) {
        return Err(fatal_engine_error(
            "generation prompt contains a token outside the model vocabulary",
        ));
    }
    if contains_token_id_outside_vocabulary(
        inference_request.forced_thinking_transition_token_ids(),
        request_limits.vocabulary_size,
    ) || contains_token_id_outside_vocabulary(
        inference_request.natural_reasoning_end_token_ids(),
        request_limits.vocabulary_size,
    ) {
        return Err(invalid_request_error(
            "thinking-budget configuration contains a token outside the model vocabulary",
        ));
    }
    Ok(())
}

fn resolve_reserved_context(
    request_limits: &TextGenerationRequestLimits,
    inference_request: &TextInferenceRequest,
) -> Result<usize, InferenceEngineError> {
    let prompt_token_count = inference_request.input_token_ids().len();
    let maximum_output_token_count = usize::from(inference_request.max_output_tokens());
    let thinking_budget_token_count = inference_request.thinking_budget().map(usize::from);
    let total_context_tokens = prompt_token_count
        .checked_add(maximum_output_token_count)
        .ok_or_else(generation_context_token_count_overflow_error)?;
    if total_context_tokens > request_limits.hard_maximum_position_count {
        return Err(invalid_request_error(
            "generation context exceeds the model maximum position count",
        ));
    }
    if total_context_tokens > request_limits.maximum_position_count {
        tracing::warn!(
            total_context_token_count = total_context_tokens,
            advertised_context_tokens = request_limits.maximum_position_count,
            "request exceeds the configured context limit; serving anyway because it fits the model artifact context window"
        );
    }
    // The memory reservation covers the thinking allowance plus a visible-answer
    // reserve, never the caller's full output budget: a 65,535-token budget with a
    // 16,384-token thinking allowance pinned ~1.3 GB of KV that never materialized
    // and forced a per-turn warm-expert reclaim in long agent conversations
    // (2026-10-16 incident). Output beyond the reserve grows as it is written,
    // under each engine's per-step memory check. Position limits above stay on
    // the true total.
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
        maximum_position_count = request_limits.maximum_position_count,
        "resolved generation context token reservation"
    );
    Ok(reserved_context_tokens)
}

#[cfg(test)]
#[path = "generation_request_validation_tests.rs"]
mod generation_request_validation_tests;
