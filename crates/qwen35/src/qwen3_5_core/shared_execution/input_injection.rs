//! Model-visible feedback injected into an active Qwen3.5 request, for either
//! engine: the feedback's prefix is read as one prompt chunk and its final
//! token is forwarded to sample the next reply token.
//!
//! Each forward is governed through [`Qwen3_5ForwardMemoryGovernance`]. Before
//! the feedback runs, the engine prepares through
//! [`Qwen3_5InputInjectionHooks`]: the resident engine checks the projected
//! context fits and suspends drafting, the streaming engine grows its
//! governed context to make room.

use ipc_protocol::RequestId;

use crate::qwen3_5_core::engine_errors::{
    fatal_engine_error, generation_context_token_count_overflow_error, invalid_request_error,
};
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::{InferenceEngineError, PerformanceOperation};

use super::forward_memory_governance::{Qwen3_5ForwardKind, Qwen3_5ForwardMemoryGovernance};
use super::generation_finalization_steps::{
    Qwen3_5FinalizationHooks, finalize_generation_request_after_error,
};
use super::text_model::Qwen3_5TextModel;

pub trait Qwen3_5InputInjectionHooks:
    Qwen3_5ForwardMemoryGovernance + Qwen3_5FinalizationHooks
{
    /// Prepares the request for feedback that grows its context to
    /// `projected_context_tokens`, refusing feedback the engine cannot hold.
    fn prepare_for_injected_input(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
        projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError>;
}

/// Injects model-visible feedback into the active request, finalizing the
/// request when the injection fails.
pub fn inject_input_tokens<Engine: Qwen3_5InputInjectionHooks>(
    engine: &mut Engine,
    request_id: RequestId,
    input_token_ids: Vec<u32>,
) -> Result<(), InferenceEngineError> {
    if input_token_ids.is_empty() {
        return Ok(());
    }
    if input_token_ids
        .iter()
        .any(|token_id| *token_id >= engine.execution_core().vocabulary_size)
    {
        return Err(fatal_engine_error(
            "injected model feedback contains a token outside the model vocabulary",
        ));
    }
    let mut active_request = engine
        .execution_core_mut()
        .active_request
        .take()
        .ok_or_else(|| {
            fatal_engine_error(
                "Qwen3.5 model feedback injection requested without an active request",
            )
        })?;
    if active_request.request_id != request_id {
        engine.execution_core_mut().active_request = Some(active_request);
        return Err(fatal_engine_error(
            "Qwen3.5 model feedback injection request correlation mismatch",
        ));
    }
    match inject_into_active_request(engine, &mut active_request, &input_token_ids) {
        Ok(()) => {
            engine.execution_core_mut().active_request = Some(active_request);
            Ok(())
        }
        Err(generation_error) => {
            finalize_generation_request_after_error(
                engine,
                active_request,
                &generation_error,
                "model feedback injection rejected",
                "model feedback injection failed",
            );
            Err(generation_error)
        }
    }
}

fn inject_into_active_request<Engine: Qwen3_5InputInjectionHooks>(
    engine: &mut Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    input_token_ids: &[u32],
) -> Result<(), InferenceEngineError> {
    let execution_core = engine.execution_core();
    let projected_context_tokens =
        project_injection_context_token_count(InjectionContextTokenProjection {
            next_position_tokens: active_request.next_position_tokens as usize,
            injected_input_token_count: input_token_ids.len(),
            remaining_output_tokens: (active_request
                .maximum_output_tokens
                .saturating_sub(active_request.generated_token_count))
                as usize,
            hard_maximum_position_count: execution_core.hard_maximum_position_count,
            advertised_maximum_position_count: execution_core.maximum_position_count,
        })?;
    engine.prepare_for_injected_input(active_request, projected_context_tokens)?;
    // External feedback changes the continuation frontier. Discard the
    // one-token-ahead successor before forwarding feedback; it does not
    // belong to the newly injected token sequence.
    active_request.pending_generated_token = None;
    let final_input_token_position = input_token_ids.len() - 1;
    if final_input_token_position > 0 {
        prefill_injected_feedback_prefix(
            engine,
            active_request,
            &input_token_ids[..final_input_token_position],
        )?;
    }
    decode_injected_final_token(
        engine,
        active_request,
        input_token_ids[final_input_token_position],
    )
}

fn prefill_injected_feedback_prefix<Engine: Qwen3_5InputInjectionHooks>(
    engine: &mut Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    feedback_prefix_token_ids: &[u32],
) -> Result<(), InferenceEngineError> {
    let forward_in_progress = engine.begin_governed_forward(
        Qwen3_5ForwardKind::PromptChunk {
            token_count: feedback_prefix_token_ids.len(),
        },
        &active_request.request_decoder_state,
    )?;
    let model = loaded_model(engine)?;
    model
        .prefill_chunk_with_performance_attribution(
            feedback_prefix_token_ids,
            active_request.next_position_tokens,
            &mut active_request.request_decoder_state,
            &mut active_request.performance_attribution,
        )
        .map_err(InferenceEngineError::from)?;
    active_request.advance_position(feedback_prefix_token_ids.len())?;
    engine.complete_governed_forward(
        forward_in_progress,
        active_request.request_decoder_state.payload_byte_count(),
        &mut active_request.performance_attribution,
    )?;
    Ok(())
}

/// Forwards the final feedback token and keeps the sampled reply token as
/// the pending successor.
///
/// The final feedback token runs the lazy forward and is submitted ahead,
/// unlike a streaming reply token. It is still exact on the streaming engine:
/// outside optimistic serving every sparse layer either serves from a table
/// that holds all its experts or brings its route to the host and reads what
/// is missing. What it skips there is bookkeeping for this one token: the
/// decode-phase bracket of the `astro_expert_cache` and the expert demand and
/// route sample that `forward_resolving_expert_routes` records, which the next
/// forward clears.
fn decode_injected_final_token<Engine: Qwen3_5InputInjectionHooks>(
    engine: &mut Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    final_input_token_id: u32,
) -> Result<(), InferenceEngineError> {
    let forward_in_progress = engine.begin_governed_forward(
        Qwen3_5ForwardKind::ReplyToken,
        &active_request.request_decoder_state,
    )?;
    let model = loaded_model(engine)?;
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
    engine.complete_governed_forward(
        forward_in_progress,
        active_request.request_decoder_state.payload_byte_count(),
        &mut active_request.performance_attribution,
    )?;
    active_request.pending_generated_token = Some(next_generated_token);
    Ok(())
}

fn loaded_model<Engine: Qwen3_5InputInjectionHooks>(
    engine: &Engine,
) -> Result<&Engine::Model, InferenceEngineError> {
    engine
        .execution_core()
        .model
        .as_ref()
        .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))
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
        .ok_or_else(generation_context_token_count_overflow_error)?;
    if projected_context_tokens > projection.hard_maximum_position_count {
        return Err(invalid_request_error(
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
#[path = "input_injection_tests.rs"]
mod input_injection_tests;
