//! One decode step for either Qwen3.5 engine: the reply token is chosen
//! (forced thinking transition, the pending successor, or the first token
//! after the prompt), read on the host, checked for the end of the reply,
//! emitted, and its successor's forward is run.
//!
//! This runs once per reply token. It is generic over the engine, so each
//! engine compiles its own copy, exactly as when each carried the code
//! itself, and nothing is added per token. Where the engines differ, a
//! [`Qwen3_5DecodeStepHooks`] method does: how prompt reading and generation
//! preparation proceed, the resident engine's drafting, what the streaming
//! engine records once a token is read, and how the successor's forward runs.

use crate::qwen3_5_core::model::Qwen3_5ServingModel;
use std::time::Instant;

use ipc_protocol::RequestId;
use mlx_bridge::MlxArray;
use runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::{
    GeneratedToken, InferenceEngineError, PerformanceAttributionOutcome, PerformanceCounter,
    PerformanceOperation,
};

use super::decode_step_decisions::pending_successor_reports_zero_first_decode_time;
use super::generated_token_emission::{
    build_generated_token_emission, generated_token_will_be_terminal,
    synchronize_generated_token_id,
};
use super::generation_finalization_steps::{
    Qwen3_5FinalizationHooks, finalize_generation_request_after_error,
};
use super::text_model::Qwen3_5TextModel;

/// What one engine advance produced: a token for a request that continues,
/// or the last token of a request that ends here.
pub enum ActiveRequestAdvance {
    Continue(GeneratedToken),
    Complete(GeneratedToken),
}

/// The points where the engines' decode steps differ.
pub trait Qwen3_5DecodeStepHooks: Qwen3_5FinalizationHooks {
    /// Reads the next stretch of the prompt while any remains; `None` once
    /// the prompt is read.
    fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError>;

    /// Prepares for the reply once the prompt is read; `Some` while an advance
    /// of its own is still owed to the client.
    fn begin_generation_preparation(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError>;

    /// Runs a drafted advance instead of a plain decode step, when the engine
    /// drafts and this step allows it.
    fn advance_drafted_generation_if_admitted(
        &mut self,
        _request_id: RequestId,
        _active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
        _forced_thinking_transition_token_id: Option<u32>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        Ok(None)
    }

    /// Runs once the reply token has been read on the host.
    fn after_reply_token_synchronized(
        &self,
        _active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    /// Runs the final prompt token's forward and samples the first reply
    /// token from it.
    fn decode_first_token_from_final_prompt(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError>;

    /// Runs the forward of the token just read and samples its successor,
    /// returning the successor and the memory reading after the forward.
    fn decode_next_token(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
        current_generated_token: &MlxArray,
        current_generated_token_id: u32,
    ) -> Result<(MlxArray, MlxMemorySnapshot), InferenceEngineError>;
}

/// Advances the active request by one engine step and finalizes it when the
/// step ends the reply or fails.
pub fn advance_generation<Engine: Qwen3_5DecodeStepHooks>(
    engine: &mut Engine,
    request_id: RequestId,
) -> Result<GeneratedToken, InferenceEngineError> {
    let mut active_request = engine
        .execution_core_mut()
        .active_request
        .take()
        .ok_or_else(|| {
            fatal_engine_error("Qwen3.5 generation advance requested without an active request")
        })?;
    if active_request.request_id != request_id {
        engine.execution_core_mut().active_request = Some(active_request);
        return Err(fatal_engine_error(
            "Qwen3.5 generation request correlation mismatch",
        ));
    }
    let advance_span =
        if active_request.prefill_cursor < active_request.input_token_ids.len().saturating_sub(1) {
            PerformanceOperation::PromptPrefillAdvanceSpan
        } else {
            PerformanceOperation::DecodeAdvanceSpan
        };
    let advance_span_started_at = active_request
        .performance_attribution
        .begin_operation_span();
    let active_request_advance = advance_active_request(engine, request_id, &mut active_request);
    active_request
        .performance_attribution
        .complete_operation_span(advance_span, advance_span_started_at);
    match active_request_advance {
        Ok(ActiveRequestAdvance::Continue(generated_token)) => {
            engine.execution_core_mut().active_request = Some(active_request);
            Ok(generated_token)
        }
        Ok(ActiveRequestAdvance::Complete(generated_token)) => {
            let generation_finalization = engine.finalize_generation_request(
                active_request,
                PerformanceAttributionOutcome::Success,
                None,
            );
            Ok(generated_token.with_generation_finalization(generation_finalization))
        }
        Err(generation_error) => {
            finalize_generation_request_after_error(
                engine,
                active_request,
                &generation_error,
                "generation advance rejected",
                "generation advance failed",
            );
            Err(generation_error)
        }
    }
}

/// One step of an active request: the prompt while any remains, then
/// preparation, then a drafted or plain reply token.
fn advance_active_request<Engine: Qwen3_5DecodeStepHooks>(
    engine: &mut Engine,
    request_id: RequestId,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
) -> Result<ActiveRequestAdvance, InferenceEngineError> {
    // Keep model borrows operation-local because the steps below take the
    // engine mutably.
    if let Some(prefill_progress) =
        engine.advance_prompt_prefill_if_pending(request_id, active_request)?
    {
        return Ok(ActiveRequestAdvance::Continue(prefill_progress));
    }
    if let Some(preparation_announcement) =
        engine.begin_generation_preparation(request_id, active_request)?
    {
        return Ok(preparation_announcement);
    }
    let final_prompt_index = active_request.input_token_ids.len() - 1;
    let forced_thinking_transition_token_id =
        active_request.next_forced_thinking_transition_token_id()?;
    if let Some(drafted_advance) = engine.advance_drafted_generation_if_admitted(
        request_id,
        active_request,
        forced_thinking_transition_token_id,
    )? {
        return Ok(drafted_advance);
    }
    let mut first_decode_forward_started_at = None;
    let current_generated_token = current_generated_token_for_decode(
        engine,
        request_id,
        active_request,
        forced_thinking_transition_token_id,
        final_prompt_index,
        &mut first_decode_forward_started_at,
    )?;
    let current_generated_token_id =
        synchronize_generated_token_id(active_request, &current_generated_token)?;
    engine.after_reply_token_synchronized(active_request)?;
    if let Some(first_decode_forward_started_at) = first_decode_forward_started_at {
        active_request.first_decode_forward_elapsed_millis = Some(
            u64::try_from(first_decode_forward_started_at.elapsed().as_millis())
                .unwrap_or(u64::MAX),
        );
    }
    if let Some(completed_advance) =
        complete_generation_if_terminal(engine, active_request, current_generated_token_id)?
    {
        return Ok(completed_advance);
    }
    emit_reply_token_and_decode_successor(
        engine,
        active_request,
        &current_generated_token,
        current_generated_token_id,
    )
}

/// Emits the reply token just read and runs its successor's forward. With
/// structured generation the token is emitted before the forward: the
/// successor is sampled ahead, and a live mask must see the commit first or
/// Juliet/Romeo prefixes stay allowed.
fn emit_reply_token_and_decode_successor<Engine: Qwen3_5DecodeStepHooks>(
    engine: &mut Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    current_generated_token: &MlxArray,
    current_generated_token_id: u32,
) -> Result<ActiveRequestAdvance, InferenceEngineError> {
    let mut early_structured_emission = if active_request.structured_generation.is_some() {
        let generated_token_emission = build_generated_token_emission(
            engine.execution_core().loaded_model()?,
            &engine.execution_core().end_of_sequence_token_ids,
            active_request,
            current_generated_token_id,
            None,
        )?;
        if generated_token_emission.is_terminal {
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }
        Some(generated_token_emission)
    } else {
        None
    };
    let (next_generated_token, completed_forward_memory_snapshot) = engine.decode_next_token(
        active_request,
        current_generated_token,
        current_generated_token_id,
    )?;
    let generated_token_emission = match early_structured_emission.take() {
        Some(generated_token_emission) => generated_token_emission,
        None => build_generated_token_emission(
            engine.execution_core().loaded_model()?,
            &engine.execution_core().end_of_sequence_token_ids,
            active_request,
            current_generated_token_id,
            Some(&completed_forward_memory_snapshot),
        )?,
    };
    if generated_token_emission.is_terminal {
        Ok(ActiveRequestAdvance::Complete(
            generated_token_emission.generated_token,
        ))
    } else {
        // Keep the successor private until the next call reads it. The
        // current token alone is observable.
        active_request.pending_generated_token = Some(next_generated_token);
        Ok(ActiveRequestAdvance::Continue(
            generated_token_emission.generated_token,
        ))
    }
}

/// The current token for one decode step: a forced thinking transition
/// token, the prefill-produced pending token, or the first decode from the
/// final prompt token.
fn current_generated_token_for_decode<Engine: Qwen3_5DecodeStepHooks>(
    engine: &mut Engine,
    request_id: RequestId,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    forced_thinking_transition_token_id: Option<u32>,
    final_prompt_index: usize,
    first_decode_forward_started_at: &mut Option<Instant>,
) -> Result<MlxArray, InferenceEngineError> {
    if let Some(forced_thinking_transition_token_id) = forced_thinking_transition_token_id {
        return build_forced_thinking_transition_token(
            engine,
            active_request,
            forced_thinking_transition_token_id,
        );
    }
    match active_request.pending_generated_token.take() {
        Some(pending_generated_token) => {
            // Final prefill already produced this token; report a truthful
            // zero rather than attributing later output handling to decode.
            if pending_successor_reports_zero_first_decode_time(
                active_request.generated_token_count,
                active_request.first_decode_forward_elapsed_millis,
            ) {
                active_request.first_decode_forward_elapsed_millis = Some(0);
            }
            Ok(pending_generated_token)
        }
        None => engine.decode_first_token_from_final_prompt(
            request_id,
            active_request,
            final_prompt_index,
            first_decode_forward_started_at,
        ),
    }
}

/// Builds the forced thinking transition token. The selected successor was
/// conditioned on the last committed reasoning token but was never itself
/// forwarded; replacing it here keeps the forced token and decoder history
/// identical.
fn build_forced_thinking_transition_token<Engine: Qwen3_5DecodeStepHooks>(
    engine: &Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    forced_thinking_transition_token_id: u32,
) -> Result<MlxArray, InferenceEngineError> {
    active_request.pending_generated_token = None;
    let model = engine.execution_core().loaded_model()?;
    let forced_generated_token = active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::ForcedThinkingTransitionTokenArrayConstruction,
            |_performance_attribution| {
                model
                    .runtime()
                    .array_from_u32(&[forced_thinking_transition_token_id], &[1, 1])
            },
        )
        .map_err(qwen3_5_runtime_error)?;
    active_request
        .performance_attribution
        .record_counter(PerformanceCounter::ForcedThinkingTransitionTokenCount, 1);
    Ok(forced_generated_token)
}

/// Completes the request when the token just read ends the reply, with the
/// memory reading taken now; `None` when the reply continues.
fn complete_generation_if_terminal<Engine: Qwen3_5DecodeStepHooks>(
    engine: &Engine,
    active_request: &mut Qwen3_5EngineRequest<Engine::RequestExtension>,
    current_generated_token_id: u32,
) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
    if !generated_token_will_be_terminal(
        &engine.execution_core().end_of_sequence_token_ids,
        active_request,
        current_generated_token_id,
    ) {
        return Ok(None);
    }
    let model = engine.execution_core().loaded_model()?;
    let completed_forward_memory_snapshot = model.memory_snapshot_for_telemetry()?;
    let generated_token_emission = build_generated_token_emission(
        model,
        &engine.execution_core().end_of_sequence_token_ids,
        active_request,
        current_generated_token_id,
        Some(&completed_forward_memory_snapshot),
    )?;
    Ok(Some(ActiveRequestAdvance::Complete(
        generated_token_emission.generated_token,
    )))
}
