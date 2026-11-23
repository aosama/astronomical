//! The resident engine's part of a decode step: its generation-preparation
//! announcement, its drafted advance, and its forwards, which are admitted by
//! the memory governor before they run and submit the successor to the
//! graphics processor one token ahead. The step itself is shared with the
//! streaming engine in `qwen3_5_core::shared_execution::decode_step`.

use crate::qwen3_5_core::shared_execution::generation_finalization_steps;
use std::time::Instant;

use ipc_protocol::RequestId;
use runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5_core::shared_execution::decode_step::{
    ActiveRequestAdvance, Qwen3_5DecodeStepHooks,
};
use mlx_bridge::MlxArray;

use crate::{
    GeneratedToken, InferenceEngineError, PerformanceAttributionOutcome, PerformanceOperation,
};

use super::completed_forward_memory::{
    CompletedForwardMemoryObservation, capture_completed_forward_memory_observation,
};
use super::engine_request::Qwen3_5EngineRequest;
use super::forward_memory::{ForwardShape, report_completed_forward};
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
use crate::qwen3_5_core::shared_execution::generated_token_emission;
impl Qwen3_5DecodeStepHooks for Qwen3_5EngineState {
    fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        Qwen3_5EngineState::advance_prompt_prefill_if_pending(self, request_id, active_request)
    }

    /// The resident engine has no residency to reconcile after prefill, so
    /// it announces preparation at once and continues on the next advance.
    fn begin_generation_preparation(
        &mut self,
        _request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        active_request.generation_residency_preparation_attempted = true;
        if active_request.generation_preparation_announced {
            return Ok(None);
        }
        active_request.generation_preparation_announced = true;
        Ok(Some(ActiveRequestAdvance::Continue(
            self.announce_generation_preparation()?,
        )))
    }

    /// Speculative sessions run their own advance: accepted blocks drain
    /// through the ordinary emission machinery, and an empty queue means one
    /// draft-and-verify round. A forced thinking boundary suspends speculation
    /// for the session's remainder: the block contract cannot re-seed a
    /// mid-block forced token, and the plain decode step handles that
    /// boundary exactly.
    fn advance_drafted_generation_if_admitted(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        forced_thinking_transition_token_id: Option<u32>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        if !active_request.engine_extension.speculative_session_admitted {
            return Ok(None);
        }
        if forced_thinking_transition_token_id.is_some() {
            active_request.suspend_speculative_session("forced_thinking_transition");
            return Ok(None);
        }
        if active_request.has_speculative_tokens_to_advance() {
            return self
                .advance_speculative_generation(request_id, active_request)
                .map(Some);
        }
        Ok(None)
    }

    fn decode_first_token_from_final_prompt(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        Qwen3_5EngineState::decode_first_token_from_final_prompt(
            self,
            request_id,
            active_request,
            final_prompt_index,
            first_decode_forward_started_at,
        )
    }

    fn decode_next_token(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
        _current_generated_token_id: u32,
    ) -> Result<(MlxArray, MlxMemorySnapshot), InferenceEngineError> {
        let (next_generated_token, completed_forward_memory) = self
            .reclaim_outgrown_slabs_and_decode_next_token(
                active_request,
                current_generated_token,
            )?;
        Ok((
            next_generated_token,
            completed_forward_memory.mlx_memory_snapshot,
        ))
    }
}

impl Qwen3_5EngineState {
    /// Announces generation preparation from the model's live telemetry. The
    /// telemetry carries the residency derived from its own reconciled
    /// breakdown, so the announced claim and snapshot describe one instant
    /// (issue #337).
    fn announce_generation_preparation(&self) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let mlx_memory_telemetry =
            generation_finalization_steps::collect_current_mlx_memory_telemetry(self)?;
        let expert_residency = match mlx_memory_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.expert_residency_telemetry())
        {
            Some(expert_residency_telemetry) => expert_residency_telemetry,
            None => model.expert_residency_telemetry_for_breakdown(
                &crate::memory::telemetry::MlxActiveMemoryBreakdown::default(),
            ),
        };
        // The resident engine has no residency to reconcile after prefill, so
        // its announcement already describes the decode residency.
        Ok(GeneratedToken::GenerationPreparationStarted {
            expert_residency: Some(expert_residency),
            mlx_memory_telemetry,
        })
    }

    /// Asks whether a reply step fits, decodes the first token from the final
    /// prompt token, submits the successor for asynchronous evaluation, and
    /// reports what the step needed.
    fn decode_first_token_from_final_prompt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        let final_prompt_token_id = active_request.input_token_ids[final_prompt_index];
        let admitted_forward = self.admit_forward(
            ForwardShape::reply_token(),
            &active_request.request_decoder_state,
            0,
            0,
        )?;
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        // This log is the first-token decode seam after prompt restore.
        tracing::info!(
            request_id = request_id.value(),
            context_token_count = active_request.input_token_ids.len(),
            expert_memory_mode = ?model.expert_memory_mode(),
            generation_residency_preparation_attempted =
                active_request.generation_residency_preparation_attempted,
            "starting first decode forward after prompt processing"
        );
        *first_decode_forward_started_at = Some(Instant::now());
        let final_prompt_logits = model
            .build_forward_chunk_with_performance_attribution(
                &[final_prompt_token_id],
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            )
            .map_err(InferenceEngineError::from)?;
        active_request.advance_position(1)?;
        let first_generated_token =
            active_request.build_generated_token(model, &final_prompt_logits)?;
        // Issue #536/#542: first decode token route history.
        // Logits evaluation already materialized the retained
        // router arrays; this only copies host identifiers.
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
        report_completed_forward(
            model,
            admitted_forward,
            active_request.request_decoder_state.payload_byte_count(),
        )?;
        Ok(first_generated_token)
    }

    /// Releases the attention slabs the previous step outgrew, then decodes
    /// the next token. The previous token is synchronized by now, so nothing
    /// is in flight while the `mlx_buffer_cache` is released.
    fn reclaim_outgrown_slabs_and_decode_next_token(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
    ) -> Result<(MlxArray, CompletedForwardMemoryObservation), InferenceEngineError> {
        let model = self.core.model.as_ref().ok_or_else(
            super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error,
        )?;
        active_request.reclaim_outgrown_attention_slabs(model.runtime())?;
        self.decode_next_token_and_collect_memory(active_request, current_generated_token)
    }

    /// Asks whether the next reply step fits, runs it, submits the successor
    /// for asynchronous evaluation, and reports what the step needed.
    fn decode_next_token_and_collect_memory(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
    ) -> Result<(MlxArray, CompletedForwardMemoryObservation), InferenceEngineError> {
        let admitted_forward = self.admit_forward(
            ForwardShape::reply_token(),
            &active_request.request_decoder_state,
            0,
            0,
        )?;
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let next_generated_token = {
            let next_logits = model
                .build_generated_token_forward_with_performance_attribution(
                    current_generated_token,
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
        let completed_forward_memory = report_completed_forward(
            model,
            admitted_forward,
            active_request.request_decoder_state.payload_byte_count(),
        )?;
        Ok((next_generated_token, completed_forward_memory))
    }
}
