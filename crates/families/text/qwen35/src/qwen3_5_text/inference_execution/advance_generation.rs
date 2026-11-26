//! The streaming engine's part of a decode step: reconciling expert
//! residency after the prompt, the expert statistics recorded once a reply
//! token is read, and its forwards, which run the successor synchronously
//! because its expert routes must be resolved before the token may become
//! visible; unlike the resident engine, nothing is submitted to the graphics
//! processor ahead of time. The step itself is shared with the resident
//! engine in `text_generation::generation_loop::decode_step`.

use std::time::Instant;
use text_generation::model::memory_breakdown::TextModelMemoryOwnership;

use ipc_protocol::RequestId;
use runtime_integration::MlxMemorySnapshot;

use mlx_bridge::MlxArray;
use text_generation::generation_loop::decode_step::{ActiveRequestAdvance, TextDecodeStepHooks};

use crate::{GeneratedToken, InferenceEngineError};

use super::Qwen3_5EngineState;
use super::completed_forward_memory::{
    CompletedForwardMemoryObservation, measure_completed_reply_forward,
};
use super::engine_request::Qwen3_5EngineRequest;

use crate::qwen3_5_text::model::Qwen3_5Model;
impl TextDecodeStepHooks for Qwen3_5EngineState {
    fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        Qwen3_5EngineState::advance_prompt_prefill_if_pending(self, request_id, active_request)
    }

    fn begin_generation_preparation(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        Qwen3_5EngineState::begin_generation_preparation(self, request_id, active_request)
    }

    /// The one place a reply is drafted: with a hosted DFlash drafter when
    /// the reply opened drafting with one, otherwise with the model's
    /// multi-token-prediction head when the engine is set to. `None` hands
    /// the step to the plain path.
    fn advance_drafted_generation_if_admitted(
        &mut self,
        _request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        forced_thinking_transition_token_id: Option<u32>,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        if let Some(drafted_advance) = self.advance_draft_tree_generation_if_admitted(
            active_request,
            forced_thinking_transition_token_id,
        )? {
            return Ok(Some(drafted_advance));
        }
        self.advance_mtp_drafted_generation_if_admitted(
            active_request,
            forced_thinking_transition_token_id,
        )
    }

    /// Warms the hot experts the token used and records the warm-insert and
    /// prefetch statistics of the forward that produced it.
    fn after_reply_token_synchronized(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<(), InferenceEngineError> {
        if let Some(model) = self.core.model.as_ref() {
            self.warm_hot_experts_and_record_utilization(model, active_request);
            model
                .record_warm_insert_and_prefetch_statistics(
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
        }
        Ok(())
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
        current_generated_token_id: u32,
    ) -> Result<(MlxArray, MlxMemorySnapshot), InferenceEngineError> {
        let (next_generated_token, completed_forward_memory) = self
            .reclaim_outgrown_slabs_and_decode_next_token(
                active_request,
                current_generated_token,
                current_generated_token_id,
            )?;
        Ok((
            next_generated_token,
            completed_forward_memory.mlx_memory_snapshot,
        ))
    }
}

impl Qwen3_5EngineState {
    /// Announces preparation the moment prompt work ends, then reconciles
    /// decode residency on the next advance. Reconciling first kept the
    /// public status on "prompt processing 100%" for as long as residency
    /// took: about 8.5 s on the first request after a load. The announcement
    /// carries no residency claim, so the prefill policy is never reported as
    /// the decode residency; the first decode observation reports the
    /// reconciled one. `Some` is the announcement to emit now.
    fn begin_generation_preparation(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        if !active_request.generation_preparation_announced {
            active_request.generation_preparation_announced = true;
            return Ok(Some(ActiveRequestAdvance::Continue(
                GeneratedToken::GenerationPreparationStarted {
                    expert_residency: None,
                    mlx_memory_telemetry: None,
                },
            )));
        }
        if !active_request.generation_residency_preparation_attempted {
            active_request.generation_residency_preparation_attempted = true;
            self.prepare_decode_expert_residency_after_prefill(request_id, active_request)?;
        }
        Ok(None)
    }

    /// Decodes the first token from the final prompt token, records the
    /// route observation, and reports the forward's measured memory to the
    /// governor. Mirrors the inlined first-decode arm, including the
    /// first-token decode seam log.
    fn decode_first_token_from_final_prompt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        let final_prompt_token_id = active_request.input_token_ids[final_prompt_index];
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let model = self.core.loaded_model()?;
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
        model.begin_decode_astro_expert_cache_forward();
        let decode_forward_started_at = active_request
            .performance_attribution
            .is_enabled()
            .then(Instant::now);
        let final_prompt_logits_result = model.forward_chunk_with_performance_attribution(
            &[final_prompt_token_id],
            active_request.next_position_tokens,
            &mut active_request.request_decoder_state,
            &mut active_request.performance_attribution,
        );
        if let Some(decode_forward_started_at) = decode_forward_started_at {
            active_request
                .performance_attribution
                .record_decode_token_forward_elapsed(decode_forward_started_at.elapsed());
        }
        model.end_decode_astro_expert_cache_forward();
        let final_prompt_logits = final_prompt_logits_result.map_err(InferenceEngineError::from)?;
        active_request.advance_position(1)?;
        let first_generated_token =
            active_request.build_generated_token(model, &final_prompt_logits)?;
        // Issue #536/#542: first decode token route history.
        // Logits evaluation already materialized the retained
        // router arrays; this only copies host identifiers.
        model.finalize_route_observation_record(
            final_prompt_token_id,
            &mut active_request.performance_attribution,
        );
        measure_completed_reply_forward(
            model,
            active_request,
            memory_baseline.active_memory_bytes,
        )?;
        Ok(first_generated_token)
    }

    /// Records the decode-phase memory ceiling utilization. Hot-expert
    /// warming (issue #372) stays inside the capacity the memory governor
    /// gives the astro expert cache, which already leaves a reply step its working
    /// memory.
    fn warm_hot_experts_and_record_utilization(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) {
        if let Ok(memory_snapshot) = model.memory_snapshot_with_forward_peak() {
            // Issue #507: record why the ceiling is not fully in use at this
            // step, alongside the warming cap that decides it. Without the
            // split, an idle gigabyte cannot be told apart from a reserve
            // held for context growth.
            let decode_active_memory_bytes =
                u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(u64::MAX);
            model.record_memory_ceiling_utilization(
                &active_request.request_decoder_state,
                0,
                decode_active_memory_bytes,
                &mut active_request.performance_attribution,
            );
        }
    }

    /// Releases the attention slabs the previous step outgrew, then decodes
    /// the next token. The previous token is synchronized by now, so nothing
    /// is in flight while the `mlx_buffer_cache` is released.
    fn reclaim_outgrown_slabs_and_decode_next_token(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
        current_generated_token_id: u32,
    ) -> Result<(MlxArray, CompletedForwardMemoryObservation), InferenceEngineError> {
        if let Some(model) = self.core.model.as_ref() {
            active_request.reclaim_outgrown_attention_slabs(model.runtime())?;
        }
        self.decode_next_token_and_collect_memory(
            active_request,
            current_generated_token,
            current_generated_token_id,
        )
    }

    /// Runs the next decode forward from a fresh memory baseline and
    /// measures it when it completes.
    fn decode_next_token_and_collect_memory(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
        current_generated_token_id: u32,
    ) -> Result<(MlxArray, CompletedForwardMemoryObservation), InferenceEngineError> {
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let model = self.core.loaded_model()?;
        let expert_route_sample_before = begin_reply_token_expert_routes(model, active_request);
        let next_generated_token = {
            model.begin_decode_astro_expert_cache_forward();
            let decode_forward_started_at = active_request
                .performance_attribution
                .is_enabled()
                .then(Instant::now);
            let next_logits_result = model
                .build_generated_token_forward_with_performance_attribution(
                    current_generated_token,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                );
            if let Some(decode_forward_started_at) = decode_forward_started_at {
                active_request
                    .performance_attribution
                    .record_decode_token_forward_elapsed(decode_forward_started_at.elapsed());
            }
            model.end_decode_astro_expert_cache_forward();
            let next_logits = next_logits_result.map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            let next_generated_token = active_request.build_generated_token(model, &next_logits)?;
            // Issue #536/#542: decode route history after sampling. Route
            // arrays were extra roots of the logits wait.
            model.finalize_route_observation_record(
                current_generated_token_id,
                &mut active_request.performance_attribution,
            );
            next_generated_token
        };
        let completed_forward_memory = measure_completed_reply_forward(
            model,
            active_request,
            memory_baseline.active_memory_bytes,
        )?;
        finish_reply_token_expert_routes(model, active_request, &expert_route_sample_before);
        active_request
            .engine_extension
            .mlx_buffer_cache_reply_record
            .record_token(
                u64::try_from(
                    completed_forward_memory
                        .mlx_memory_snapshot
                        .mlx_buffer_cache_bytes(),
                )
                .unwrap_or(u64::MAX),
            );
        Ok((next_generated_token, completed_forward_memory))
    }
}

/// Before a reply token's forward: notes the reply's start for a recorded
/// reply (its first token only) and samples the session's expert counters.
fn begin_reply_token_expert_routes(
    model: &crate::qwen3_5_text::model::Qwen3_5Model,
    active_request: &super::engine_request::Qwen3_5EngineRequest,
) -> crate::memory::reply_expert_route_record::SessionExpertRouteSample {
    if active_request
        .engine_extension
        .reply_expert_route_record
        .is_empty()
    {
        model.note_reply_start_for_route_recording(
            active_request.performance_attribution.is_enabled(),
        );
    }
    model.session_expert_route_sample()
}

/// After a reply token's forward: records what the token did to the
/// `astro_expert_cache`, the change in the session's counters across it.
fn finish_reply_token_expert_routes(
    model: &crate::qwen3_5_text::model::Qwen3_5Model,
    active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    expert_route_sample_before: &crate::memory::reply_expert_route_record::SessionExpertRouteSample,
) {
    let expert_route_token_change = model
        .session_expert_route_sample()
        .change_since(expert_route_sample_before);
    active_request
        .engine_extension
        .reply_expert_route_record
        .record_token(expert_route_token_change);
}
