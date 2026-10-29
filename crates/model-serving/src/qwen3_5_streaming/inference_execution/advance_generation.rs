//! Serial request advancement with one-token-ahead graphics-processor submission.
//!
//! The token returned to the user is normally the predecessor of the token
//! already being evaluated.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use mlx_c_rust::MlxArray;

use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, GeneratedToken,
    InferenceEngineError, PerformanceAttributionOutcome, PerformanceOperation,
};

use super::completed_forward_memory::{
    CompletedForwardMemoryObservation, capture_completed_forward_memory_observation,
    collect_completed_forward_memory_snapshot, record_completed_adaptive_ram_growth,
};
use super::generated_token_emission;
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};

use crate::qwen3_5_streaming::model::Qwen3_5Model;
impl Qwen3_5EngineState {
    pub(super) fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let mut active_request = self.active_request.take().ok_or_else(|| {
            super::fatal_engine_error(
                "Qwen3.5 generation advance requested without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(super::fatal_engine_error(
                "Qwen3.5 generation request correlation mismatch",
            ));
        }

        let advance_span = if active_request.prefill_cursor
            < active_request.input_token_ids.len().saturating_sub(1)
        {
            PerformanceOperation::PromptPrefillAdvanceSpan
        } else {
            PerformanceOperation::DecodeAdvanceSpan
        };
        let advance_span_started_at = active_request
            .performance_attribution
            .begin_operation_span();
        let active_request_advance = self.advance_active_request(request_id, &mut active_request);
        active_request
            .performance_attribution
            .complete_operation_span(advance_span, advance_span_started_at);
        match active_request_advance {
            Ok(ActiveRequestAdvance::Continue(generated_token)) => {
                self.active_request = Some(active_request);
                Ok(generated_token)
            }
            Ok(ActiveRequestAdvance::Complete(generated_token)) => {
                let generation_finalization = self.finalize_generation_request(
                    active_request,
                    PerformanceAttributionOutcome::Success,
                    None,
                );
                Ok(generated_token.with_generation_finalization(generation_finalization))
            }
            Err(generation_error) => {
                self.finalize_generation_request_after_error(
                    active_request,
                    &generation_error,
                    "generation advance rejected",
                    "generation advance failed",
                );
                Err(generation_error)
            }
        }
    }

    /// Advances the active request by one decode step: prefill handoff,
    /// residency preparation and announcement, the forced/pending/first-decode
    /// token trichotomy, hot-expert warming, the terminal check, and the
    /// next-token forward with its memory observation and emission.
    fn advance_active_request(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        // Keep model borrows operation-local because memory admission may reclaim
        // retained pages through `&mut self`.
        if let Some(prefill_progress) =
            self.advance_prompt_prefill_if_pending(request_id, active_request)?
        {
            return Ok(ActiveRequestAdvance::Continue(prefill_progress));
        }
        // Restore the decode-phase retained-page budget before announcing
        // generation preparation, so the UI does not report the prefill policy.
        if !active_request.generation_residency_preparation_attempted {
            active_request.generation_residency_preparation_attempted = true;
            self.prepare_decode_expert_residency_after_prefill(request_id, active_request)?;
        }
        if !active_request.generation_preparation_announced {
            active_request.generation_preparation_announced = true;
            return Ok(ActiveRequestAdvance::Continue(
                self.announce_generation_preparation()?,
            ));
        }
        let final_prompt_index = active_request.input_token_ids.len() - 1;

        let forced_thinking_transition_token_id =
            active_request.next_forced_thinking_transition_token_id()?;
        let mut first_decode_forward_started_at = None;
        let current_generated_token = self.current_generated_token_for_decode(
            request_id,
            active_request,
            forced_thinking_transition_token_id,
            final_prompt_index,
            &mut first_decode_forward_started_at,
        )?;

        let current_generated_token_id = generated_token_emission::synchronize_generated_token_id(
            active_request,
            &current_generated_token,
        )?;
        if let Some(model) = self.model.as_ref() {
            self.warm_hot_experts_and_record_utilization(model, active_request);
            model
                .record_warm_insert_and_prefetch_statistics(
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
        }
        if let Some(first_decode_forward_started_at) = first_decode_forward_started_at {
            active_request.first_decode_forward_elapsed_millis = Some(
                u64::try_from(first_decode_forward_started_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
            );
        }
        if let Some(completed_advance) =
            self.complete_generation_if_terminal(active_request, current_generated_token_id)?
        {
            return Ok(completed_advance);
        }

        // Prefetch samples the successor before this token is accepted. A live
        // mask must see the commit first or Juliet/Romeo prefixes stay allowed.
        let mut early_structured_emission = if active_request.structured_generation.is_some() {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let generated_token_emission = self.build_generated_token_emission(
                model,
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
        let (next_generated_token, completed_forward_memory) = self
            .decode_next_token_and_collect_memory(
                active_request,
                &current_generated_token,
                current_generated_token_id,
                adaptive_decode_ram_growth_context(),
            )?;

        let generated_token_emission = match early_structured_emission.take() {
            Some(generated_token_emission) => generated_token_emission,
            None => {
                let model = self.model.as_ref().ok_or_else(|| {
                    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                })?;
                self.build_generated_token_emission(
                    model,
                    active_request,
                    current_generated_token_id,
                    Some(&completed_forward_memory),
                )?
            }
        };
        if generated_token_emission.is_terminal {
            Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ))
        } else {
            // Keep the asynchronously submitted successor private until the
            // next call synchronizes it. The current token alone is observable.
            active_request.pending_generated_token = Some(next_generated_token);
            Ok(ActiveRequestAdvance::Continue(
                generated_token_emission.generated_token,
            ))
        }
    }

    /// Completes the generation when the synchronized token is terminal.
    /// `Ok(None)` means the generation continues.
    fn complete_generation_if_terminal(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token_id: u32,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        if !self.generated_token_will_be_terminal(active_request, current_generated_token_id) {
            return Ok(None);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let completed_forward_memory = if self.adaptive_ram_growth_guard_enabled {
            Some(capture_completed_forward_memory_observation(model)?)
        } else {
            None
        };
        let generated_token_emission = self.build_generated_token_emission(
            model,
            active_request,
            current_generated_token_id,
            completed_forward_memory.as_ref(),
        )?;
        Ok(Some(ActiveRequestAdvance::Complete(
            generated_token_emission.generated_token,
        )))
    }

    /// Announces generation preparation from the model's live telemetry. The
    /// telemetry carries the residency derived from its own reconciled
    /// breakdown, so the announced claim and snapshot describe one instant
    /// (issue #337).
    fn announce_generation_preparation(&self) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry()?;
        let expert_residency = mlx_memory_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.expert_residency_telemetry())
            .unwrap_or_else(|| {
                model.expert_residency_telemetry_for_breakdown(
                    &crate::memory::telemetry::MlxActiveMemoryBreakdown::default(),
                )
            });
        Ok(GeneratedToken::GenerationPreparationStarted {
            total_layer_count: expert_residency.total_layer_count,
            resident_expert_count: expert_residency.resident_expert_count,
            resident_expert_payload_bytes: expert_residency.resident_expert_payload_bytes,
            mlx_memory_telemetry,
        })
    }

    /// The current token for one decode step: a forced thinking transition
    /// token, the prefill-produced pending token, or the first decode from the
    /// final prompt token.
    fn current_generated_token_for_decode(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        forced_thinking_transition_token_id: Option<u32>,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        if let Some(forced_thinking_transition_token_id) = forced_thinking_transition_token_id {
            return self.build_forced_thinking_transition_token(
                active_request,
                forced_thinking_transition_token_id,
            );
        }
        match active_request.pending_generated_token.take() {
            Some(pending_generated_token) => {
                // Final prefill already produced this token; report a truthful
                // zero rather than attributing later output handling to decode.
                if active_request.generated_token_count == 0
                    && active_request.first_decode_forward_elapsed_millis.is_none()
                {
                    active_request.first_decode_forward_elapsed_millis = Some(0);
                }
                Ok(pending_generated_token)
            }
            None => self.decode_first_token_from_final_prompt(
                request_id,
                active_request,
                final_prompt_index,
                first_decode_forward_started_at,
            ),
        }
    }

    /// Builds the forced thinking transition token. The asynchronously selected
    /// successor was conditioned on the last committed reasoning token but was
    /// never itself forwarded; replacing it here keeps the forced token and
    /// decoder history identical. Mirrors the inlined arm.
    fn build_forced_thinking_transition_token(
        &self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        forced_thinking_transition_token_id: u32,
    ) -> Result<MlxArray, InferenceEngineError> {
        active_request.pending_generated_token = None;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
        active_request.performance_attribution.record_counter(
            crate::PerformanceCounter::ForcedThinkingTransitionTokenCount,
            1,
        );
        Ok(forced_generated_token)
    }

    /// Decodes the first token from the final prompt token under the decode
    /// admission baseline, records the route observation, and files the
    /// completed-forward memory evidence. Mirrors the inlined first-decode
    /// arm, including the first-token decode seam log.
    fn decode_first_token_from_final_prompt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        final_prompt_index: usize,
        first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        let final_prompt_token_id = active_request.input_token_ids[final_prompt_index];
        let adaptive_ram_growth_context =
            AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Paged);
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
        model.begin_decode_expert_cache_forward();
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
        model.end_decode_expert_cache_forward();
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
        record_completed_adaptive_ram_growth(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_execution_profile(AdaptiveRamGrowthExecutionProfile::Paged),
            true,
            model,
            active_request.request_context_extent(),
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
        Ok(first_generated_token)
    }

    /// Records the decode-phase memory ceiling utilization and bounds retained
    /// expert ownership before warming. Hot-expert warming (issue #372) must
    /// never take the adaptive growth guard's headroom: the learned transient
    /// reserve is what the next forward's admission needs, so warming
    /// self-regulates and yields tables back under pressure. Mirrors the
    /// inlined block.
    fn warm_hot_experts_and_record_utilization(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) {
        if self.adaptive_ram_growth_guard_enabled
            && let Ok(memory_snapshot) = model.runtime().memory_snapshot()
        {
            // Issue #507: record why the ceiling is not fully in use at this
            // step, alongside the warming cap that decides it. Without the
            // split, an idle gigabyte cannot be told apart from a reserve
            // held for context growth.
            let decode_context_token_count = u64::try_from(
                active_request
                    .input_token_ids
                    .len()
                    .saturating_add(usize::from(active_request.generated_token_count)),
            )
            .unwrap_or(u64::MAX);
            let decode_active_memory_bytes =
                u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(u64::MAX);
            model.record_memory_ceiling_utilization(
                crate::MemoryPhase::Decode,
                decode_context_token_count,
                1,
                &active_request.request_decoder_state,
                0,
                decode_active_memory_bytes,
                &mut active_request.performance_attribution,
            );
            let routed_expert_page_reservation_bytes = model
                .expert_page_reservation_bytes_for_forward(1)
                .unwrap_or(u64::MAX);
            let warm_retention_ceiling_bytes = self
                .adaptive_ram_growth_guard
                .hot_expert_retention_ceiling_bytes(
                    crate::MemoryPhase::Decode,
                    memory_snapshot.active_memory_bytes(),
                    model
                        .expert_weight_memory_cache_statistics()
                        .resident_payload_byte_count,
                    usize::try_from(routed_expert_page_reservation_bytes).unwrap_or(usize::MAX),
                );
            // The plan-composed ceiling already subtracted the composer's
            // activation workspace, context reserve, and stream slot;
            // warming must respect whichever ceiling is tighter.
            let plan_retained_ceiling_bytes = model.retained_expert_normal_ceiling_bytes();
            model.limit_retained_experts_to(
                warm_retention_ceiling_bytes.min(plan_retained_ceiling_bytes),
            );
        }
    }

    /// Runs the next decode forward under a fresh admission baseline and
    /// collects the completed-forward memory snapshot. Mirrors the inlined
    /// admission + forward + collection sequence.
    fn decode_next_token_and_collect_memory(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
        current_generated_token_id: u32,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    ) -> Result<(MlxArray, CompletedForwardMemoryObservation), InferenceEngineError> {
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
        let next_generated_token = {
            model.begin_decode_expert_cache_forward();
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
            model.end_decode_expert_cache_forward();
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
        let completed_forward_memory = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_execution_profile(AdaptiveRamGrowthExecutionProfile::Paged),
            true,
            model,
            active_request.request_context_extent(),
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
        Ok((next_generated_token, completed_forward_memory))
    }
}

pub(super) enum ActiveRequestAdvance {
    Continue(GeneratedToken),
    Complete(GeneratedToken),
}

/// The decode admission context for one streaming decode step.
fn adaptive_decode_ram_growth_context() -> AdaptiveRamGrowthContext {
    AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Paged)
}
