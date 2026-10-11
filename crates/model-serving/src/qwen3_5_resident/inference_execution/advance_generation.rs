//! Serial request advancement with one-token-ahead graphics-processor submission.
//!
//! The token returned to the user is normally the predecessor of the token
//! already being evaluated.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;

use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, GeneratedToken,
    InferenceEngineError, PerformanceAttributionOutcome, PerformanceOperation,
};

use super::completed_forward_memory::{
    capture_completed_forward_memory_observation, collect_completed_forward_memory_snapshot,
    record_completed_adaptive_ram_growth,
};
use super::generated_token_emission;
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
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

    fn advance_active_request(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<ActiveRequestAdvance, InferenceEngineError> {
        // Keep model borrows operation-local so request completion can release memory.
        if let Some(prefill_progress) =
            self.advance_prompt_prefill_if_pending(request_id, active_request)?
        {
            return Ok(ActiveRequestAdvance::Continue(prefill_progress));
        }
        if !active_request.generation_residency_preparation_attempted {
            active_request.generation_residency_preparation_attempted = true;
        }
        if !active_request.generation_preparation_announced {
            active_request.generation_preparation_announced = true;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry()?;
            // The telemetry carries the residency derived from its own reconciled
            // breakdown, so the announced claim and snapshot describe one instant
            // (issue #337).
            let expert_residency = mlx_memory_telemetry
                .as_ref()
                .and_then(|telemetry| telemetry.expert_residency_telemetry())
                .unwrap_or_else(|| {
                    model.expert_residency_telemetry_for_breakdown(
                        &crate::MlxActiveMemoryBreakdown::default(),
                    )
                });
            return Ok(ActiveRequestAdvance::Continue(
                crate::GeneratedToken::GenerationPreparationStarted {
                    total_layer_count: expert_residency.total_layer_count,
                    resident_expert_count: expert_residency.resident_expert_count,
                    resident_expert_payload_bytes: expert_residency.resident_expert_payload_bytes,
                    mlx_memory_telemetry,
                },
            ));
        }
        let final_prompt_index = active_request.input_token_ids.len() - 1;

        let forced_thinking_transition_token_id =
            active_request.next_forced_thinking_transition_token_id()?;
        let mut first_decode_forward_started_at = None;
        let current_generated_token = if let Some(forced_thinking_transition_token_id) =
            forced_thinking_transition_token_id
        {
            // The asynchronously selected successor was conditioned on the last
            // committed reasoning token but was never itself forwarded. Replacing
            // it here keeps the forced token and decoder history identical.
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
            forced_generated_token
        } else {
            match active_request.pending_generated_token.take() {
                Some(pending_generated_token) => {
                    // Final prefill already produced this token; report a truthful
                    // zero rather than attributing later output handling to decode.
                    if active_request.generated_token_count == 0
                        && active_request.first_decode_forward_elapsed_millis.is_none()
                    {
                        active_request.first_decode_forward_elapsed_millis = Some(0);
                    }
                    pending_generated_token
                }
                None => {
                    let final_prompt_token_id = active_request.input_token_ids[final_prompt_index];
                    let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
                        1,
                        AdaptiveRamGrowthExecutionProfile::Resident,
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
                    let model = self.model.as_ref().ok_or_else(|| {
                        super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    // This log is the first-token decode seam after prompt restore.
                    tracing::info!(
                        request_id = request_id.value(),
                        context_token_count = active_request.input_token_ids.len(),
                        expert_memory_mode = ?model.expert_memory_mode(),
                        generation_residency_preparation_attempted =
                            active_request.generation_residency_preparation_attempted,
                        "starting first decode forward after prompt processing"
                    );
                    first_decode_forward_started_at = Some(Instant::now());
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
                    record_completed_adaptive_ram_growth(
                        &mut self.adaptive_ram_growth_guard,
                        adaptive_ram_growth_context,
                        true,
                        model,
                        active_memory_bytes_before_growth,
                        retained_expert_payload_bytes_before_growth,
                        0,
                        streamed_expert_page_bytes_before_growth,
                        &mut active_request.performance_attribution,
                    )?;
                    first_generated_token
                }
            }
        };

        let current_generated_token_id = generated_token_emission::synchronize_generated_token_id(
            active_request,
            &current_generated_token,
        )?;
        if let Some(first_decode_forward_started_at) = first_decode_forward_started_at {
            active_request.first_decode_forward_elapsed_millis = Some(
                u64::try_from(first_decode_forward_started_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
            );
        }
        if self.generated_token_will_be_terminal(active_request, current_generated_token_id) {
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
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }

        let adaptive_ram_growth_context =
            AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident);
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
        // Prefetch samples the successor before this token is accepted. A live
        // mask must see the commit first or Juliet/Romeo prefixes stay allowed.
        let mut generated_token_emission = if active_request.structured_generation.is_some() {
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
        let next_generated_token = {
            let next_logits = model
                .build_generated_token_forward_with_performance_attribution(
                    &current_generated_token,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            let next_generated_token = active_request.build_generated_token(model, &next_logits)?;
            next_generated_token
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
        let completed_forward_memory = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            true,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;

        let generated_token_emission = match generated_token_emission.take() {
            Some(generated_token_emission) => generated_token_emission,
            None => self.build_generated_token_emission(
                model,
                active_request,
                current_generated_token_id,
                Some(&completed_forward_memory),
            )?,
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
}

pub(super) enum ActiveRequestAdvance {
    Continue(GeneratedToken),
    Complete(GeneratedToken),
}
