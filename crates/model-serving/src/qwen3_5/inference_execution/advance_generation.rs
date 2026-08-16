//! Serial request advancement with one-token-ahead graphics-processor submission.
//!
//! The token returned to the user is normally the predecessor of the token
//! already being evaluated.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;

use crate::{
    AdaptiveRamGrowthContext, GeneratedToken, InferenceEngineError, PerformanceAttributionOutcome,
    PerformanceOperation,
};

use super::completed_forward_memory::{
    collect_completed_forward_memory_snapshot, record_completed_adaptive_ram_growth,
};
use super::generated_token_emission::synchronize_generated_token_id;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::multi_token_prediction::{
    forward_initial_target_token_with_prediction_state,
    forward_next_target_token_with_prediction_state, take_queued_prediction_token,
};
impl Qwen3_5EngineState {
    pub(super) fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let mut active_request = self.active_request.take().ok_or_else(|| {
            fatal_engine_error("Qwen3.5 generation advance requested without an active request")
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(fatal_engine_error(
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
        // Keep model borrows operation-local. Memory admission may demote the
        // complete resident owner through `&mut self`; every forward must borrow
        // the model again afterward and observe the newly selected mode.
        if let Some(prefill_progress) =
            self.advance_prompt_prefill_if_pending(request_id, active_request)?
        {
            return Ok(ActiveRequestAdvance::Continue(prefill_progress));
        }
        if !active_request.generation_preparation_announced {
            active_request.generation_preparation_announced = true;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let expert_residency = model.expert_residency_telemetry();
            return Ok(ActiveRequestAdvance::Continue(
                crate::GeneratedToken::GenerationPreparationStarted {
                    total_layer_count: expert_residency.total_layer_count,
                    complete_layer_count: expert_residency.complete_layer_count,
                    complete_layer_payload_bytes: expert_residency.complete_layer_payload_bytes,
                    partial_layer_count: expert_residency.partial_layer_count,
                    partial_layer_payload_bytes: expert_residency.partial_layer_payload_bytes,
                },
            ));
        }
        // Prefill just finished. Reconcile the larger generation-phase budget
        // without reading storage; mandatory decode reads populate empty route
        // ownership later. The one-shot flag prevents repeated planning.
        if !active_request.generation_residency_preparation_attempted {
            active_request.generation_residency_preparation_attempted = true;
            self.prepare_decode_expert_residency_after_prefill(request_id, active_request)?;
        }
        let final_prompt_index = active_request.input_token_ids.len() - 1;

        if let Some(queued_prediction_token_id) = take_queued_prediction_token(active_request) {
            // The prediction path already computed the first observable token,
            // so no additional first decode forward is required at this boundary.
            if active_request.generated_token_count == 0
                && active_request.first_decode_forward_elapsed_millis.is_none()
            {
                active_request.first_decode_forward_elapsed_millis = Some(0);
            }
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let mlx_memory_snapshot = if !self.adaptive_ram_growth_guard_enabled {
                None
            } else {
                Some(
                    model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_runtime_error)?,
                )
            };
            let generated_token_emission = self.build_generated_token_emission(
                model,
                active_request,
                queued_prediction_token_id,
                mlx_memory_snapshot.as_ref(),
            )?;
            return if generated_token_emission.is_terminal {
                Ok(ActiveRequestAdvance::Complete(
                    generated_token_emission.generated_token,
                ))
            } else {
                Ok(ActiveRequestAdvance::Continue(
                    generated_token_emission.generated_token,
                ))
            };
        }

        let mut first_decode_forward_started_at = None;
        let current_generated_token = match active_request.pending_generated_token.take() {
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
                let sparse_experts_are_paged = self
                    .model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                    .sparse_experts_are_paged();
                let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
                    1,
                    active_request.has_optional_prediction_session(),
                    sparse_experts_are_paged,
                );
                let (
                    active_memory_bytes_before_growth,
                    retained_expert_payload_bytes_before_growth,
                ) = self.measure_adaptive_ram_growth_memory_admission(
                    adaptive_ram_growth_context,
                    &mut active_request.performance_attribution,
                    &active_request.request_decoder_state,
                    0,
                    0,
                )?;
                self.save_speculative_prefill_target_prefix(active_request)?;
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
                // This log is the first-token decode seam. After the restore
                // above, `sparse_experts_are_paged` tells whether generation
                // will use the complete owner or stream/split retained pages.
                tracing::info!(
                    request_id = request_id.value(),
                    context_token_count = active_request.input_token_ids.len(),
                    sparse_experts_are_paged,
                    generation_residency_preparation_attempted =
                        active_request.generation_residency_preparation_attempted,
                    "starting first decode forward after prompt processing"
                );
                first_decode_forward_started_at = Some(Instant::now());
                let first_generated_token = if let Some(prediction_token) =
                    forward_initial_target_token_with_prediction_state(
                        model,
                        active_request,
                        final_prompt_token_id,
                    )? {
                    prediction_token
                } else {
                    let final_prompt_logits = if model.sparse_experts_are_paged() {
                        // Paged decode resolves deferred GPU missing-route bitmaps
                        // on a synchronous completion root before the token is
                        // observable.
                        model
                            .forward_chunk_with_performance_attribution(
                                &[final_prompt_token_id],
                                active_request.next_position_tokens,
                                &mut active_request.request_decoder_state,
                                &mut active_request.performance_attribution,
                            )
                            .map_err(InferenceEngineError::from)?
                    } else {
                        model
                            .build_forward_chunk_with_performance_attribution(
                                &[final_prompt_token_id],
                                active_request.next_position_tokens,
                                &mut active_request.request_decoder_state,
                                &mut active_request.performance_attribution,
                            )
                            .map_err(InferenceEngineError::from)?
                    };
                    active_request.advance_position(1)?;
                    active_request.build_generated_token(model, &final_prompt_logits)?
                };
                if !model.sparse_experts_are_paged() {
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
                }
                record_completed_adaptive_ram_growth(
                    &mut self.adaptive_ram_growth_guard,
                    adaptive_ram_growth_context
                        .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                    true,
                    model,
                    active_memory_bytes_before_growth,
                    retained_expert_payload_bytes_before_growth,
                    0,
                    &mut active_request.performance_attribution,
                )?;
                first_generated_token
            }
        };

        let current_generated_token_id =
            synchronize_generated_token_id(active_request, &current_generated_token)?;
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
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let mlx_memory_snapshot = if self.adaptive_ram_growth_guard_enabled {
                Some(
                    model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_runtime_error)?,
                )
            } else {
                None
            };
            let generated_token_emission = self.build_generated_token_emission(
                model,
                active_request,
                current_generated_token_id,
                mlx_memory_snapshot.as_ref(),
            )?;
            return Ok(ActiveRequestAdvance::Complete(
                generated_token_emission.generated_token,
            ));
        }

        if let Some(prediction_advance) = self.attempt_mtp_decode_window(
            request_id,
            active_request,
            &current_generated_token,
            current_generated_token_id,
        )? {
            return Ok(prediction_advance);
        }

        let sparse_experts_are_paged = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .sparse_experts_are_paged();
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(
            1,
            active_request.has_optional_prediction_session(),
            sparse_experts_are_paged,
        );
        let (active_memory_bytes_before_growth, retained_expert_payload_bytes_before_growth) = self
            .measure_adaptive_ram_growth_memory_admission(
                adaptive_ram_growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                0,
                0,
            )?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let next_generated_token = if let Some(prediction_token) =
            forward_next_target_token_with_prediction_state(
                model,
                active_request,
                &current_generated_token,
            )? {
            prediction_token
        } else {
            let next_logits = model
                .build_generated_token_forward_with_performance_attribution(
                    &current_generated_token,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            active_request.advance_position(1)?;
            active_request.build_generated_token(model, &next_logits)?
        };
        if !model.sparse_experts_are_paged() {
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
        }
        let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            true,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            0,
            &mut active_request.performance_attribution,
        )?;

        let generated_token_emission = self.build_generated_token_emission(
            model,
            active_request,
            current_generated_token_id,
            mlx_memory_snapshot.as_ref(),
        )?;
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
