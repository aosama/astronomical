//! A prompt read one layer at a time, advanced one layer per engine step.
//!
//! One layer runs over every span of the prompt before the next layer is
//! read (issue #1985 for dense models, issue #2221 for expert models). The
//! whole prompt is one forward as far as the prompt cache and the request's
//! cursor are concerned, so no token is finished until the last layer has
//! run. Advancing one layer per step lets the engine report how many layers
//! are done after each one, and lets a request be cancelled between layers
//! instead of only when the whole prompt has been read.

use std::time::Instant;

use astronomical_ipc_protocol::WorkerPromptLayerProgress;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::forward_memory_baseline::ForwardMemoryBaseline;
use super::layer_major_span_plan::LayerMajorSpanPlan;
use super::prefill_chunk_planning::PrefillChunkPlan;
use super::prefill_memory_recovery::{PrefillAttemptFailure, ran_out_of_memory};
use super::prompt_prefill::PromptPrefillChunkOutcome;
use super::prompt_prefill_errors::{self, PromptPrefillChunkAttemptError};
use crate::qwen3_5_streaming::model::layer_major_prefill::{
    LayerMajorCheckpointContract, LayerMajorPrefillRequest, LayerMajorPrefillRun,
};
use crate::{GeneratedToken, InferenceEngineError, PerformanceCounter};
use astronomical_ipc_protocol::RequestId;

/// A layer-at-a-time prompt between its first layer and its last.
pub(in crate::qwen3_5_streaming) struct LayerAtATimePrefillInFlight {
    layer_major_prefill_run: LayerMajorPrefillRun,
    plan: PrefillChunkPlan,
    prefill_start: usize,
    prefill_end: usize,
    memory_baseline: ForwardMemoryBaseline,
    started_at: Instant,
}

impl LayerAtATimePrefillInFlight {
    pub(super) const fn prefill_start(&self) -> usize {
        self.prefill_start
    }

    pub(super) const fn prefill_token_count(&self) -> usize {
        self.prefill_end - self.prefill_start
    }
}

/// What one step of a layer-at-a-time prompt left.
pub(super) enum LayerAtATimePrefillStep {
    /// A layer finished and more remain.
    LayerFinished(WorkerPromptLayerProgress),
    /// The last layer finished: the prompt's tokens are read.
    PromptFinished {
        prompt_prefill_chunk_outcome: PromptPrefillChunkOutcome,
        prefill_start: usize,
        prefill_end: usize,
        started_at: Instant,
    },
}

impl Qwen3_5EngineState {
    /// Plans the prompt's spans, samples what is held, and embeds the spans,
    /// leaving the request ready to run its first layer.
    pub(super) fn begin_layer_at_a_time_prefill(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        layer_major_span_plan: LayerMajorSpanPlan,
    ) -> Result<(), PromptPrefillChunkAttemptError> {
        let plan = self.plan_prompt_prefill_chunk(
            active_request,
            prefill_start,
            prefill_end,
            Some(layer_major_span_plan),
        )?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunkCount, 1);
        let memory_baseline = self.begin_forward_memory_measurement()?;
        let span_to_fail_once = self.prompt_span_to_fail_once_for_tests.take();
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let spans = plan
            .layer_major_span_plan
            .as_ref()
            .map_or(&[][..], |span_plan| span_plan.spans.as_slice());
        let checkpoint_contract =
            plan.persistent_prompt_cache_block_token_count
                .map(|block_token_count| LayerMajorCheckpointContract {
                    boundary_tensor_count: model.decoder_cache_layout().boundary_tensor_count(),
                    block_token_count,
                });
        let mut layer_major_prefill_run = model
            .begin_layer_major_prefill(
                &active_request.input_token_ids[prefill_start..prefill_end],
                active_request.next_position_tokens,
                LayerMajorPrefillRequest {
                    spans,
                    checkpoint_contract,
                    should_materialize_logits: prefill_end == active_request.input_token_ids.len(),
                },
            )
            .map_err(prompt_prefill_errors::prefill_execution_error)?;
        if let Some((layer_index, span_index)) = span_to_fail_once {
            layer_major_prefill_run.fail_span_once_for_tests(layer_index, span_index);
        }
        active_request.engine_extension.layer_at_a_time_prefill =
            Some(LayerAtATimePrefillInFlight {
                layer_major_prefill_run,
                plan,
                prefill_start,
                prefill_end,
                memory_baseline,
                started_at: Instant::now(),
            });
        Ok(())
    }

    /// Runs the prompt's next layer. A layer that ran out of memory and was
    /// put back to its failed span stays in flight, so a redo runs that
    /// span again; any other failure drops the prompt in flight.
    pub(super) fn advance_layer_at_a_time_prefill(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<LayerAtATimePrefillStep, PromptPrefillChunkAttemptError> {
        let mut prefill_in_flight = active_request
            .engine_extension
            .layer_at_a_time_prefill
            .take()
            .ok_or_else(|| {
                super::fatal_engine_error("no layer-at-a-time prompt is in flight to advance")
            })?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let advance_outcome = model
            .advance_layer_major_prefill(
                &mut prefill_in_flight.layer_major_prefill_run,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            )
            .map_err(prompt_prefill_errors::prefill_execution_error);
        let layer_major_prefill_outcome = match advance_outcome {
            Ok(layer_major_prefill_outcome) => layer_major_prefill_outcome,
            Err(prompt_prefill_chunk_attempt_error) => {
                if ran_out_of_memory(&prompt_prefill_chunk_attempt_error)
                    && prefill_in_flight
                        .layer_major_prefill_run
                        .can_run_failed_span_again()
                {
                    active_request.engine_extension.layer_at_a_time_prefill =
                        Some(prefill_in_flight);
                }
                return Err(prompt_prefill_chunk_attempt_error);
            }
        };
        let Some(layer_major_prefill_outcome) = layer_major_prefill_outcome else {
            let prompt_layer_progress = WorkerPromptLayerProgress {
                completed_layer_count: u32::try_from(
                    prefill_in_flight
                        .layer_major_prefill_run
                        .completed_layer_count(),
                )
                .unwrap_or(u32::MAX),
                layer_count: model.config.layer_count(),
            };
            active_request.engine_extension.layer_at_a_time_prefill = Some(prefill_in_flight);
            return Ok(LayerAtATimePrefillStep::LayerFinished(
                prompt_layer_progress,
            ));
        };
        if let Some(final_prompt_logits) = layer_major_prefill_outcome.final_logits.as_ref() {
            let first_generated_token =
                active_request.build_generated_token(model, final_prompt_logits)?;
            active_request.set_pending_generated_token(first_generated_token);
        }
        let LayerAtATimePrefillInFlight {
            plan,
            prefill_start,
            prefill_end,
            memory_baseline,
            started_at,
            ..
        } = prefill_in_flight;
        let mut boundary_checkpoints = layer_major_prefill_outcome.boundary_checkpoints;
        crate::qwen3_5_core::prompt_cache_capture::record_persistent_prompt_cache_boundary_checkpoint(
            active_request,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;
        Ok(LayerAtATimePrefillStep::PromptFinished {
            prompt_prefill_chunk_outcome: PromptPrefillChunkOutcome {
                active_memory_bytes_before_forward: memory_baseline.active_memory_bytes,
                forward_chunk_elapsed_millis: u64::try_from(started_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
                measured_forward_token_count: plan.measured_forward_token_count,
                boundary_checkpoints,
            },
            prefill_start,
            prefill_end,
            started_at,
        })
    }

    /// Runs the next layer of the prompt in flight and reports it: how many
    /// layers are done, or, after the last, the prompt's tokens as read.
    pub(super) fn continue_layer_at_a_time_prefill(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let prefill_attempt_failure = active_request
            .engine_extension
            .layer_at_a_time_prefill
            .as_ref()
            .map(|prefill_in_flight| PrefillAttemptFailure {
                prefill_start: prefill_in_flight.prefill_start(),
                attempted_prefill_chunk_token_count: prefill_in_flight.prefill_token_count(),
                is_layer_major_attempt: true,
            })
            .ok_or_else(|| super::fatal_engine_error("no layer-at-a-time prompt is in flight"))?;
        match self.advance_layer_at_a_time_prefill(active_request) {
            Ok(LayerAtATimePrefillStep::LayerFinished(prompt_layer_progress)) => Ok(Some(
                self.prompt_layer_progress_token(active_request, prompt_layer_progress)?,
            )),
            Ok(LayerAtATimePrefillStep::PromptFinished {
                prompt_prefill_chunk_outcome,
                prefill_start,
                prefill_end,
                started_at,
            }) => self.complete_prefill_attempt(
                request_id,
                active_request,
                prefill_start,
                prefill_end,
                prompt_prefill_chunk_outcome,
                started_at,
            ),
            Err(prompt_prefill_chunk_attempt_error) => self.recover_failed_prefill_attempt(
                request_id,
                active_request,
                prefill_attempt_failure,
                prompt_prefill_chunk_attempt_error,
            ),
        }
    }

    /// Reports a finished layer: how many are done, with what MLX holds
    /// now. No token is finished yet, so none is reported as read.
    fn prompt_layer_progress_token(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prompt_layer_progress: WorkerPromptLayerProgress,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let mlx_memory_telemetry = match model.memory_snapshot_with_forward_peak().ok() {
            Some(mlx_memory_snapshot) => Some(
                super::prefill_advance::prefill_mlx_memory_telemetry_from_snapshot(
                    model,
                    active_request,
                    0,
                    mlx_memory_snapshot,
                )?,
            ),
            None => None,
        };
        let expert_residency_telemetry = mlx_memory_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.expert_residency_telemetry());
        Ok(GeneratedToken::PrefillProgress {
            processed_token_count: 0,
            elapsed_millis: 0,
            forward_prefill_chunk_elapsed_millis: 0,
            completed_prefill_chunk_tokens: 0,
            mlx_memory_telemetry,
            expert_residency_telemetry,
            expert_memory_mode: Some(model.observed_expert_memory_mode()),
            prompt_work_reuse: active_request.prompt_work_reuse,
            persistent_prompt_cache_diagnostics: active_request
                .persistent_prompt_cache_diagnostics
                .clone(),
            neural_engine_chunk: None,
            prompt_layer_progress: Some(prompt_layer_progress),
        })
    }
}
