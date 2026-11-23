//! Advances one user-visible prompt-processing chunk.
//!
//! The order in this file is a correctness contract:
//!
//! 1. Choose a deterministic fixed chunk from resolved configuration.
//! 2. Clamp it at semantic control-span and durable astro SSD prompt cache boundaries.
//! 3. Ask the memory governor whether that chunk fits, then execute exactly it.
//! 4. Publish required astro SSD prompt cache state before advancing the in-memory cursor.
//! 5. Synchronize the chunk tape and reclaim MLX buffer cache only above threshold.
//! 6. Retain request-local capacity evidence; decode fills demand-selected pages after prefill.
//! 7. Emit progress only after all required state transitions succeeded.
//!
//! A stable-footprint miss, a transient-peak miss, and an execution
//! out-of-memory failure are terminal for the resident engine: the request
//! reports the typed cause and moves to the streaming engine. The chunk is
//! never shrunk to make a forward fit. Halving ladders on both engines turned
//! one clear capacity decision into many smaller forwards that each re-paid
//! every per-chunk fixed cost, so throughput collapsed exactly when memory
//! was tight; the configured chunk size is the only prefill chunk size.

use crate::qwen3_5_core::shared_execution::prompt_reading_steps::{
    self, PromptStretch, PromptStretchTiming,
};
use std::time::{Duration, Instant};

use crate::{
    GeneratedToken, InferenceEngineError, ResidentForwardAdmissionRejection,
    StreamingEngineRequirement,
};
use ipc_protocol::RequestId;

use super::super::engine_request::Qwen3_5EngineRequest;
use super::Qwen3_5EngineState;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use crate::qwen3_5_resident::model::request_errors;
impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident::inference_execution) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // Target-only prefill consumes every prompt token and samples first
        // output from the last chunk.
        let prefill_end_exclusive = active_request.input_token_ids.len();
        let Some(prefill_chunk_span) = prompt_reading_steps::resolve_next_prefill_chunk_span(
            self,
            active_request,
            prefill_end_exclusive,
        ) else {
            return Ok(None);
        };
        let engine_core::astro_ssd_prompt_cache::ResolvedPrefillChunkSpan {
            prefill_start,
            requested_prefill_chunk_end: prefill_end,
            attempted_prefill_chunk_token_count,
        } = prefill_chunk_span;
        let read_stretch = PromptStretch {
            prefill_start,
            prefill_end,
        };
        let forward_chunk_started_at = Instant::now();
        tracing::info!(
            request_id = request_id.value(),
            prefill_start,
            attempted_prefill_chunk_token_count,
            "executing prompt-prefill chunk"
        );
        let prompt_prefill_chunk_outcome =
            match self.execute_prompt_prefill_chunk(active_request, prefill_start, prefill_end) {
                Ok(prompt_prefill_chunk_outcome) => prompt_prefill_chunk_outcome,
                Err(prompt_prefill_chunk_attempt_error) => {
                    return self.handle_prompt_prefill_chunk_attempt_failure(
                        request_id,
                        active_request,
                        prefill_start,
                        attempted_prefill_chunk_token_count,
                        prompt_prefill_chunk_attempt_error,
                    );
                }
            };
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunk_outcome.forward_chunk_elapsed_millis;
        let neural_engine_chunk = self.note_neural_engine_chunk_finished(
            prefill_start,
            prefill_end,
            forward_chunk_elapsed_millis,
        );
        prompt_reading_steps::publish_and_advance_durable_prefill_state(
            self,
            active_request,
            read_stretch,
            prompt_prefill_chunk_outcome.boundary_checkpoints,
        )?;
        prompt_reading_steps::reclaim_mlx_buffer_cache_after_prefill_chunk(
            self,
            request_id,
            active_request,
            read_stretch,
        )?;
        self.report_completed_prompt_chunk(
            active_request,
            prompt_prefill_chunk_outcome.admitted_forward,
        )?;
        Ok(Some(prompt_reading_steps::prefill_progress_token(
            self,
            active_request,
            request_id,
            read_stretch,
            PromptStretchTiming {
                forward_chunk_started_at,
                forward_chunk_elapsed_millis,
            },
            neural_engine_chunk,
        )?))
    }

    fn note_neural_engine_chunk_finished(
        &self,
        prefill_start: usize,
        prefill_end: usize,
        forward_chunk_elapsed_millis: u64,
    ) -> Option<ipc_protocol::NeuralEngineChunkReport> {
        self.core.model.as_ref()?.note_neural_engine_chunk_finished(
            prefill_start,
            prefill_end - prefill_start,
            Duration::from_millis(forward_chunk_elapsed_millis),
        )
    }

    /// Dispatches one failed resident chunk attempt. Engine errors propagate;
    /// every admission rejection demotes to the streaming engine with a typed
    /// cause and a failure description; the memory failures carry their own
    /// terminal descriptions. Mirrors the inlined dispatch.
    fn handle_prompt_prefill_chunk_attempt_failure(
        &mut self,
        request_id: RequestId,
        _active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        prompt_prefill_chunk_attempt_error: PromptPrefillChunkAttemptError,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // This engine's failure path never touches the request: every
        // failure moves the whole request to the streaming engine.
        match prompt_prefill_chunk_attempt_error {
            PromptPrefillChunkAttemptError::Engine(generation_error) => Err(generation_error),
            PromptPrefillChunkAttemptError::ForwardAdmissionRejected {
                rejection:
                    ResidentForwardAdmissionRejection::RequireStreamingEngine {
                        cause: prefill_admission_cause,
                    },
                reason,
            } => self.demote_prefill_to_streaming_engine(
                request_id,
                prefill_start,
                attempted_prefill_chunk_token_count,
                prefill_admission_cause,
                &reason,
            ),
            PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            } => Err(self.active_memory_limit_prefill_failure(
                request_id,
                prefill_start,
                attempted_prefill_chunk_token_count,
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            )),
            PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { reason } => {
                Err(self.graphics_processor_memory_exhausted_prefill_failure(
                    request_id,
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    &reason,
                ))
            }
        }
    }

    /// Demotes a rejected resident chunk to the streaming engine with the
    /// typed cause. The transient-peak, stable-footprint, request-context,
    /// forward-allocation, and GPU-exhausted causes each carry their own
    /// terminal description and warn log, exactly as the inlined arms did.
    #[allow(clippy::too_many_arguments)]
    fn demote_prefill_to_streaming_engine(
        &self,
        request_id: RequestId,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        prefill_admission_cause: StreamingEngineRequirement,
        reason: &str,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        match prefill_admission_cause {
            StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk { .. }
            | StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk { .. } => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    reason = %reason,
                    "prefill admission rejected the transient peak of the configured resident chunk"
                );
                Err(request_errors::resident_capacity_error(prefill_admission_cause,
                    super::prefill_terminal_failure_descriptions::transient_peak_shortfall_failure_description(
                        attempted_prefill_chunk_token_count,
                        reason,
                    ),
                ))
            }
            // A stable-footprint miss is terminal: no smaller forward reduces
            // memory that stays resident after the forward completes.
            StreamingEngineRequirement::StableShortfall { .. } => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    reason = %reason,
                    "prefill admission rejected the stable footprint of the attempted chunk"
                );
                Err(request_errors::resident_capacity_error(prefill_admission_cause,
                    super::prefill_terminal_failure_descriptions::stable_footprint_shortfall_failure_description(
                        attempted_prefill_chunk_token_count,
                        reason,
                    ),
                ))
            }
            // Request-context shortfalls and runtime allocation failures
            // are terminal for the resident engine: no smaller chunk
            // changes the request's own workspace or an allocation that
            // already missed, so they demote to the streaming engine with
            // the typed cause instead of retrying.
            StreamingEngineRequirement::RequestContextShortfall { .. }
            | StreamingEngineRequirement::ForwardAllocationShortfall { .. }
            | StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    reason = %reason,
                    "prefill admission rejected the request context or a runtime allocation"
                );
                Err(request_errors::resident_capacity_error(prefill_admission_cause,
                    super::prefill_terminal_failure_descriptions::request_context_or_allocation_shortfall_failure_description(
                        attempted_prefill_chunk_token_count,
                        reason,
                    ),
                ))
            }
        }
    }

    /// Builds the active-memory-limit failure with its shortfall bytes and
    /// terminal description. Mirrors the inlined arm, including the warn log.
    #[allow(clippy::too_many_arguments)]
    fn active_memory_limit_prefill_failure(
        &self,
        request_id: RequestId,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
    ) -> InferenceEngineError {
        tracing::warn!(
            request_id = request_id.value(),
            prefill_start,
            attempted_prefill_chunk_token_count,
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            "prefill chunk forward exceeded the MLX active-memory limit"
        );
        request_errors::resident_capacity_error(StreamingEngineRequirement::ForwardAllocationShortfall {
                shortfall_bytes: super::prefill_terminal_failure_descriptions::active_memory_limit_shortfall_bytes(
                    active_memory_bytes,
                    attempted_allocation_bytes,
                    allowed_active_memory_bytes,
                ),
            },
            super::prefill_terminal_failure_descriptions::active_memory_limit_exceeded_failure_description(
                attempted_prefill_chunk_token_count,
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            ),
        )
    }

    /// Builds the GPU-memory-exhausted failure with its terminal description.
    /// Mirrors the inlined arm, including the warn log.
    fn graphics_processor_memory_exhausted_prefill_failure(
        &self,
        request_id: RequestId,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        reason: &str,
    ) -> InferenceEngineError {
        tracing::warn!(
            request_id = request_id.value(),
            prefill_start,
            attempted_prefill_chunk_token_count,
            reason = %reason,
            "prefill chunk forward exhausted graphics-processor memory"
        );
        request_errors::resident_capacity_error(StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted,
            super::prefill_terminal_failure_descriptions::graphics_processor_memory_exhausted_failure_description(
                attempted_prefill_chunk_token_count,
                reason,
            ),
        )
    }

    /// Tells the memory governor what the chunk needed. Every completed
    /// chunk counts, the short ones at `astro_ssd_prompt_cache` boundaries and prompt tails
    /// included, since the governor keeps the largest figure per chunk size.
    fn report_completed_prompt_chunk(
        &mut self,
        active_request: &Qwen3_5EngineRequest,
        admitted_forward: super::forward_memory::AdmittedForward,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        super::forward_memory::report_completed_forward(
            model,
            admitted_forward,
            active_request.request_decoder_state.payload_byte_count(),
        )?;
        Ok(())
    }
}
