//! Advances one user-visible prompt-processing chunk.
//!
//! The order in this file is a correctness contract:
//!
//! 1. Choose a deterministic fixed chunk from resolved configuration.
//! 2. Clamp it at semantic control-span and durable prompt-cache boundaries.
//! 3. Execute exactly that chunk under adaptive memory admission.
//! 4. Publish required prompt-cache state before advancing the in-memory cursor.
//! 5. Synchronize the chunk tape and reclaim allocator cache only above threshold.
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

use std::time::Instant;

use crate::{
    GeneratedToken, InferenceEngineError, MemoryPhase, PerformanceCounter, PerformanceOperation,
    ResidentForwardAdmissionRejection, StreamingEngineRequirement,
};
use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemorySnapshot,
};

use super::super::engine_request::Qwen3_5EngineRequest;
use super::super::generation_session_lifecycle::generation_error_constructors::completed_prefill_chunk_tokens_exceed_u32_range_error;
use super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error;
use super::super::generation_session_lifecycle::generation_error_constructors::mlx_active_memory_bytes_exceed_u64_range_error;
use super::super::generation_session_lifecycle::generation_error_constructors::mlx_allocator_cache_memory_bytes_exceed_u64_range_error;
use super::super::generation_session_lifecycle::generation_error_constructors::mlx_peak_memory_bytes_exceed_u64_range_error;
use super::completed_forward_memory;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::model::memory_admission;
impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident::inference_execution) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // Target-only prefill consumes every prompt token and samples first
        // output from the last chunk.
        let prefill_end_exclusive = active_request.input_token_ids.len();
        let Some(prefill_chunk_span) =
            self.resolve_next_prefill_chunk_span(active_request, prefill_end_exclusive)
        else {
            return Ok(None);
        };
        let crate::persistent_cache::ResolvedPrefillChunkSpan {
            prefill_start,
            requested_prefill_chunk_end: prefill_end,
            attempted_prefill_chunk_token_count,
        } = prefill_chunk_span;
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
        let boundary_checkpoints = prompt_prefill_chunk_outcome.boundary_checkpoints;
        self.publish_and_advance_durable_prefill_state(
            active_request,
            prefill_start,
            prefill_end,
            boundary_checkpoints,
        )?;
        self.reclaim_allocator_cache_after_prefill_chunk(
            request_id,
            active_request,
            prefill_start,
            prefill_end,
        )?;
        self.record_completed_forward_memory(
            active_request,
            prompt_prefill_chunk_outcome.adaptive_ram_growth_context,
            prompt_prefill_chunk_outcome.active_memory_bytes_before_growth,
            prompt_prefill_chunk_outcome.retained_expert_payload_bytes_before_growth,
            prompt_prefill_chunk_outcome.exact_temporary_workspace_bytes,
            prompt_prefill_chunk_outcome.streamed_expert_page_bytes_before_growth,
        )?;
        Ok(Some(self.prefill_progress_token(
            active_request,
            request_id,
            prefill_start,
            prefill_end,
            forward_chunk_started_at,
            forward_chunk_elapsed_millis,
        )?))
    }

    /// Resolves the next prompt-prefill chunk span through the shared pure
    /// resolver: the configured chunk end from the sizer, clamped at the
    /// durable prompt-cache boundary when one applies. The resident engine
    /// executes exactly the requested (possibly clamped) end.
    fn resolve_next_prefill_chunk_span(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_end_exclusive: usize,
    ) -> Option<crate::persistent_cache::ResolvedPrefillChunkSpan> {
        let persistent_prompt_cache_block_token_count = self
            .persistent_prompt_cache
            .as_ref()
            .map(|persistent_prompt_cache| {
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count()
            })
            .unwrap_or(0);
        crate::persistent_cache::resolve_prompt_prefill_chunk_span(
            active_request.prefill_cursor,
            prefill_end_exclusive,
            self.prompt_processing_chunk_sizer
                .next_prompt_processing_chunk_end(
                    active_request.prefill_cursor,
                    prefill_end_exclusive,
                ),
            self.persistent_prompt_cache.is_some(),
            active_request.can_use_persistent_prompt_cache,
            persistent_prompt_cache_block_token_count,
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
        // The resident failure path never touches the request: every failure
        // demotes the request wholesale. The parameter stays for dispatch
        // symmetry with the streaming twin's recovery path.
        let model = self
            .model
            .as_ref()
            .ok_or_else(engine_lost_loaded_model_error)?;
        match prompt_prefill_chunk_attempt_error {
            PromptPrefillChunkAttemptError::Engine(generation_error) => Err(generation_error),
            PromptPrefillChunkAttemptError::AdaptiveForwardAdmissionRejected {
                rejection:
                    ResidentForwardAdmissionRejection::RequireStreamingEngine {
                        cause: prefill_admission_cause,
                    },
                reason,
            } => self.demote_prefill_to_streaming_engine(
                model,
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
                model,
                request_id,
                prefill_start,
                attempted_prefill_chunk_token_count,
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            )),
            PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { reason } => {
                Err(self.graphics_processor_memory_exhausted_prefill_failure(
                    model,
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
        model: &Qwen3_5ResidentModel,
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
                Err(memory_admission::resident_capacity_error(
                    model,
                    prefill_admission_cause,
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
                Err(memory_admission::resident_capacity_error(
                    model,
                    prefill_admission_cause,
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
                Err(memory_admission::resident_capacity_error(
                    model,
                    prefill_admission_cause,
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
        model: &Qwen3_5ResidentModel,
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
        memory_admission::resident_capacity_error(
            model,
            StreamingEngineRequirement::ForwardAllocationShortfall {
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
        model: &Qwen3_5ResidentModel,
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
        memory_admission::resident_capacity_error(
            model,
            StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted,
            super::prefill_terminal_failure_descriptions::graphics_processor_memory_exhausted_failure_description(
                attempted_prefill_chunk_token_count,
                reason,
            ),
        )
    }

    /// Publishes durable prompt-cache blocks and the partial tail, then
    /// advances the request position and cursor. Publication precedes both
    /// moves so a storage failure leaves restartable state. Mirrors the
    /// inlined block.
    fn publish_and_advance_durable_prefill_state(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        boundary_checkpoints: Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(engine_lost_loaded_model_error)?;
        let prefill_token_count = prefill_end - prefill_start;
        // Publication deliberately precedes `advance_position` and cursor update.
        // If storage fails, the request stops at its last durable parent rather
        // than continuing with an in-memory chain that restart cannot reproduce.
        if let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() {
            if active_request.can_use_persistent_prompt_cache && !boundary_checkpoints.is_empty() {
                self.capture_persistent_prompt_cache_blocks(
                    persistent_prompt_cache,
                    model,
                    active_request,
                    prefill_start,
                    prefill_end,
                    boundary_checkpoints,
                )?;
            }
            self.capture_persistent_prompt_cache_partial_tail_block(
                persistent_prompt_cache,
                model,
                active_request,
                prefill_end,
            );
        }
        // Only durable publication success may move these frontiers. If capture
        // failed above, the function returned with both position and cursor at
        // their previous parent, allowing restart to reproduce valid state.
        active_request.advance_position(prefill_token_count)?;
        active_request.prefill_cursor = prefill_end;
        model.learn_context_state_payload(
            active_request.next_position_tokens,
            &active_request.request_decoder_state,
        );
        Ok(())
    }

    /// Retires the chunk tape and reclaims the allocator cache above the
    /// IOGPU-visible threshold, logging both memory snapshots. Mirrors the
    /// inlined block.
    fn reclaim_allocator_cache_after_prefill_chunk(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(engine_lost_loaded_model_error)?;
        // Retire the chunk tape. Reclaim the allocator cache only when it is
        // large enough to justify an IOGPU-visible clear.
        let memory_snapshot_before_end_of_prefill_chunk_cleanup =
            model.runtime().memory_snapshot().ok();
        let end_of_prefill_chunk_cleanup_started_at = Instant::now();
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    model
                        .runtime()
                        .synchronize_gpu_stream_and_reclaim_allocator_cache_above_threshold(
                            ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
                        )
                },
            )
            .map_err(qwen3_5_runtime_error)?;
        let memory_snapshot_after_end_of_prefill_chunk_cleanup =
            model.runtime().memory_snapshot().ok();
        tracing::info!(
            request_id = request_id.value(),
            cleanup_stage = "end_of_prefill_chunk_cleanup",
            prefill_start,
            prefill_end,
            active_memory_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            allocator_cache_memory_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.allocator_cache_memory_bytes()),
            active_memory_bytes_after_cleanup = memory_snapshot_after_end_of_prefill_chunk_cleanup
                .as_ref()
                .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            allocator_cache_memory_bytes_after_cleanup =
                memory_snapshot_after_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.allocator_cache_memory_bytes()),
            runtime_active_memory_limit_bytes =
                model.runtime().memory_limits().active_memory_limit_bytes(),
            cleanup_elapsed_millis = end_of_prefill_chunk_cleanup_started_at
                .elapsed()
                .as_millis(),
            "cleared MLX allocator-cache storage after prompt-processing chunk"
        );
        Ok(())
    }

    /// Records one completed forward's memory evidence for adaptive RAM
    /// growth. Every completed forward is valid model-local transient
    /// evidence: cache boundaries and terminal tails deliberately produce
    /// smaller chunks; discarding them would let a long request complete real
    /// prefill work without ever teaching admission about its workspace.
    /// Mirrors the inlined collection call.
    #[allow(clippy::too_many_arguments)]
    fn record_completed_forward_memory(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        adaptive_ram_growth_context: crate::AdaptiveRamGrowthContext,
        active_memory_bytes_before_growth: usize,
        retained_expert_payload_bytes_before_growth: u64,
        exact_temporary_workspace_bytes: usize,
        streamed_expert_page_bytes_before_growth: u64,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(engine_lost_loaded_model_error)?;
        // Every completed forward is valid model-local transient evidence. Cache
        // boundaries and terminal tails deliberately produce smaller chunks; if
        // they are discarded, a long OpenCode request can complete real prefill
        // work without ever teaching subsequent admission about its workspace.
        let should_retain_adaptive_ram_growth_observation = true;
        completed_forward_memory::collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            should_retain_adaptive_ram_growth_observation,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            exact_temporary_workspace_bytes,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
        Ok(())
    }

    /// Builds the prefill-phase MLX telemetry and the public prefill progress
    /// token for one completed chunk. Mirrors the inlined tail.
    #[allow(clippy::too_many_arguments)]
    fn prefill_progress_token(
        &self,
        active_request: &Qwen3_5EngineRequest,
        request_id: RequestId,
        prefill_start: usize,
        prefill_end: usize,
        forward_chunk_started_at: Instant,
        forward_chunk_elapsed_millis: u64,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(engine_lost_loaded_model_error)?;
        let prefill_token_count = prefill_end - prefill_start;
        // Demand-selected pages are materialized once after all prompt chunks.
        // Rebuilding them at every barrier would repeatedly read the same model
        // payload while the demand histogram is still changing.
        let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
        let prefill_chunk_elapsed_millis = forward_chunk_started_at.elapsed().as_millis() as u64;
        tracing::trace!(
            request_id = request_id.value(),
            prefill_start_token = prefill_start,
            prefill_end_token = prefill_end,
            prefill_token_count,
            forward_chunk_elapsed_millis,
            prefill_chunk_elapsed_millis,
            "completed synchronous Qwen3.5 prompt-processing chunk"
        );
        let mlx_memory_telemetry = match mlx_memory_snapshot {
            Some(mlx_memory_snapshot) => Some(prefill_mlx_memory_telemetry_from_snapshot(
                model,
                active_request,
                prefill_token_count,
                mlx_memory_snapshot,
            )?),
            None => None,
        };
        let expert_residency = mlx_memory_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.expert_residency_telemetry());
        Ok(GeneratedToken::PrefillProgress {
            processed_token_count: prefill_token_count as u32,
            elapsed_millis: prefill_chunk_elapsed_millis,
            forward_prefill_chunk_elapsed_millis: forward_chunk_elapsed_millis,
            completed_prefill_chunk_tokens: u32::try_from(prefill_token_count)
                .map_err(completed_prefill_chunk_tokens_exceed_u32_range_error)?,
            mlx_memory_telemetry,
            expert_residency_telemetry: expert_residency,
            expert_memory_mode: Some(model.expert_memory_mode()),
            prompt_work_reuse: active_request.prompt_work_reuse,
            persistent_prompt_cache_diagnostics: active_request
                .persistent_prompt_cache_diagnostics
                .clone(),
        })
    }
}

/// Builds the prefill-phase MLX memory telemetry from one post-chunk memory
/// snapshot.
///
/// Extracted from the prefill advance step so each snapshot conversion reads
/// as a named value instead of a nested `try_from`/`map_err` chain inside an
/// `Option::map` closure; the decode-phase sibling lives in
/// `decode_lifecycle/generated_token_emission`. The resident generation path
/// reports its range failures through the shared named error constructors.
fn prefill_mlx_memory_telemetry_from_snapshot(
    model: &Qwen3_5ResidentModel,
    active_request: &Qwen3_5EngineRequest,
    prefill_token_count: usize,
    mlx_memory_snapshot: MlxMemorySnapshot,
) -> Result<crate::memory::telemetry::MlxMemoryTelemetry, InferenceEngineError> {
    let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(mlx_active_memory_bytes_exceed_u64_range_error)?;
    let active_memory_breakdown = model.active_memory_breakdown(
        &active_request.request_decoder_state,
        0,
        active_memory_bytes,
        0,
    );
    let allocator_cache_memory_bytes =
        u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes())
            .map_err(mlx_allocator_cache_memory_bytes_exceed_u64_range_error)?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(mlx_peak_memory_bytes_exceed_u64_range_error)?;
    let prompt_token_count =
        u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
    let chunk_token_count = u64::try_from(prefill_token_count).unwrap_or(u64::MAX);
    Ok(crate::memory::telemetry::MlxMemoryTelemetry::new(
        active_memory_bytes,
        allocator_cache_memory_bytes,
        peak_memory_bytes,
        active_memory_breakdown,
    )
    // The claim derives from the breakdown of this exact snapshot
    // so the progress pair cannot straddle a promotion or demote
    // (issue #337).
    .with_expert_residency_telemetry(
        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
    )
    // Issue #510: the prefill progress sample carries the same
    // utilization split, planned against the full prompt so the
    // context reserve reflects the request the user submitted.
    // The activation reserve is operation-scoped (issue #644):
    // it plans this chunk's own workspace, not the prompt.
    .with_memory_ceiling_utilization(model.memory_ceiling_utilization_for_breakdown(
        crate::MemoryPhase::Prefill,
        prompt_token_count,
        chunk_token_count,
        active_memory_bytes,
        active_memory_breakdown,
    )))
}
