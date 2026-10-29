//! Advances one user-visible prompt-processing chunk.
//!
//! The order in this file is a correctness contract:
//!
//! 1. Choose a deterministic fixed chunk from resolved configuration.
//! 2. Clamp it at semantic control-span and durable prompt-cache boundaries.
//! 3. Execute that unchanged chunk under adaptive memory admission.
//! 4. Publish required prompt-cache state before advancing the in-memory cursor.
//! 5. Synchronize the chunk tape and reclaim allocator cache only above threshold.
//! 6. Retain request-local capacity evidence; decode fills demand-selected pages after prefill.
//! 7. Emit progress only after all required state transitions succeeded.
//!
//! A capacity failure never retries a smaller chunk. The retired ladder halved
//! the executable chunk after each failure, which collapsed real requests into
//! dozens of forwards that re-paid complete-layer storage reads while making no
//! visible progress. Instead, the first memory failure of a text request sheds
//! retained expert residency and re-seeds from the persistent prompt cache
//! (`prefill_memory_recovery`); only a second failure is terminal.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemorySnapshot,
};

use crate::{
    GeneratedToken, InferenceEngineError, MemoryPhase, PerformanceOperation,
    last_prefill_chunk_demand_weight,
};

use super::completed_forward_memory;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_memory_recovery;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::qwen3_5_streaming::model::memory_admission;
impl Qwen3_5EngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
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
            attempted_prefill_chunk_token_count,
            ..
        } = prefill_chunk_span;
        self.refresh_prefill_residency_plan(active_request)?;
        let forward_chunk_started_at = Instant::now();
        tracing::info!(
            request_id = request_id.value(),
            prefill_start,
            attempted_prefill_chunk_token_count,
            "executing prompt-prefill chunk"
        );
        self.open_expert_demand_window_for_prefill(
            active_request,
            prefill_start,
            attempted_prefill_chunk_token_count,
            prefill_end_exclusive,
        );
        let prefill_end = self
            .prefill_end_for_attempted_chunk(prefill_start, attempted_prefill_chunk_token_count)?;
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
        // Decode records its own route demand at plain assignment weight; the
        // boost belongs to the final prefill chunk's evidence only.
        if let Some(model) = self.model.as_ref() {
            model.set_expert_demand_assignment_weight_for_residency(1);
        }
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunk_outcome.forward_chunk_elapsed_millis;
        let boundary_checkpoints = prompt_prefill_chunk_outcome.boundary_checkpoints;
        let active_memory_bytes_before_growth =
            prompt_prefill_chunk_outcome.active_memory_bytes_before_growth;
        let retained_expert_payload_bytes_before_growth =
            prompt_prefill_chunk_outcome.retained_expert_payload_bytes_before_growth;
        let streamed_expert_page_bytes_before_growth =
            prompt_prefill_chunk_outcome.streamed_expert_page_bytes_before_growth;
        let adaptive_ram_growth_context = prompt_prefill_chunk_outcome.adaptive_ram_growth_context;
        let exact_temporary_workspace_bytes =
            prompt_prefill_chunk_outcome.exact_temporary_workspace_bytes;
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
            adaptive_ram_growth_context,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            exact_temporary_workspace_bytes,
            streamed_expert_page_bytes_before_growth,
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
}

impl Qwen3_5EngineState {
    /// Resolves the next prompt-prefill chunk span through the shared pure
    /// resolver: the configured chunk end from the sizer, clamped at the
    /// durable prompt-cache boundary when one applies.
    fn resolve_next_prefill_chunk_span(
        &self,
        active_request: &super::engine_request::Qwen3_5EngineRequest,
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

    /// Refreshes the phase-aware expert residency plan for the prefill phase.
    /// The activation reserve is operation-scoped (issue #644): the residency
    /// plan must survive the largest chunk this request can still attempt,
    /// not the whole prompt's imagined workspace. Mirrors the inlined block.
    fn refresh_prefill_residency_plan(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<(), InferenceEngineError> {
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .refresh_phase_aware_expert_residency_plan(
                MemoryPhase::Prefill,
                u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX),
                u64::try_from(
                    self.prompt_processing_chunk_sizer
                        .maximum_prompt_processing_chunk_size_tokens(),
                )
                .unwrap_or(u64::MAX),
                &mut active_request.performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)
    }

    /// Opens the request's one demand-evidence window for prefill and sets the
    /// final chunk's boosted assignment weight. Mirrors the inlined block.
    fn open_expert_demand_window_for_prefill(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        prefill_end_exclusive: usize,
    ) {
        // One demand-evidence window per request's prefill: earlier chunks
        // record at weight one and the final chunk records boosted, so the
        // prefill-to-decode handoff ranks prompt-tail routes by token density
        // instead of raw assignment count.
        if !active_request.has_opened_expert_demand_window {
            if let Some(model) = self.model.as_ref() {
                model.clear_expert_demand_for_residency();
            }
            active_request.has_opened_expert_demand_window = true;
        }
        let demand_assignment_weight =
            if prefill_start + attempted_prefill_chunk_token_count >= prefill_end_exclusive {
                last_prefill_chunk_demand_weight(
                    u64::try_from(prefill_start).unwrap_or(u64::MAX),
                    u64::try_from(attempted_prefill_chunk_token_count).unwrap_or(u64::MAX),
                )
            } else {
                1
            };
        if let Some(model) = self.model.as_ref() {
            model.set_expert_demand_assignment_weight_for_residency(demand_assignment_weight);
        }
    }

    /// The exclusive chunk end for one attempted span; overflow is a fatal
    /// generation error, exactly as the inlined checked_add was.
    fn prefill_end_for_attempted_chunk(
        &self,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
    ) -> Result<usize, InferenceEngineError> {
        prefill_start
            .checked_add(attempted_prefill_chunk_token_count)
            .ok_or_else(|| fatal_engine_error("prefill chunk end overflowed"))
    }

    /// Dispatches one failed chunk attempt: engine errors propagate; every
    /// memory failure either triggers the one-shot prefill memory recovery
    /// (recurring into the advance) or is terminal. Mirrors the inlined
    /// dispatch.
    fn handle_prompt_prefill_chunk_attempt_failure(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        prompt_prefill_chunk_attempt_error: PromptPrefillChunkAttemptError,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let prefill_memory_failure_error = match prompt_prefill_chunk_attempt_error {
            PromptPrefillChunkAttemptError::Engine(generation_error) => {
                return Err(generation_error);
            }
            PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded {
                decision: crate::PagedForwardAdmissionDecision::Reject { .. },
                reason,
            } => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    reason = %reason,
                    "prefill admission rejected the configured chunk"
                );
                memory_admission::invalid_request_error(format!(
                    "configured prefill chunk of {attempted_prefill_chunk_token_count} tokens cannot fit under the MLX ceiling: {reason}"
                ))
            }
            PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded { decision, reason } => {
                return Err(memory_admission::invalid_request_error(format!(
                    "unexpected paged forward admission decision {decision:?}: {reason}"
                )));
            }
            PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            } => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    active_memory_bytes,
                    attempted_allocation_bytes,
                    allowed_active_memory_bytes,
                    "prefill chunk forward exceeded the MLX active-memory limit"
                );
                memory_admission::invalid_request_error(format!(
                    "configured prefill chunk of {attempted_prefill_chunk_token_count} tokens exceeded the MLX active-memory limit: {active_memory_bytes} active bytes, {attempted_allocation_bytes} requested bytes, {allowed_active_memory_bytes} allowed bytes"
                ))
            }
            PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { reason } => {
                tracing::warn!(
                    request_id = request_id.value(),
                    prefill_start,
                    attempted_prefill_chunk_token_count,
                    reason = %reason,
                    "prefill chunk forward exhausted graphics-processor memory"
                );
                memory_admission::invalid_request_error(format!(
                    "configured prefill chunk of {attempted_prefill_chunk_token_count} tokens exhausted GPU memory: {reason}"
                ))
            }
        };
        if !prefill_memory_recovery::should_attempt_prefill_memory_recovery(
            active_request.completed_prefill_memory_recovery_attempt_count,
            active_request.has_visual_inputs,
        ) {
            return Err(prefill_memory_failure_error);
        }
        self.recover_prefill_after_memory_exhaustion(request_id, active_request, prefill_start)?;
        // Recursion depth is bounded by the one-recovery-per-request
        // counter incremented inside the recovery.
        self.advance_prompt_prefill_if_pending(request_id, active_request)
    }

    /// Publishes durable prompt-cache blocks and the partial tail, then
    /// advances the request position and cursor. Publication precedes both
    /// moves so a storage failure leaves restartable state. Mirrors the
    /// inlined block.
    fn publish_and_advance_durable_prefill_state(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        boundary_checkpoints: Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    ) -> Result<(), InferenceEngineError> {
        let prefill_token_count = prefill_end - prefill_start;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        adaptive_ram_growth_context: crate::AdaptiveRamGrowthContext,
        active_memory_bytes_before_growth: usize,
        retained_expert_payload_bytes_before_growth: u64,
        exact_temporary_workspace_bytes: usize,
        streamed_expert_page_bytes_before_growth: u64,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        completed_forward_memory::collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            true,
            model,
            active_request.request_context_extent(),
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
        active_request: &super::engine_request::Qwen3_5EngineRequest,
        request_id: RequestId,
        prefill_start: usize,
        prefill_end: usize,
        forward_chunk_started_at: Instant,
        forward_chunk_elapsed_millis: u64,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
            completed_prefill_chunk_tokens: u32::try_from(prefill_token_count).map_err(|_| {
                fatal_engine_error("completed_prefill_chunk_tokens exceeds the u32 range")
            })?,
            mlx_memory_telemetry,
            expert_residency_telemetry: expert_residency,
            expert_memory_mode: Some(model.observed_expert_memory_mode()),
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
/// `generated_token_emission`.
fn prefill_mlx_memory_telemetry_from_snapshot(
    model: &Qwen3_5Model,
    active_request: &Qwen3_5EngineRequest,
    prefill_token_count: usize,
    mlx_memory_snapshot: MlxMemorySnapshot,
) -> Result<crate::memory::telemetry::MlxMemoryTelemetry, InferenceEngineError> {
    let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(|_| fatal_engine_error("MLX active memory bytes exceed the u64 range"))?;
    let active_memory_breakdown = model.active_memory_breakdown(
        &active_request.request_decoder_state,
        0,
        active_memory_bytes,
        0,
    );
    let allocator_cache_memory_bytes =
        u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).map_err(|_| {
            fatal_engine_error("MLX allocator-cache memory bytes exceed the u64 range")
        })?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(|_| fatal_engine_error("MLX peak memory bytes exceed the u64 range"))?;
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
