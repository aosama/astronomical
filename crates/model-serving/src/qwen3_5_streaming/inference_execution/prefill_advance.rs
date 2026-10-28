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
    #[expect(clippy::too_many_lines)]
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // Target-only prefill consumes every prompt token and samples first
        // output from the last chunk.
        let prefill_end_exclusive = active_request.input_token_ids.len();
        if active_request.prefill_cursor >= prefill_end_exclusive {
            return Ok(None);
        }
        let prefill_start = active_request.prefill_cursor;
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .refresh_phase_aware_expert_residency_plan(
                MemoryPhase::Prefill,
                u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX),
                // The activation reserve is operation-scoped (issue #644): the
                // residency plan must survive the largest chunk this request can
                // still attempt, not the whole prompt's imagined workspace.
                u64::try_from(
                    self.prompt_processing_chunk_sizer
                        .maximum_prompt_processing_chunk_size_tokens(),
                )
                .unwrap_or(u64::MAX),
                &mut active_request.performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
        let configured_prompt_processing_chunk_end = self
            .prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(active_request.prefill_cursor, prefill_end_exclusive);
        let requested_prefill_chunk_end = configured_prompt_processing_chunk_end;
        // Required publication is synchronous, so do not let one forward cross
        // multiple durable boundaries. A successful forward produces one exact
        // checkpoint, publishes it, and only then advances the request cursor.
        // One-boundary-per-forward keeps publication workspace exact when a
        // small cache block would otherwise sit inside a 2,048-token chunk many
        // times. When the configured chunk crosses at most one durable boundary,
        // clamping only splits a later complete-layer SSD stream (1,792 then a
        // 561-token stub) without reducing publication count.
        let requested_prefill_chunk_end = if self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
        {
            let persistent_prompt_cache_block_token_count = self
                .persistent_prompt_cache
                .as_ref()
                .map(|persistent_prompt_cache| {
                    persistent_prompt_cache
                        .model_contract_ref()
                        .block_token_count()
                })
                .unwrap_or(0);
            // One shared function applies both clamps (at most one durable
            // block publication per forward; terminal chunks stop one token
            // short so the captured partial tail stays probeable). Keeping the
            // composition in the tested pure layer is deliberate: the boundary
            // clamp was silently discarded once by an innocent refactor.
            crate::persistent_prompt_cache_clamped_prefill_chunk_end(
                prefill_start,
                requested_prefill_chunk_end,
                prefill_end_exclusive,
                persistent_prompt_cache_block_token_count,
            )
        } else {
            requested_prefill_chunk_end
        };
        let forward_chunk_started_at = Instant::now();
        let attempted_prefill_chunk_token_count = requested_prefill_chunk_end - prefill_start;
        tracing::info!(
            request_id = request_id.value(),
            prefill_start,
            attempted_prefill_chunk_token_count,
            "executing prompt-prefill chunk"
        );
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
        let prefill_end = prefill_start
            .checked_add(attempted_prefill_chunk_token_count)
            .ok_or_else(|| fatal_engine_error("prefill chunk end overflowed"))?;
        let prompt_prefill_chunk_outcome = match self.execute_prompt_prefill_chunk(
            active_request,
            prefill_start,
            prefill_end,
        ) {
            Ok(prompt_prefill_chunk_outcome) => prompt_prefill_chunk_outcome,
            Err(prompt_prefill_chunk_attempt_error) => {
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
                    PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded {
                        decision,
                        reason,
                    } => {
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
                self.recover_prefill_after_memory_exhaustion(
                    request_id,
                    active_request,
                    prefill_start,
                )?;
                // Recursion depth is bounded by the one-recovery-per-request
                // counter incremented inside the recovery.
                return self.advance_prompt_prefill_if_pending(request_id, active_request);
            }
        };
        // Decode records its own route demand at plain assignment weight; the
        // boost belongs to the final prefill chunk's evidence only.
        if let Some(model) = self.model.as_ref() {
            model.set_expert_demand_assignment_weight_for_residency(1);
        }
        let active_memory_bytes_before_growth =
            prompt_prefill_chunk_outcome.active_memory_bytes_before_growth;
        let retained_expert_payload_bytes_before_growth =
            prompt_prefill_chunk_outcome.retained_expert_payload_bytes_before_growth;
        let streamed_expert_page_bytes_before_growth =
            prompt_prefill_chunk_outcome.streamed_expert_page_bytes_before_growth;
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunk_outcome.forward_chunk_elapsed_millis;
        let adaptive_ram_growth_context = prompt_prefill_chunk_outcome.adaptive_ram_growth_context;
        let exact_temporary_workspace_bytes =
            prompt_prefill_chunk_outcome.exact_temporary_workspace_bytes;
        let boundary_checkpoints = prompt_prefill_chunk_outcome.boundary_checkpoints;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_token_count = prefill_end - prefill_start;
        // Every completed forward is valid model-local transient evidence. Cache
        // boundaries and terminal tails deliberately produce smaller chunks; if
        // they are discarded, a long OpenCode request can complete real prefill
        // work without ever teaching subsequent admission about its workspace.
        let should_retain_adaptive_ram_growth_observation = true;
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
        completed_forward_memory::collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            should_retain_adaptive_ram_growth_observation,
            model,
            active_request.request_context_extent(),
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            exact_temporary_workspace_bytes,
            streamed_expert_page_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
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
        Ok(Some(GeneratedToken::PrefillProgress {
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
        }))
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
