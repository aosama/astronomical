//! Starts the next stretch of a prompt and finishes it once its last layer
//! has run.
//!
//! Every prompt is read one layer at a time over its spans
//! (`layer_at_a_time_prefill`). The order in this file is a correctness
//! contract:
//!
//! 1. Cut the spans from the cursor: the configured size, clamped at
//!    semantic control-span and durable astro SSD prompt cache boundaries. A stretch
//!    may publish several astro SSD prompt cache blocks, each from the state its span left.
//! 2. Read the stretch one layer at a time, reporting every measured step
//!    to the memory governor.
//! 3. Publish required astro SSD prompt cache state before advancing the in-memory cursor.
//! 4. Synchronize the tape and reclaim MLX buffer cache only above threshold.
//! 5. Emit progress only after all required state transitions succeeded.
//!
//! A capacity failure never retries smaller spans. The retired ladder halved
//! the executable chunk after each failure, which collapsed real requests into
//! dozens of forwards that re-paid complete-layer storage reads while making no
//! visible progress. Instead, a memory failure is redone with room leased
//! from the memory governor (`prefill_memory_recovery`): an expert model
//! runs its failed span again, and a prompt that cannot be put back to a
//! span is read again from the `astro_ssd_prompt_cache`. Either is terminal
//! only once the governor's attempts are spent.

use crate::qwen3_5_core::astro_ssd_prompt_cache_capture::{
    capture_astro_ssd_prompt_cache_blocks, capture_astro_ssd_prompt_cache_partial_tail_block,
};
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use std::time::Instant;

use ipc_protocol::RequestId;
use runtime_integration::{MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemorySnapshot};

use crate::{
    GeneratedToken, InferenceEngineError, PerformanceOperation, last_prefill_chunk_demand_weight,
};

use super::completed_forward_memory;
use super::engine_request::Qwen3_5EngineRequest;
use super::layer_at_a_time_prefill::PromptPrefillChunkOutcome;
use super::prefill_memory_recovery::PrefillAttemptFailure;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
impl Qwen3_5EngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        if active_request
            .engine_extension
            .layer_at_a_time_prefill
            .is_some()
        {
            return self.continue_layer_at_a_time_prefill(request_id, active_request);
        }
        // Target-only prefill consumes every prompt token and samples first
        // output from the last chunk.
        let prefill_end_exclusive = active_request.input_token_ids.len();
        let Some(PrefillAttempt {
            prefill_start,
            attempted_prefill_chunk_token_count,
            layer_major_span_plan,
        }) = self.next_prompt_spans_to_read(active_request, prefill_end_exclusive)
        else {
            return Ok(None);
        };
        log_prefill_attempt(
            request_id,
            prefill_start,
            attempted_prefill_chunk_token_count,
            &layer_major_span_plan,
        );
        self.open_expert_demand_window_for_prefill(
            active_request,
            prefill_start,
            attempted_prefill_chunk_token_count,
            prefill_end_exclusive,
        );
        let prefill_end = self
            .prefill_end_for_attempted_chunk(prefill_start, attempted_prefill_chunk_token_count)?;
        let prefill_attempt_failure = PrefillAttemptFailure {
            prefill_start,
            attempted_prefill_chunk_token_count,
        };
        let begin_outcome = self.begin_layer_at_a_time_prefill(
            active_request,
            prefill_start,
            prefill_end,
            layer_major_span_plan,
        );
        match begin_outcome {
            Ok(()) => self.continue_layer_at_a_time_prefill(request_id, active_request),
            Err(prompt_prefill_chunk_attempt_error) => self.recover_failed_prefill_attempt(
                request_id,
                active_request,
                prefill_attempt_failure,
                prompt_prefill_chunk_attempt_error,
            ),
        }
    }

    /// After a forward read its tokens: publishes the astro SSD prompt cache state,
    /// moves the cursor, measures the forward and reports the tokens read.
    pub(super) fn complete_prefill_attempt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        prompt_prefill_chunk_outcome: PromptPrefillChunkOutcome,
        forward_chunk_started_at: Instant,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // No post-chunk weight reset is needed: decode evidence ignores the
        // prefill multiplier entirely (see `demand_phase`), and the next
        // chunk's window-open sets its own weight before recording.
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunk_outcome.forward_chunk_elapsed_millis;
        self.publish_and_advance_durable_prefill_state(
            active_request,
            prefill_start,
            prefill_end,
            prompt_prefill_chunk_outcome.boundary_checkpoints,
        )?;
        self.reclaim_mlx_buffer_cache_after_prefill_chunk(
            request_id,
            active_request,
            prefill_start,
            prefill_end,
        )?;
        self.record_completed_forward_memory(
            active_request,
            prompt_prefill_chunk_outcome.measured_forward_token_count,
            prompt_prefill_chunk_outcome.active_memory_bytes_before_forward,
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

fn log_prefill_attempt(
    request_id: RequestId,
    prefill_start: usize,
    attempted_prefill_chunk_token_count: usize,
    layer_major_span_plan: &super::layer_major_span_plan::LayerMajorSpanPlan,
) {
    tracing::info!(
        request_id = request_id.value(),
        prefill_start,
        attempted_prefill_chunk_token_count,
        span_count = layer_major_span_plan.spans.len(),
        "reading prompt tokens one layer at a time"
    );
}

/// What the next prefill attempt runs: the spans from the cursor to where
/// the plan stops, read one layer at a time.
struct PrefillAttempt {
    prefill_start: usize,
    attempted_prefill_chunk_token_count: usize,
    layer_major_span_plan: super::layer_major_span_plan::LayerMajorSpanPlan,
}

impl Qwen3_5EngineState {
    /// The spans the next attempt reads. Every prompt is read one layer at
    /// a time over its spans, so a layer read from storage is read once.
    fn next_prompt_spans_to_read(
        &self,
        active_request: &super::engine_request::Qwen3_5EngineRequest,
        prefill_end_exclusive: usize,
    ) -> Option<PrefillAttempt> {
        let crate::astro_ssd_prompt_cache::ResolvedPrefillChunkSpan { prefill_start, .. } =
            self.resolve_next_prefill_chunk_span(active_request, prefill_end_exclusive)?;
        let layer_major_span_plan =
            self.layer_major_span_plan_for(active_request, prefill_start, prefill_end_exclusive)?;
        Some(PrefillAttempt {
            prefill_start,
            attempted_prefill_chunk_token_count: layer_major_span_plan.group_end_token
                - prefill_start,
            layer_major_span_plan,
        })
    }

    /// Resolves the next prompt-prefill chunk span through the shared pure
    /// resolver: the configured chunk end from the sizer, clamped at the
    /// durable astro SSD prompt cache boundary when one applies.
    fn resolve_next_prefill_chunk_span(
        &self,
        active_request: &super::engine_request::Qwen3_5EngineRequest,
        prefill_end_exclusive: usize,
    ) -> Option<crate::astro_ssd_prompt_cache::ResolvedPrefillChunkSpan> {
        let astro_ssd_prompt_cache_block_token_count = self
            .core
            .astro_ssd_prompt_cache
            .as_ref()
            .map(|astro_ssd_prompt_cache| {
                astro_ssd_prompt_cache
                    .model_contract_ref()
                    .block_token_count()
            })
            .unwrap_or(0);
        crate::astro_ssd_prompt_cache::resolve_prompt_prefill_chunk_span(
            active_request.prefill_cursor,
            prefill_end_exclusive,
            self.core
                .prompt_processing_chunk_sizer
                .next_prompt_processing_chunk_end(
                    active_request.prefill_cursor,
                    prefill_end_exclusive,
                ),
            self.core.astro_ssd_prompt_cache.is_some(),
            active_request.can_use_astro_ssd_prompt_cache,
            astro_ssd_prompt_cache_block_token_count,
        )
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
        // One prefill demand window per request: the first chunk opens it —
        // discarding the previous request's prompt evidence while lifetime
        // decode evidence persists — and every chunk sets its own weight so
        // earlier chunks record at weight one while the final chunk records
        // boosted, ranking prompt-tail routes by token density for the
        // prefill-to-decode handoff.
        if !active_request
            .engine_extension
            .has_opened_prefill_demand_window
        {
            if let Some(model) = self.core.model.as_ref() {
                model.begin_prefill_demand_window_for_residency();
            }
            active_request
                .engine_extension
                .has_opened_prefill_demand_window = true;
        }
        let prefill_chunk_demand_weight =
            if prefill_start + attempted_prefill_chunk_token_count >= prefill_end_exclusive {
                last_prefill_chunk_demand_weight(
                    u64::try_from(prefill_start).unwrap_or(u64::MAX),
                    u64::try_from(attempted_prefill_chunk_token_count).unwrap_or(u64::MAX),
                )
            } else {
                1
            };
        if let Some(model) = self.core.model.as_ref() {
            model.set_prefill_demand_assignment_weight_for_residency(prefill_chunk_demand_weight);
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

    /// Publishes durable astro SSD prompt cache blocks and the partial tail, then
    /// advances the request position and cursor. Publication precedes both
    /// moves so a storage failure leaves restartable state. Mirrors the
    /// inlined block.
    fn publish_and_advance_durable_prefill_state(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        boundary_checkpoints: Vec<crate::Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint>,
    ) -> Result<(), InferenceEngineError> {
        let prefill_token_count = prefill_end - prefill_start;
        let model = self.core.loaded_model()?;
        // Publication deliberately precedes `advance_position` and cursor update.
        // If storage fails, the request stops at its last durable parent rather
        // than continuing with an in-memory chain that restart cannot reproduce.
        if let Some(astro_ssd_prompt_cache) = self.core.astro_ssd_prompt_cache.as_ref() {
            if active_request.can_use_astro_ssd_prompt_cache && !boundary_checkpoints.is_empty() {
                capture_astro_ssd_prompt_cache_blocks(
                    self,
                    astro_ssd_prompt_cache,
                    model,
                    active_request,
                    prefill_start,
                    prefill_end,
                    boundary_checkpoints,
                )?;
            }
            capture_astro_ssd_prompt_cache_partial_tail_block(
                self,
                astro_ssd_prompt_cache,
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
        Ok(())
    }

    /// Retires the chunk tape and reclaims the MLX buffer cache above the
    /// IOGPU-visible threshold, logging both memory snapshots. Mirrors the
    /// inlined block.
    fn reclaim_mlx_buffer_cache_after_prefill_chunk(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        // Retire the chunk tape. Reclaim the MLX buffer cache only when it is
        // large enough to justify an IOGPU-visible clear.
        let memory_snapshot_before_end_of_prefill_chunk_cleanup =
            model.memory_snapshot_with_forward_peak().ok();
        let end_of_prefill_chunk_cleanup_started_at = Instant::now();
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::MlxBufferCacheCleanup,
                |_performance_attribution| {
                    model
                        .runtime()
                        .synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
                            MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
                        )
                },
            )
            .map_err(qwen3_5_runtime_error)?;
        let memory_snapshot_after_end_of_prefill_chunk_cleanup =
            model.memory_snapshot_with_forward_peak().ok();
        tracing::info!(
            request_id = request_id.value(),
            cleanup_stage = "end_of_prefill_chunk_cleanup",
            prefill_start,
            prefill_end,
            active_memory_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            mlx_buffer_cache_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.mlx_buffer_cache_bytes()),
            active_memory_bytes_after_cleanup = memory_snapshot_after_end_of_prefill_chunk_cleanup
                .as_ref()
                .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            mlx_buffer_cache_bytes_after_cleanup =
                memory_snapshot_after_end_of_prefill_chunk_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.mlx_buffer_cache_bytes()),
            runtime_active_memory_limit_bytes =
                model.runtime().memory_limits().active_memory_limit_bytes(),
            cleanup_elapsed_millis = end_of_prefill_chunk_cleanup_started_at
                .elapsed()
                .as_millis(),
            "cleared MLX buffer cache storage after prompt-processing chunk"
        );
        Ok(())
    }

    /// Measures one completed prompt forward and tells the memory governor.
    fn record_completed_forward_memory(
        &mut self,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
        measured_forward_token_count: usize,
        active_memory_bytes_before_forward: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        completed_forward_memory::measure_completed_forward(
            model,
            completed_forward_memory::MeasuredForward::prompt(measured_forward_token_count),
            active_request.request_decoder_state.payload_byte_count(),
            active_memory_bytes_before_forward,
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
        let model = self.core.loaded_model()?;
        let prefill_token_count = prefill_end - prefill_start;
        // Demand-selected pages are materialized once after all prompt chunks.
        // Rebuilding them at every barrier would repeatedly read the same model
        // payload while the demand histogram is still changing.
        let mlx_memory_snapshot = model.memory_snapshot_with_forward_peak().ok();
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
            astro_ssd_prompt_cache_diagnostics: active_request
                .astro_ssd_prompt_cache_diagnostics
                .clone(),
            neural_engine_chunk: None,
            prompt_layer_progress: None,
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
pub(super) fn prefill_mlx_memory_telemetry_from_snapshot(
    model: &Qwen3_5Model,
    active_request: &Qwen3_5EngineRequest,
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
    let mlx_buffer_cache_bytes = u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes())
        .map_err(|_| fatal_engine_error("MLX buffer cache memory bytes exceed the u64 range"))?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(|_| fatal_engine_error("MLX peak memory bytes exceed the u64 range"))?;
    Ok(crate::memory::telemetry::MlxMemoryTelemetry::new(
        active_memory_bytes,
        mlx_buffer_cache_bytes,
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
    // utilization split, composed from the memory governor's accounts.
    .with_memory_ceiling_utilization(
        model
            .memory_ceiling_utilization_for_breakdown(active_memory_bytes, active_memory_breakdown),
    )
    .with_memory_ledger_reading(model.latest_memory_ledger_reading()))
}
