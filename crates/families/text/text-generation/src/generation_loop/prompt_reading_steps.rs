//! Steps every text engine take around one stretch of prompt: choosing
//! where it ends, publishing what it leaves to the `astro_ssd_prompt_cache`,
//! reclaiming the `mlx_buffer_cache`, and reporting the stretch as read.
//!
//! How a stretch is read and what a failed read does stay with each engine.
//! The Qwen3.5 text engine reads it one layer at a time and redoes a failed
//! span with room leased from its memory governor.
//!
//! The order of the steps is a correctness contract: publication to the
//! `astro_ssd_prompt_cache` comes before the cursor moves, and progress is
//! reported only after every state change succeeded.

use crate::model::TextServingModel;
use std::time::Instant;

use ipc_protocol::{NeuralEngineChunkReport, RequestId};
use runtime_integration::{MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemorySnapshot};

use crate::astro_ssd_prompt_cache_capture::{
    TextAstroSsdPromptCacheCapture, capture_astro_ssd_prompt_cache_blocks,
    capture_astro_ssd_prompt_cache_partial_tail_block,
};
use crate::decoder_state::RequestDecoderStateStack;
use crate::engine_errors::{
    fatal_engine_error, fatal_runtime_error, mlx_active_memory_bytes_exceed_u64_range_error,
    mlx_buffer_cache_bytes_exceed_u64_range_error, mlx_peak_memory_bytes_exceed_u64_range_error,
};
use crate::engine_request::TextEngineRequest;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::{
    AstroSsdPromptCacheBoundaryCheckpoint, GeneratedToken, InferenceEngineError,
    PerformanceOperation,
};
use engine_core::astro_ssd_prompt_cache::{
    ResolvedPrefillChunkSpan, resolve_prompt_prefill_chunk_span,
};

use super::execution_core::TextExecutionCoreOwner;
use crate::model::text_model::TextModel;

/// A stretch of prompt tokens, from `prefill_start` up to but excluding
/// `prefill_end`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptStretch {
    pub prefill_start: usize,
    pub prefill_end: usize,
}

impl PromptStretch {
    pub fn token_count(self) -> usize {
        self.prefill_end - self.prefill_start
    }
}

/// When a read stretch started and how long its forward took, for the
/// progress the worker shows.
#[derive(Clone, Copy, Debug)]
pub struct PromptStretchTiming {
    pub forward_chunk_started_at: Instant,
    pub forward_chunk_elapsed_millis: u64,
}

/// The next stretch to read: the configured chunk end from the sizer,
/// clamped at the durable `astro_ssd_prompt_cache` boundary when one applies.
/// None once the whole prompt is read.
pub fn resolve_next_prefill_chunk_span<Engine: TextExecutionCoreOwner>(
    engine: &Engine,
    active_request: &TextEngineRequest<Engine::RequestExtension>,
    prefill_end_exclusive: usize,
) -> Option<ResolvedPrefillChunkSpan> {
    let execution_core = engine.execution_core();
    let astro_ssd_prompt_cache_block_token_count = match &execution_core.astro_ssd_prompt_cache {
        Some(astro_ssd_prompt_cache) => astro_ssd_prompt_cache
            .model_contract_ref()
            .block_token_count(),
        None => 0,
    };
    resolve_prompt_prefill_chunk_span(
        active_request.prefill_cursor,
        prefill_end_exclusive,
        execution_core
            .prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(active_request.prefill_cursor, prefill_end_exclusive),
        execution_core.astro_ssd_prompt_cache.is_some(),
        active_request.can_use_astro_ssd_prompt_cache,
        astro_ssd_prompt_cache_block_token_count,
    )
}

/// Hands the stretch's `astro_ssd_prompt_cache` blocks and its partial tail to
/// the background writer, then advances the request position and cursor.
///
/// Blocks are queued in chain order, so the writer always writes a parent
/// before its children. A block that cannot be copied or queued ends saving for the rest of the
/// request instead of the request itself: the cache keeps the prefix saved so
/// far, which a later request restores and continues from, and the reply goes
/// on. Stopping the request instead turned one unsaved block into a failed
/// reply (Ternary Bonsai 2.5 in 0.2.136).
pub fn publish_and_advance_durable_prefill_state<Engine: TextAstroSsdPromptCacheCapture>(
    engine: &Engine,
    active_request: &mut TextEngineRequest<Engine::RequestExtension>,
    read_stretch: PromptStretch,
    boundary_checkpoints: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
) -> Result<(), InferenceEngineError> {
    let execution_core = engine.execution_core();
    let model = execution_core.loaded_model()?;
    if let Some(astro_ssd_prompt_cache) = execution_core.astro_ssd_prompt_cache.as_ref() {
        if active_request.can_use_astro_ssd_prompt_cache && !boundary_checkpoints.is_empty() {
            let capture_outcome = capture_astro_ssd_prompt_cache_blocks(
                engine,
                astro_ssd_prompt_cache,
                model,
                active_request,
                read_stretch.prefill_start,
                read_stretch.prefill_end,
                boundary_checkpoints,
            );
            stop_saving_after_a_failed_save(active_request, capture_outcome)?;
        }
        capture_astro_ssd_prompt_cache_partial_tail_block(
            engine,
            astro_ssd_prompt_cache,
            model,
            active_request,
            read_stretch.prefill_end,
        );
    }
    active_request.advance_position(read_stretch.token_count())?;
    active_request.prefill_cursor = read_stretch.prefill_end;
    Ok(())
}

/// A save the `astro_ssd_prompt_cache` refused, which the capture reports as
/// an invalid request, turns saving off for the rest of this request and lets
/// it continue; any other failure (the GPU, the model) still stops it.
fn stop_saving_after_a_failed_save<RequestExtension>(
    active_request: &mut TextEngineRequest<RequestExtension>,
    capture_outcome: Result<(), InferenceEngineError>,
) -> Result<(), InferenceEngineError> {
    match capture_outcome {
        Err(InferenceEngineError::InvalidRequest { reason }) => {
            tracing::warn!(
                %reason,
                "astro_ssd_prompt_cache saving stopped for this request; the reply continues"
            );
            active_request.can_use_astro_ssd_prompt_cache = false;
            Ok(())
        }
        other_outcome => other_outcome,
    }
}

/// Retires the stretch's tape and reclaims the `mlx_buffer_cache` only when
/// it is large enough to justify an IOGPU-visible clear, logging the memory
/// readings either side.
pub fn reclaim_mlx_buffer_cache_after_prefill_chunk<Engine>(
    engine: &Engine,
    request_id: RequestId,
    active_request: &mut TextEngineRequest<Engine::RequestExtension>,
    read_stretch: PromptStretch,
) -> Result<(), InferenceEngineError>
where
    Engine: TextExecutionCoreOwner<Model: TextModel>,
{
    let model = engine.execution_core().loaded_model()?;
    let memory_snapshot_before_cleanup = model.memory_snapshot_for_telemetry().ok();
    let cleanup_started_at = Instant::now();
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
        .map_err(fatal_runtime_error)?;
    let memory_snapshot_after_cleanup = model.memory_snapshot_for_telemetry().ok();
    tracing::info!(
        request_id = request_id.value(),
        cleanup_stage = "end_of_prefill_chunk_cleanup",
        prefill_start = read_stretch.prefill_start,
        prefill_end = read_stretch.prefill_end,
        active_memory_bytes_before_cleanup = memory_snapshot_before_cleanup
            .as_ref()
            .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
        mlx_buffer_cache_bytes_before_cleanup = memory_snapshot_before_cleanup
            .as_ref()
            .map(|memory_snapshot| memory_snapshot.mlx_buffer_cache_bytes()),
        active_memory_bytes_after_cleanup = memory_snapshot_after_cleanup
            .as_ref()
            .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
        mlx_buffer_cache_bytes_after_cleanup = memory_snapshot_after_cleanup
            .as_ref()
            .map(|memory_snapshot| memory_snapshot.mlx_buffer_cache_bytes()),
        runtime_active_memory_limit_bytes =
            model.runtime().memory_limits().active_memory_limit_bytes(),
        cleanup_elapsed_millis = cleanup_started_at.elapsed().as_millis(),
        "cleared MLX buffer cache storage after prompt-processing chunk"
    );
    Ok(())
}

/// The progress token for one read stretch, with the memory telemetry of the
/// moment it finished.
///
/// Demand-selected pages are materialized once after all prompt stretches;
/// rebuilding them at every stretch would repeatedly read the same model
/// payload while the demand histogram is still changing.
pub fn prefill_progress_token<Engine>(
    engine: &Engine,
    active_request: &TextEngineRequest<Engine::RequestExtension>,
    request_id: RequestId,
    read_stretch: PromptStretch,
    stretch_timing: PromptStretchTiming,
    neural_engine_chunk: Option<NeuralEngineChunkReport>,
) -> Result<GeneratedToken, InferenceEngineError>
where
    Engine: TextExecutionCoreOwner<Model: TextModel>,
{
    let model = engine.execution_core().loaded_model()?;
    let prefill_token_count = read_stretch.token_count();
    let mlx_memory_snapshot = model.memory_snapshot_for_telemetry().ok();
    let prefill_chunk_elapsed_millis = stretch_timing
        .forward_chunk_started_at
        .elapsed()
        .as_millis() as u64;
    tracing::trace!(
        request_id = request_id.value(),
        prefill_start_token = read_stretch.prefill_start,
        prefill_end_token = read_stretch.prefill_end,
        prefill_token_count,
        forward_chunk_elapsed_millis = stretch_timing.forward_chunk_elapsed_millis,
        prefill_chunk_elapsed_millis,
        "completed synchronous prompt-processing chunk"
    );
    let mlx_memory_telemetry = match mlx_memory_snapshot {
        Some(mlx_memory_snapshot) => Some(prefill_mlx_memory_telemetry_from_snapshot(
            model,
            &active_request.request_decoder_state,
            mlx_memory_snapshot,
        )?),
        None => None,
    };
    let expert_residency_telemetry = mlx_memory_telemetry
        .as_ref()
        .and_then(|telemetry| telemetry.expert_residency_telemetry());
    Ok(GeneratedToken::PrefillProgress {
        processed_token_count: prefill_token_count as u32,
        elapsed_millis: prefill_chunk_elapsed_millis,
        forward_prefill_chunk_elapsed_millis: stretch_timing.forward_chunk_elapsed_millis,
        completed_prefill_chunk_tokens: u32::try_from(prefill_token_count).map_err(|_| {
            fatal_engine_error("completed_prefill_chunk_tokens exceeds the u32 range")
        })?,
        mlx_memory_telemetry,
        expert_residency_telemetry,
        expert_memory_mode: Some(model.observed_expert_memory_mode()),
        prompt_work_reuse: active_request.prompt_work_reuse,
        astro_ssd_prompt_cache_diagnostics: active_request
            .astro_ssd_prompt_cache_diagnostics
            .clone(),
        neural_engine_chunk,
        prompt_layer_progress: None,
    })
}

/// The prompt-phase MLX memory telemetry from one memory reading. The
/// decode-phase sibling lives in `generated_token_emission`.
pub fn prefill_mlx_memory_telemetry_from_snapshot<Model: TextModel>(
    model: &Model,
    request_decoder_state: &RequestDecoderStateStack,
    mlx_memory_snapshot: MlxMemorySnapshot,
) -> Result<MlxMemoryTelemetry, InferenceEngineError> {
    let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(mlx_active_memory_bytes_exceed_u64_range_error)?;
    let active_memory_breakdown =
        model.active_memory_breakdown(request_decoder_state, 0, active_memory_bytes, 0);
    let mlx_buffer_cache_bytes = u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes())
        .map_err(mlx_buffer_cache_bytes_exceed_u64_range_error)?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(mlx_peak_memory_bytes_exceed_u64_range_error)?;
    Ok(MlxMemoryTelemetry::new(
        active_memory_bytes,
        mlx_buffer_cache_bytes,
        peak_memory_bytes,
        active_memory_breakdown,
    )
    // The claim derives from the breakdown of this exact reading so the
    // progress pair cannot straddle a change of expert memory mode (issue #337).
    .with_expert_residency_telemetry(
        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
    )
    // Issue #510: the prompt progress sample carries the same utilization
    // split, composed from the memory governor's accounts.
    .with_memory_ceiling_utilization(
        model
            .memory_ceiling_utilization_for_breakdown(active_memory_bytes, active_memory_breakdown),
    )
    .with_memory_ledger_reading(model.latest_memory_ledger_reading()))
}

#[cfg(test)]
#[path = "prompt_reading_steps_tests.rs"]
mod prompt_reading_steps_tests;
