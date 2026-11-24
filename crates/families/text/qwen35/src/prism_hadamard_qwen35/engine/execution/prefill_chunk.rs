//! Chunked prefill for the owner-thread folded-ternary execution.
//!
//! The shared decode step asks for one chunk at a time while any of the
//! prompt remains, and measures the whole chunk as its prompt advance span;
//! the measured regions inside it keep their boundaries.

use std::time::Instant;

use ipc_protocol::{ExpertMemoryMode, WorkerPromptWorkReuse};

use mlx_bridge::MlxArray;

use crate::AstroSsdPromptCacheDiskStore;
use crate::GeneratedToken;
use crate::InferenceEngineError;
use crate::MlxMemoryTelemetry;
use crate::PerformanceAttribution;
use crate::PerformanceOperation;
use crate::astro_ssd_prompt_cache_boundary_clamped_prefill_chunk_end;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;

use super::engine_trait_impl::gpu_error;
use super::{PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution};

/// Anchors a prefill chunk at the processed-prompt cursor and returns how
/// many prompt tokens it consumes, stopping the chunk at the first
/// persistent astro SSD prompt cache block boundary it would cross.
///
/// A durable boundary snapshot is only valid at a forward edge, so at most
/// one durable publication per forward keeps the boundary snapshot honest.
pub(super) fn prefill_chunk_length(
    prefill_cursor: usize,
    remaining_prompt_token_count: usize,
    prompt_processing_chunk_tokens: usize,
    astro_ssd_prompt_cache_block_token_count: Option<usize>,
) -> usize {
    let chunk_start = prefill_cursor;
    let requested_chunk_end =
        chunk_start + remaining_prompt_token_count.min(prompt_processing_chunk_tokens);
    let clamped_chunk_end = match astro_ssd_prompt_cache_block_token_count {
        Some(block_token_count) => astro_ssd_prompt_cache_boundary_clamped_prefill_chunk_end(
            chunk_start,
            requested_chunk_end,
            block_token_count,
        ),
        None => requested_chunk_end,
    };
    clamped_chunk_end - chunk_start
}

/// Builds the chunk's `[1, chunk_length]` token-index array and runs the
/// straight forward pass. The shared step's prompt advance span already
/// covers this forward, so it is not measured again here.
pub(super) fn forward_prefill_chunk_hidden_states(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    chunk_token_ids: &[i32],
    chunk_length: usize,
) -> Result<MlxArray, InferenceEngineError> {
    let token_indices = model
        .runtime
        .array_from_i32(chunk_token_ids, &[1, chunk_length as i32])
        .map_err(gpu_error)?;
    active_generation
        .forward_hidden_states(model, &token_indices)
        .map_err(gpu_error)
}

/// Computes the final logits when this chunk exhausted the prompt, then
/// submits the chunk's state graph for asynchronous evaluation. Both
/// submission branches sit inside their own
/// `PrefillStateAsyncEvaluationSubmission` region and return the final
/// logits so the first decode token can be sampled from them.
pub(super) fn submit_prefill_state_evaluation(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    hidden_states: &MlxArray,
    remaining_prompt_token_ids_are_empty: bool,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    let final_logits = if remaining_prompt_token_ids_are_empty {
        Some(
            active_generation
                .measure_operation_with_request(
                    PerformanceOperation::FinalLogitsGraphConstruction,
                    |_| model.logits_for_last_hidden_state(hidden_states),
                )
                .map_err(gpu_error)?,
        )
    } else {
        None
    };
    if let Some(final_logits) = final_logits.as_ref() {
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_| {
                    model
                        .runtime
                        .evaluate_arrays(&[hidden_states, final_logits])
                },
            )
            .map_err(gpu_error)?;
    } else {
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_| model.runtime.evaluate_arrays(&[hidden_states]),
            )
            .map_err(gpu_error)?;
    }
    Ok(final_logits)
}

/// When the chunk completed the prompt, samples the first decode token with
/// the shared sampler inside `DecodeSamplingSpan`, kicks off its asynchronous
/// evaluation, and queues it as the pending token the next decode step reads.
/// The output budget is the shared decode step's to enforce, so the token is
/// always queued.
pub(super) fn queue_prefill_first_token(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    final_logits: Option<&MlxArray>,
) -> Result<(), InferenceEngineError> {
    if let Some(final_logits) = final_logits {
        let first_token = active_generation.measure_operation_with_request(
            PerformanceOperation::DecodeSamplingSpan,
            |active_generation| active_generation.build_generated_token(model, final_logits),
        )?;
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::DecodeAsyncEvaluationSubmission,
                |_| model.runtime.async_eval_arrays(&[&first_token]),
            )
            .map_err(gpu_error)?;
        active_generation.pending_generated_token = Some(first_token);
    }
    Ok(())
}

/// Publishes the astro SSD prompt cache block the chunk just completed, if any.
///
/// The capture's own attribution is swapped in for the call and swapped back
/// afterwards so the chunk's runtime attribution never records capture work.
pub(super) fn capture_completed_prefill_astro_ssd_prompt_cache_block(
    model: &PrismHadamardTextModel,
    astro_ssd_prompt_cache: Option<&AstroSsdPromptCacheDiskStore>,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
) -> Result<(), InferenceEngineError> {
    let Some(astro_ssd_prompt_cache) = astro_ssd_prompt_cache else {
        return Ok(());
    };
    let mut capture_attribution = std::mem::replace(
        &mut active_generation.performance_attribution,
        PerformanceAttribution::disabled(),
    );
    let capture_result =
        super::super::astro_ssd_prompt_cache::capture_completed_astro_ssd_prompt_cache_block(
            &model.runtime,
            astro_ssd_prompt_cache,
            active_generation,
            &mut capture_attribution,
        );
    active_generation.performance_attribution = capture_attribution;
    capture_result
}

impl PrismHadamardQwen35InferenceExecution {
    /// Advances one bounded prompt-processing chunk of `active` through the
    /// straight forward pass and reports prefill progress to the worker. The
    /// shared decode step holds the request outside the core while it runs.
    pub(super) fn prefill_next_chunk(
        &mut self,
        active: &mut PrismHadamardQwen35ActiveGeneration,
        prompt_processing_chunk_tokens: usize,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        // Sampled before the chunk runs; chunk boundaries are far enough
        // apart that one-chunk-old memory reads stay honest.
        let mlx_memory_telemetry =
            self.mlx_memory_telemetry_with_live_decoder_state(Some(&active.request_decoder_state));
        let model = self.execution_core.loaded_model()?;
        let astro_ssd_prompt_cache = self.execution_core.astro_ssd_prompt_cache.clone();
        let chunk_length = prefill_chunk_length(
            active.prefill_cursor,
            active.engine_extension.remaining_prompt_token_ids.len(),
            prompt_processing_chunk_tokens,
            astro_ssd_prompt_cache
                .as_ref()
                .map(|astro_ssd_prompt_cache| {
                    astro_ssd_prompt_cache
                        .model_contract_ref()
                        .block_token_count()
                }),
        );
        let chunk_token_ids: Vec<i32> = active
            .engine_extension
            .remaining_prompt_token_ids
            .drain(..chunk_length)
            .map(|token_id| token_id as i32)
            .collect();
        let chunk_started_at = Instant::now();
        let hidden_states =
            forward_prefill_chunk_hidden_states(model, active, &chunk_token_ids, chunk_length)?;
        let remaining_prompt_token_ids_are_empty = active
            .engine_extension
            .remaining_prompt_token_ids
            .is_empty();
        let final_logits = submit_prefill_state_evaluation(
            model,
            active,
            &hidden_states,
            remaining_prompt_token_ids_are_empty,
        )?;
        let elapsed_millis = chunk_started_at
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        queue_prefill_first_token(model, active, final_logits.as_ref())?;
        // Capture runs after the chunk's state graph was submitted: the
        // published tensors chain onto the lazy arrays and materialize during
        // publication, so no extra device synchronization happens here.
        active.prefill_cursor += chunk_length;
        capture_completed_prefill_astro_ssd_prompt_cache_block(
            model,
            astro_ssd_prompt_cache.as_deref(),
            active,
        )?;
        Ok(prefill_progress_token(
            chunk_length,
            elapsed_millis,
            mlx_memory_telemetry,
            active.prompt_work_reuse,
        ))
    }
}

/// Builds the prefill progress token for exactly one chunk.
///
/// The worker accumulates per-chunk deltas into the request-wide prefill
/// cursor, so this must stay the chunk's own token count — a cumulative
/// count double-counts every earlier chunk and pins the status surface at
/// 100% while most of the prompt is still processing.
pub(super) fn prefill_progress_token(
    chunk_length: usize,
    elapsed_millis: u64,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    prompt_work_reuse: WorkerPromptWorkReuse,
) -> GeneratedToken {
    GeneratedToken::PrefillProgress {
        processed_token_count: chunk_length as u32,
        elapsed_millis,
        forward_prefill_chunk_elapsed_millis: elapsed_millis,
        completed_prefill_chunk_tokens: chunk_length as u32,
        mlx_memory_telemetry,
        expert_residency_telemetry: None,
        expert_memory_mode: Some(ExpertMemoryMode::Resident),
        prompt_work_reuse,
        astro_ssd_prompt_cache_diagnostics: None,
        neural_engine_chunk: None,
        prompt_layer_progress: None,
    }
}

#[cfg(test)]
#[path = "prefill_chunk_tests.rs"]
mod prefill_chunk_tests;
