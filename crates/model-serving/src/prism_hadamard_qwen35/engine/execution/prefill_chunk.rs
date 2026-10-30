//! Chunked-prefill units for the owner-thread folded-ternary execution.
//!
//! Split from `engine_trait_impl` so each file stays inside the source-size
//! budget; the measured regions (`measure_operation` spans) keep their exact
//! original boundaries because performance attribution depends on them.

use astronomical_ipc_protocol::{ExpertMemoryMode, WorkerPromptWorkReuse};

use mlx_c_rust::MlxArray;

use crate::GeneratedToken;
use crate::InferenceEngineError;
use crate::MlxMemoryTelemetry;
use crate::PerformanceAttribution;
use crate::PerformanceOperation;
use crate::PersistentPromptCacheDiskStore;
use crate::persistent_prompt_cache_boundary_clamped_prefill_chunk_end;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::prism_hadamard_qwen35::model::PrismStateSlotLayout;

use super::PrismHadamardQwen35ActiveGeneration;
use super::engine_trait_impl::gpu_error;
use super::sampling::sample_token;
use super::successor_stays_pending;

/// Anchors a prefill chunk at the processed-prompt cursor and returns how
/// many prompt tokens it consumes, stopping the chunk at the first
/// persistent prompt-cache block boundary it would cross.
///
/// A durable boundary snapshot is only valid at a forward edge, so at most
/// one durable publication per forward keeps the boundary snapshot honest.
pub(super) fn prefill_chunk_length(
    processed_prompt_token_count: u32,
    remaining_prompt_token_count: usize,
    prompt_processing_chunk_tokens: u32,
    persistent_prompt_cache_block_token_count: Option<usize>,
) -> usize {
    let chunk_start = processed_prompt_token_count as usize;
    let requested_chunk_end =
        chunk_start + remaining_prompt_token_count.min(prompt_processing_chunk_tokens as usize);
    let clamped_chunk_end = match persistent_prompt_cache_block_token_count {
        Some(block_token_count) => persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
            chunk_start,
            requested_chunk_end,
            block_token_count,
        ),
        None => requested_chunk_end,
    };
    clamped_chunk_end - chunk_start
}

/// Builds the chunk's `[1, chunk_length]` token-index array and runs the
/// straight forward pass inside its `PromptPrefillAdvanceSpan` region.
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
        .measure_operation(
            PerformanceOperation::PromptPrefillAdvanceSpan,
            |active_generation| {
                model.forward_hidden_states(&token_indices, &mut active_generation.generation_state)
            },
        )
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
                .measure_operation(PerformanceOperation::FinalLogitsGraphConstruction, |_| {
                    model.logits_for_last_hidden_state(hidden_states)
                })
                .map_err(gpu_error)?,
        )
    } else {
        None
    };
    if let Some(final_logits) = final_logits.as_ref() {
        active_generation
            .measure_operation(
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
            .measure_operation(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_| model.runtime.evaluate_arrays(&[hidden_states]),
            )
            .map_err(gpu_error)?;
    }
    Ok(final_logits)
}

/// When the chunk completed the prompt, samples the first decode token
/// inside `DecodeSamplingSpan`, kicks off its asynchronous evaluation, and
/// queues it as the pending sampled token the next decode step consumes.
pub(super) fn queue_prefill_first_token(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    final_logits: Option<&MlxArray>,
) -> Result<(), InferenceEngineError> {
    if let Some(final_logits) = final_logits {
        let first_token = active_generation.measure_operation(
            PerformanceOperation::DecodeSamplingSpan,
            |active_generation| sample_token(model, final_logits, active_generation),
        )?;
        active_generation
            .measure_operation(
                PerformanceOperation::DecodeAsyncEvaluationSubmission,
                |_| model.runtime.async_eval_arrays(&[&first_token]),
            )
            .map_err(gpu_error)?;
        active_generation.pending_sampled_token =
            successor_stays_pending(active_generation.remaining_output_tokens)
                .then_some(first_token);
    }
    Ok(())
}

/// Publishes the prompt-cache block the chunk just completed, if any.
///
/// The capture's own attribution is swapped in for the call and swapped back
/// afterwards so the chunk's runtime attribution never records capture work.
pub(super) fn capture_completed_prefill_cache_block(
    model: &PrismHadamardTextModel,
    persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
) -> Result<(), InferenceEngineError> {
    let Some(persistent_prompt_cache) = persistent_prompt_cache else {
        return Ok(());
    };
    let slot_layout = PrismStateSlotLayout::for_model(model);
    let mut capture_attribution = std::mem::replace(
        &mut active_generation.performance_attribution,
        PerformanceAttribution::disabled(),
    );
    let capture_result = super::super::prompt_cache::capture_completed_cache_block(
        &model.runtime,
        persistent_prompt_cache,
        active_generation,
        &slot_layout,
        &mut capture_attribution,
    );
    active_generation.performance_attribution = capture_attribution;
    capture_result
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
        persistent_prompt_cache_diagnostics: None,
    }
}

#[cfg(test)]
#[path = "prefill_chunk_tests.rs"]
mod prefill_chunk_tests;
