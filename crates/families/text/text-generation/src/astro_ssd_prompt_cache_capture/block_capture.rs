//! Request-side extraction and `astro_ssd_prompt_cache` publication, shared by
//! every text engine.
//!
//! The engine thread copies each block's state to the host and queues it for
//! the background writer (`engine_core::astro_ssd_prompt_cache::publication_queue`),
//! then moves the request on: the block's state is already in memory, and the
//! store's readers wait for the queue to drain, so a later request still finds
//! it. Writing inside the prompt loop held a 15,000-token prompt of Ornith
//! 35B-A3B back by ~450 ms before its first token (2026-11-27, M5 Pro).
//!
//! The parent chain advances when a block is queued. Capacity pressure while
//! copying gets one narrowly typed retry, after allocator cleanup and whatever
//! the engine's `reclaim_before_publication_retry` frees: retained expert
//! pages in the Qwen3.5 text engine. A block the writer later fails to write is
//! logged and dropped, and so are its children, which fail the store's
//! published-parent check; the reply continues from memory.

use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheBoundaryCheckpoint,
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreError,
    AstroSsdPromptCachePublicationOutcome, InferenceEngineError, PerformanceAttribution,
    PerformanceOperation,
};
use engine_core::astro_ssd_prompt_cache::{
    AstroSsdPromptCacheHostTensors, AstroSsdPromptCachePublicationJob,
    AstroSsdPromptCachePublicationQueue, copy_state_tensors_to_host,
};
use mlx_bridge::MlxArray;
use runtime_integration::MlxMemorySnapshot;
use std::collections::HashMap;

use super::capture_hook::TextAstroSsdPromptCacheCapture;
use crate::engine_errors::fatal_runtime_error;
use crate::engine_request::TextEngineRequest;
use crate::model::TextServingModel;
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use engine_core::astro_ssd_prompt_cache::capture_planning;

/// Converts a required persistence failure into the user-visible engine error.
pub(super) fn required_prompt_state_persistence_failure<RequestExtension>(
    active_request: &TextEngineRequest<RequestExtension>,
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    tracing::error!(
        request_id = active_request.request_id.value(),
        failure_stage,
        error = %internal_error,
        "required persistent astro-ssd-prompt-cache capture stopped the request"
    );
    InferenceEngineError::InvalidRequest {
        reason: format!(
            "astro SSD prompt cache failed during {failure_stage}; the request was stopped"
        ),
    }
}

pub fn capture_astro_ssd_prompt_cache_blocks<CaptureEngine: TextAstroSsdPromptCacheCapture>(
    capture_engine: &CaptureEngine,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    model: &CaptureEngine::Model,
    active_request: &mut TextEngineRequest<CaptureEngine::RequestExtension>,
    successful_prefill_start: usize,
    successful_prefill_end: usize,
    boundary_checkpoints: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
) -> Result<(), InferenceEngineError> {
    let astro_ssd_prompt_cache_block_token_count =
        astro_ssd_prompt_cache.model_contract.block_token_count();
    for boundary_checkpoint in boundary_checkpoints {
        let PlannedCaptureBlock {
            block_start,
            block_end,
            astro_ssd_prompt_cache_block_key,
        } = plan_captured_block(
            astro_ssd_prompt_cache,
            active_request,
            successful_prefill_start,
            successful_prefill_end,
            boundary_checkpoint.completed_prefill_chunk_tokens,
        )?;
        // Extraction returns MLX arrays referencing exact decoder state. Keep
        // these same arrays alive through a possible retry; recapturing after
        // reclamation could observe mutated request state or duplicate work.
        let kv_block_tensors = extract_kv_block_tensors_with_measurement(
            model,
            active_request,
            block_start,
            block_end,
            astro_ssd_prompt_cache_block_token_count,
        )?;
        release_publication_workspace_if_needed(model, active_request)?;
        // Cloned so publication can borrow the request mutably while still
        // naming this block's parent.
        let parent_block_key = active_request
            .last_restored_astro_ssd_prompt_cache_block_key
            .clone();
        let save_outcome = publish_block_with_reclamation_retry(
            capture_engine,
            model,
            active_request,
            astro_ssd_prompt_cache,
            &astro_ssd_prompt_cache_block_key,
            parent_block_key.as_ref(),
            &kv_block_tensors,
            &boundary_checkpoint.recurrent_snapshot_tensors,
        );
        apply_publication_outcome(
            active_request,
            save_outcome,
            astro_ssd_prompt_cache_block_key,
            block_start,
            block_end,
        )?;
    }
    Ok(())
}

/// Where one boundary's block sits in the prompt, and its identity.
struct PlannedCaptureBlock {
    block_start: usize,
    block_end: usize,
    astro_ssd_prompt_cache_block_key: AstroSsdPromptCacheBlockKey,
}

/// Plans the block a boundary checkpoint completes: its token window, the
/// causal input bound into its identity, and the identity itself. Any
/// inconsistency stops the request, because capture is required.
fn plan_captured_block<RequestExtension>(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    active_request: &TextEngineRequest<RequestExtension>,
    successful_prefill_start: usize,
    successful_prefill_end: usize,
    completed_prefill_chunk_tokens: usize,
) -> Result<PlannedCaptureBlock, InferenceEngineError> {
    let capture_planning::CaptureBlockWindow {
        block_start,
        block_end,
        block_index,
    } = match capture_planning::capture_block_window(
        successful_prefill_start,
        completed_prefill_chunk_tokens,
        astro_ssd_prompt_cache.model_contract.block_token_count(),
        successful_prefill_end,
    ) {
        Ok(capture_block_window) => capture_block_window,
        Err(rejection) => {
            return Err(required_prompt_state_persistence_failure(
                active_request,
                "required persistent prompt-state capture",
                rejection.description(),
            ));
        }
    };
    let empty_block_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
    let block_causal_input = match capture_planning::select_block_causal_input(
        &active_request.astro_ssd_prompt_cache_block_causal_inputs,
        block_index,
    ) {
        Ok(capture_planning::BlockCausalInputSelection::Empty) => &empty_block_causal_input,
        Ok(capture_planning::BlockCausalInputSelection::FromPlan(block_causal_input)) => {
            block_causal_input
        }
        Err(rejection) => {
            return Err(required_prompt_state_persistence_failure(
                active_request,
                "required persistent prompt-state capture",
                rejection.description(),
            ));
        }
    };
    let astro_ssd_prompt_cache_block_key = match capture_planning::block_key_for_captured_block(
        active_request
            .last_restored_astro_ssd_prompt_cache_block_key
            .as_ref(),
        &astro_ssd_prompt_cache.model_contract,
        &active_request.input_token_ids[block_start..block_end],
        block_causal_input,
    ) {
        Ok(astro_ssd_prompt_cache_block_key) => astro_ssd_prompt_cache_block_key,
        Err(_) => {
            return Err(required_prompt_state_persistence_failure(
                active_request,
                "required persistent prompt-state capture",
                "astro-ssd-prompt-cache block identity construction failed",
            ));
        }
    };
    Ok(PlannedCaptureBlock {
        block_start,
        block_end,
        astro_ssd_prompt_cache_block_key,
    })
}

#[allow(clippy::too_many_arguments)]
/// Publishes one block: copies its state to the host, retrying once after
/// the engine reclaims what memory it can when the typed MLX active-memory
/// limit blocked the copy, then hands it to the background writer. Reusing
/// the caller's extracted tensors is essential: this is resource
/// reclamation, not a second logical capture.
///
/// A block the store already holds complete is not copied again. With no
/// writer (one could not be started) the block is written here, as before.
pub(super) fn publish_block_with_reclamation_retry<
    CaptureEngine: TextAstroSsdPromptCacheCapture,
>(
    capture_engine: &CaptureEngine,
    model: &CaptureEngine::Model,
    active_request: &mut TextEngineRequest<CaptureEngine::RequestExtension>,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    block_key: &AstroSsdPromptCacheBlockKey,
    parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
    kv_block_tensors: &HashMap<String, MlxArray>,
    recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
) -> Result<AstroSsdPromptCachePublicationOutcome, PromptBlockPublicationFailure> {
    if astro_ssd_prompt_cache.holds_complete_block(&block_key.block_hash()) {
        return Ok(AstroSsdPromptCachePublicationOutcome::AlreadyPublished);
    }
    let host_block = copy_block_to_host_with_reclamation_retry(
        capture_engine,
        model,
        active_request,
        kv_block_tensors,
        recurrent_snapshot_tensors,
    )?;
    let publication_job = AstroSsdPromptCachePublicationJob {
        block_key: block_key.clone(),
        parent_block_key: parent_block_key.cloned(),
        sequence_state_tensors: host_block.sequence_state_tensors,
        boundary_state_tensors: host_block.boundary_state_tensors,
    };
    hand_block_to_writer(
        capture_engine
            .execution_core()
            .astro_ssd_prompt_cache_publication_queue
            .as_ref(),
        astro_ssd_prompt_cache,
        active_request,
        publication_job,
    )
}

/// A block's state copied to the host, ready for any thread to write.
struct HostBlock {
    sequence_state_tensors: AstroSsdPromptCacheHostTensors,
    boundary_state_tensors: AstroSsdPromptCacheHostTensors,
}

/// Copies the block's tensors to the host, retrying once, and only for the
/// typed MLX active-memory limit, after the engine reclaims memory.
fn copy_block_to_host_with_reclamation_retry<CaptureEngine: TextAstroSsdPromptCacheCapture>(
    capture_engine: &CaptureEngine,
    model: &CaptureEngine::Model,
    active_request: &mut TextEngineRequest<CaptureEngine::RequestExtension>,
    kv_block_tensors: &HashMap<String, MlxArray>,
    recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
) -> Result<HostBlock, PromptBlockPublicationFailure> {
    match copy_block_to_host(
        model,
        active_request,
        kv_block_tensors,
        recurrent_snapshot_tensors,
    ) {
        Err(copy_error) if copy_error.active_memory_deficit_bytes().is_some() => {
            let active_memory_deficit_bytes = copy_error.active_memory_deficit_bytes().unwrap_or(0);
            capture_engine
                .reclaim_before_publication_retry(
                    model,
                    active_request,
                    active_memory_deficit_bytes,
                )
                .map_err(PromptBlockPublicationFailure::Reclamation)?;
            copy_block_to_host(
                model,
                active_request,
                kv_block_tensors,
                recurrent_snapshot_tensors,
            )
            .map_err(PromptBlockPublicationFailure::Publication)
        }
        copy_outcome => copy_outcome.map_err(PromptBlockPublicationFailure::Publication),
    }
}

fn copy_block_to_host<RequestExtension>(
    model: &impl TextServingModel,
    active_request: &mut TextEngineRequest<RequestExtension>,
    kv_block_tensors: &HashMap<String, MlxArray>,
    recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
) -> Result<HostBlock, AstroSsdPromptCacheDiskStoreError> {
    active_request.performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |_performance_attribution| {
            Ok(HostBlock {
                sequence_state_tensors: copy_state_tensors_to_host(
                    model.runtime(),
                    kv_block_tensors,
                )?,
                boundary_state_tensors: copy_state_tensors_to_host(
                    model.runtime(),
                    recurrent_snapshot_tensors,
                )?,
            })
        },
    )
}

/// Queues the block for the background writer, or writes it on this thread
/// when there is no writer. A queued block counts as published: it is in
/// memory and will be on disk before any reader looks. A writer that stopped
/// (only a panic stops it) refuses the block, which is then not saved.
fn hand_block_to_writer<RequestExtension>(
    publication_queue: Option<&AstroSsdPromptCachePublicationQueue>,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    active_request: &mut TextEngineRequest<RequestExtension>,
    publication_job: AstroSsdPromptCachePublicationJob,
) -> Result<AstroSsdPromptCachePublicationOutcome, PromptBlockPublicationFailure> {
    let Some(publication_queue) = publication_queue else {
        return publish_on_this_thread(astro_ssd_prompt_cache, active_request, &publication_job);
    };
    match publication_queue.enqueue(publication_job) {
        Ok(()) => Ok(AstroSsdPromptCachePublicationOutcome::Published),
        Err(queue_error) => {
            // The job went down with the writer; the request keeps going
            // without this block rather than writing state it no longer holds.
            tracing::warn!(%queue_error, "astro_ssd_prompt_cache writer unavailable");
            Err(PromptBlockPublicationFailure::Publication(queue_error))
        }
    }
}

fn publish_on_this_thread<RequestExtension>(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    active_request: &mut TextEngineRequest<RequestExtension>,
    publication_job: &AstroSsdPromptCachePublicationJob,
) -> Result<AstroSsdPromptCachePublicationOutcome, PromptBlockPublicationFailure> {
    // The disk-store API needs mutable attribution while the request also
    // remains mutably borrowed. Move the owner out and put it back before
    // interpreting the outcome.
    let mut request_performance_attribution = std::mem::replace(
        &mut active_request.performance_attribution,
        PerformanceAttribution::disabled(),
    );
    let publication_outcome = astro_ssd_prompt_cache
        .publish_host_block_with_performance_attribution(
            &publication_job.block_key,
            publication_job.parent_block_key.as_ref(),
            &publication_job.sequence_state_tensors,
            &publication_job.boundary_state_tensors,
            &mut request_performance_attribution,
        );
    active_request.performance_attribution = request_performance_attribution;
    publication_outcome.map_err(PromptBlockPublicationFailure::Publication)
}

/// Extracts one block's KV tensors under the request's extraction measurement,
/// converting a failure into the request-stopping persistence error.
fn extract_kv_block_tensors_with_measurement<RequestExtension>(
    model: &impl TextServingModel,
    active_request: &mut TextEngineRequest<RequestExtension>,
    kv_block_start: usize,
    kv_block_end: usize,
    astro_ssd_prompt_cache_block_token_count: usize,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    match active_request.measure_operation_with_request(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |active_request| {
            active_request
                .request_decoder_state
                .extract_astro_ssd_prompt_cache_kv_block_tensors(
                    model.runtime(),
                    kv_block_start,
                    kv_block_end,
                    astro_ssd_prompt_cache_block_token_count,
                )
        },
    ) {
        Ok(kv_block_tensors) => Ok(kv_block_tensors),
        Err(error) => {
            tracing::warn!(
                kv_block_start,
                kv_block_end,
                block_start = kv_block_start,
                block_end = kv_block_end,
                %error,
                "astro-ssd-prompt-cache KV extraction failed"
            );
            Err(required_prompt_state_persistence_failure(
                active_request,
                "required persistent prompt-state capture",
                error,
            ))
        }
    }
}

/// Releases allocator workspace before publication when the active-memory
/// ceiling needs the room, and records the bytes the release cleared.
fn release_publication_workspace_if_needed<RequestExtension>(
    model: &impl TextServingModel,
    active_request: &mut TextEngineRequest<RequestExtension>,
) -> Result<(), InferenceEngineError> {
    let memory_snapshot_before_cleanup = model.runtime().memory_snapshot().ok();
    let is_publication_workspace_needed = capture_planning::publication_workspace_decision(
        memory_snapshot_before_cleanup
            .as_ref()
            .map(memory_snapshot_active_memory_bytes),
        model
            .runtime()
            .configured_memory_limit_bytes()
            .ok()
            .map(configured_memory_limit_bytes_u64),
    );
    if !is_publication_workspace_needed {
        return Ok(());
    }
    active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::MlxBufferCacheCleanup,
            |_performance_attribution| {
                model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            },
        )
        .map_err(fatal_runtime_error)?;
    let memory_snapshot_after_cleanup = model.runtime().memory_snapshot().ok();
    if let Some(astro_ssd_prompt_cache_diagnostics) =
        active_request.astro_ssd_prompt_cache_diagnostics.as_mut()
    {
        let allocator_bytes_cleared = capture_planning::allocator_bytes_cleared(
            memory_snapshot_before_cleanup
                .as_ref()
                .map(memory_snapshot_mlx_buffer_cache_bytes),
            memory_snapshot_after_cleanup
                .as_ref()
                .map(memory_snapshot_mlx_buffer_cache_bytes),
        );
        astro_ssd_prompt_cache_diagnostics.allocator_bytes_cleared_for_publication =
            astro_ssd_prompt_cache_diagnostics
                .allocator_bytes_cleared_for_publication
                .saturating_add(allocator_bytes_cleared);
    }
    Ok(())
}

/// Applies one publication outcome: records diagnostics, advances the parent
/// chain on durable availability, and translates both failure owners into
/// their request-stopping errors.
fn apply_publication_outcome<RequestExtension>(
    active_request: &mut TextEngineRequest<RequestExtension>,
    save_outcome: Result<AstroSsdPromptCachePublicationOutcome, PromptBlockPublicationFailure>,
    astro_ssd_prompt_cache_block_key: AstroSsdPromptCacheBlockKey,
    block_start: usize,
    block_end: usize,
) -> Result<(), InferenceEngineError> {
    match save_outcome {
        Ok(publication_outcome) => {
            if publication_outcome == AstroSsdPromptCachePublicationOutcome::Published
                && let Some(astro_ssd_prompt_cache_diagnostics) =
                    active_request.astro_ssd_prompt_cache_diagnostics.as_mut()
            {
                astro_ssd_prompt_cache_diagnostics.record_published_block();
            }
            // Both outcomes make the block available to every later reader
            // (a queued block is written before any reader looks), so the next
            // block may use this key as its parent. Diagnostics count blocks
            // handed to the writer, not blocks the store already held.
            active_request.last_restored_astro_ssd_prompt_cache_block_key =
                Some(astro_ssd_prompt_cache_block_key);
            Ok(())
        }
        Err(PromptBlockPublicationFailure::Reclamation(error)) => Err(error),
        Err(PromptBlockPublicationFailure::Publication(error)) => {
            tracing::warn!(block_start, block_end, %error, "astro-ssd-prompt-cache block save failed");
            Err(required_prompt_state_persistence_failure(
                active_request,
                "required persistent prompt-state capture",
                error,
            ))
        }
    }
}

fn memory_snapshot_active_memory_bytes(memory_snapshot: &MlxMemorySnapshot) -> u64 {
    u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(u64::MAX)
}

fn memory_snapshot_mlx_buffer_cache_bytes(memory_snapshot: &MlxMemorySnapshot) -> u64 {
    u64::try_from(memory_snapshot.mlx_buffer_cache_bytes()).unwrap_or(u64::MAX)
}

fn configured_memory_limit_bytes_u64(configured_memory_limit_bytes: usize) -> u64 {
    u64::try_from(configured_memory_limit_bytes).unwrap_or(u64::MAX)
}

/// Distinguishes the two failure owners of a publication with reclamation
/// retry: a failed reclamation propagates as its own engine error, while a
/// failed publication is translated by the persistence owner upstream.
#[derive(Debug)]
pub(super) enum PromptBlockPublicationFailure {
    Reclamation(InferenceEngineError),
    Publication(AstroSsdPromptCacheDiskStoreError),
}

#[must_use]
pub fn astro_ssd_prompt_cache_publication_advances_parent_chain(
    publication_outcome: AstroSsdPromptCachePublicationOutcome,
) -> bool {
    // Kept as a public pure contract for direct tests and alternate callers:
    // there is intentionally no non-durable success variant.
    matches!(
        publication_outcome,
        AstroSsdPromptCachePublicationOutcome::Published
            | AstroSsdPromptCachePublicationOutcome::AlreadyPublished
    )
}
