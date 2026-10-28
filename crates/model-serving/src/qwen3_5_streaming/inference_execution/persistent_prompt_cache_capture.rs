//! Qwen3.5 request-side extraction and required synchronous cache publication.
//!
//! The request cursor advances only after the matching block is durably published
//! or an exact block was already present. Capacity pressure gets one narrowly
//! typed retry after allocator and pageable-expert reclamation; storage, quota,
//! validation, and topology failures stop the request without retry.

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreError, PersistentPromptCachePublicationOutcome,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
};
use astronomical_runtime_integration::MlxMemorySnapshot;
use mlx_c_rust::MlxArray;
use std::collections::HashMap;

use super::engine_request::Qwen3_5EngineRequest;
use super::{Qwen3_5EngineState, Qwen3_5Model, qwen3_5_runtime_error};
use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::capture_planning;
use crate::qwen3_5_streaming;

/// Converts a required persistence failure into the user-visible engine error.
pub(super) fn required_prompt_state_persistence_failure(
    active_request: &Qwen3_5EngineRequest,
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    tracing::error!(
        request_id = active_request.request_id.value(),
        failure_stage,
        error = %internal_error,
        "required persistent prompt-cache capture stopped the request"
    );
    InferenceEngineError::InvalidRequest {
        reason: format!(
            "persistent prompt cache failed during {failure_stage}; the request was stopped"
        ),
    }
}

impl Qwen3_5EngineState {
    pub(super) fn capture_persistent_prompt_cache_blocks(
        &self,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        successful_prefill_start: usize,
        successful_prefill_end: usize,
        boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    ) -> Result<(), InferenceEngineError> {
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        for boundary_checkpoint in boundary_checkpoints {
            let capture_planning::CaptureBlockWindow {
                block_start,
                block_end,
                block_index,
            } = match capture_planning::capture_block_window(
                successful_prefill_start,
                boundary_checkpoint.completed_prefill_chunk_tokens,
                persistent_prompt_cache_block_token_count,
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
            let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
            let block_causal_input = match capture_planning::select_block_causal_input(
                &active_request.persistent_prompt_cache_block_causal_inputs,
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
            let persistent_prompt_cache_block_key =
                match capture_planning::block_key_for_captured_block(
                    active_request
                        .last_restored_persistent_prompt_cache_block_key
                        .as_ref(),
                    &persistent_prompt_cache.model_contract,
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ) {
                    Ok(persistent_prompt_cache_block_key) => persistent_prompt_cache_block_key,
                    Err(_) => {
                        return Err(required_prompt_state_persistence_failure(
                            active_request,
                            "required persistent prompt-state capture",
                            "prompt-cache block identity construction failed",
                        ));
                    }
                };
            // Extraction returns MLX arrays referencing exact decoder state. Keep
            // these same arrays alive through a possible retry; recapturing after
            // reclamation could observe mutated request state or duplicate work.
            let kv_block_tensors = extract_kv_block_tensors_with_measurement(
                model,
                active_request,
                block_start,
                block_end,
                persistent_prompt_cache_block_token_count,
            )?;
            release_publication_workspace_if_needed(model, active_request)?;
            // The disk-store API needs mutable attribution while the request also
            // remains mutably borrowed. Move the owner out temporarily and put it
            // back on every return path before interpreting publication outcome.
            let parent_block_key = active_request
                .last_restored_persistent_prompt_cache_block_key
                .clone();
            let save_outcome = self.publish_block_with_reclamation_retry(
                model,
                active_request,
                persistent_prompt_cache,
                &persistent_prompt_cache_block_key,
                parent_block_key.as_ref(),
                &kv_block_tensors,
                &boundary_checkpoint.recurrent_snapshot_tensors,
            );
            apply_publication_outcome(
                active_request,
                save_outcome,
                persistent_prompt_cache_block_key,
                block_start,
                block_end,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Publishes one block, retrying once after expert reclamation when the
    /// typed MLX active-memory limit blocked serialization. Reusing the
    /// caller's extracted tensors is essential: this is resource reclamation,
    /// not a second logical capture.
    pub(super) fn publish_block_with_reclamation_retry(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        block_key: &PersistentPromptCacheBlockKey,
        parent_block_key: Option<&PersistentPromptCacheBlockKey>,
        kv_block_tensors: &HashMap<String, MlxArray>,
        recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
    ) -> Result<PersistentPromptCachePublicationOutcome, PromptBlockPublicationFailure> {
        // The disk-store API needs mutable attribution while the request also
        // remains mutably borrowed. Move the owner out temporarily and put it
        // back on every return path before interpreting publication outcome.
        let mut request_performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let save_outcome = persistent_prompt_cache.publish_block_with_performance_attribution(
            model.runtime(),
            block_key,
            parent_block_key,
            kv_block_tensors,
            recurrent_snapshot_tensors,
            &mut request_performance_attribution,
        );
        active_request.performance_attribution = request_performance_attribution;
        // Retry exactly once and only for the typed MLX active-memory limit.
        let save_outcome = match save_outcome {
            Err(publication_error) if publication_error.active_memory_deficit_bytes().is_some() => {
                let active_memory_deficit_bytes =
                    publication_error.active_memory_deficit_bytes().unwrap_or(0);
                let expert_payload_bytes_before_reclamation = model
                    .expert_weight_memory_cache_statistics()
                    .resident_payload_byte_count;
                if let Err(reclamation_error) =
                    active_request.performance_attribution.measure_operation(
                        PerformanceOperation::ExpertRetentionReclamation,
                        |_performance_attribution| {
                            qwen3_5_streaming::reclaim_retained_experts_for_request_memory_pressure(
                                model,
                                active_memory_deficit_bytes,
                            )
                        },
                    )
                {
                    return Err(PromptBlockPublicationFailure::Reclamation(
                        reclamation_error,
                    ));
                }
                let expert_payload_bytes_after_reclamation = model
                    .expert_weight_memory_cache_statistics()
                    .resident_payload_byte_count;
                if let Some(persistent_prompt_cache_diagnostics) =
                    active_request.persistent_prompt_cache_diagnostics.as_mut()
                {
                    persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_publication =
                        persistent_prompt_cache_diagnostics
                            .expert_bytes_reclaimed_for_publication
                            .saturating_add(
                                expert_payload_bytes_before_reclamation
                                    .saturating_sub(expert_payload_bytes_after_reclamation),
                            );
                }
                // Expert eviction is measured separately from serialization
                // so performance reports can attribute why publication paused.
                let mut retry_performance_attribution = std::mem::replace(
                    &mut active_request.performance_attribution,
                    PerformanceAttribution::disabled(),
                );
                let retry_outcome = persistent_prompt_cache
                    .publish_block_with_performance_attribution(
                        model.runtime(),
                        block_key,
                        parent_block_key,
                        kv_block_tensors,
                        recurrent_snapshot_tensors,
                        &mut retry_performance_attribution,
                    );
                active_request.performance_attribution = retry_performance_attribution;
                retry_outcome
            }
            save_outcome => save_outcome,
        };
        save_outcome.map_err(PromptBlockPublicationFailure::Publication)
    }
}

/// Extracts one block's KV tensors under the request's extraction measurement,
/// converting a failure into the request-stopping persistence error.
fn extract_kv_block_tensors_with_measurement(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    kv_block_start: usize,
    kv_block_end: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    match active_request.measure_operation_with_request(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        |active_request| {
            active_request
                .request_decoder_state
                .extract_persistent_prompt_cache_kv_block_tensors(
                    model.runtime(),
                    kv_block_start,
                    kv_block_end,
                    persistent_prompt_cache_block_token_count,
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
                "prompt-cache KV extraction failed"
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
fn release_publication_workspace_if_needed(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
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
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_allocator_cache()
            },
        )
        .map_err(qwen3_5_runtime_error)?;
    let memory_snapshot_after_cleanup = model.runtime().memory_snapshot().ok();
    if let Some(persistent_prompt_cache_diagnostics) =
        active_request.persistent_prompt_cache_diagnostics.as_mut()
    {
        let allocator_bytes_cleared = capture_planning::allocator_bytes_cleared(
            memory_snapshot_before_cleanup
                .as_ref()
                .map(memory_snapshot_allocator_cache_bytes),
            memory_snapshot_after_cleanup
                .as_ref()
                .map(memory_snapshot_allocator_cache_bytes),
        );
        persistent_prompt_cache_diagnostics.allocator_bytes_cleared_for_publication =
            persistent_prompt_cache_diagnostics
                .allocator_bytes_cleared_for_publication
                .saturating_add(allocator_bytes_cleared);
    }
    Ok(())
}

/// Applies one publication outcome: records diagnostics, advances the parent
/// chain on durable availability, and translates both failure owners into
/// their request-stopping errors.
fn apply_publication_outcome(
    active_request: &mut Qwen3_5EngineRequest,
    save_outcome: Result<PersistentPromptCachePublicationOutcome, PromptBlockPublicationFailure>,
    persistent_prompt_cache_block_key: PersistentPromptCacheBlockKey,
    block_start: usize,
    block_end: usize,
) -> Result<(), InferenceEngineError> {
    match save_outcome {
        Ok(publication_outcome) => {
            if publication_outcome == PersistentPromptCachePublicationOutcome::Published
                && let Some(persistent_prompt_cache_diagnostics) =
                    active_request.persistent_prompt_cache_diagnostics.as_mut()
            {
                persistent_prompt_cache_diagnostics.record_published_block();
            }
            // Both successful outcomes prove durable availability, so the
            // next block may safely use this key as its parent. Diagnostics
            // count physical publications only, not idempotent reuse.
            active_request.last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
            Ok(())
        }
        Err(PromptBlockPublicationFailure::Reclamation(error)) => Err(error),
        Err(PromptBlockPublicationFailure::Publication(error)) => {
            tracing::warn!(block_start, block_end, %error, "prompt-cache block save failed");
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

fn memory_snapshot_allocator_cache_bytes(memory_snapshot: &MlxMemorySnapshot) -> u64 {
    u64::try_from(memory_snapshot.allocator_cache_memory_bytes()).unwrap_or(u64::MAX)
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
    Publication(PersistentPromptCacheDiskStoreError),
}

#[must_use]
pub fn persistent_prompt_cache_publication_advances_parent_chain(
    publication_outcome: PersistentPromptCachePublicationOutcome,
) -> bool {
    // Kept as a public pure contract for direct tests and alternate callers:
    // there is intentionally no non-durable success variant.
    matches!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
            | PersistentPromptCachePublicationOutcome::AlreadyPublished
    )
}
