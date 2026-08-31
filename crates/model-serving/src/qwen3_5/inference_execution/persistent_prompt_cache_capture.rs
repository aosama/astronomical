//! Qwen3.5 request-side extraction and required synchronous cache publication.
//!
//! The request cursor advances only after the matching block is durably published
//! or an exact block was already present. Capacity pressure gets one narrowly
//! typed retry after allocator and pageable-expert reclamation; storage, quota,
//! validation, and topology failures stop the request without retry.

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCachePublicationOutcome, Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
};

use super::engine_request::Qwen3_5EngineRequest;
use super::speculative_prefill::configured_speculative_prefill_failure;
use super::{Qwen3_5EngineState, Qwen3_5Model, qwen3_5_runtime_error};
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;

/// Owns the user-visible failure contract for one required prompt-state write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PromptStatePersistenceOwner {
    /// Ordinary dense prompt-cache state; failure names that user-visible feature.
    PersistentPromptCache,
    /// Selection-bound state created while configured SpecPrefill is active.
    SpeculativePrefill,
}

impl PromptStatePersistenceOwner {
    #[must_use]
    pub(super) const fn for_active_request(active_request: &Qwen3_5EngineRequest) -> Self {
        // Both paths use the same synchronous publication machinery, but their
        // no-fallback error contracts must remain distinguishable to the user.
        if active_request.should_use_speculative_prefill {
            return Self::SpeculativePrefill;
        }
        Self::PersistentPromptCache
    }
}

/// Converts a required persistence failure according to the operation that owns it.
pub(super) fn required_prompt_state_persistence_failure(
    prompt_state_persistence_owner: PromptStatePersistenceOwner,
    active_request: &Qwen3_5EngineRequest,
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    // Translate only at this shared boundary; low-level storage code remains
    // independent of whichever prompt-processing policy requested publication.
    match prompt_state_persistence_owner {
        PromptStatePersistenceOwner::SpeculativePrefill => configured_speculative_prefill_failure(
            active_request.request_id,
            failure_stage,
            internal_error,
        ),
        PromptStatePersistenceOwner::PersistentPromptCache => {
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
        prompt_state_persistence_owner: PromptStatePersistenceOwner,
    ) -> Result<(), InferenceEngineError> {
        // Boundary checkpoints are emitted by the successful forward. Recompute
        // absolute positions here and validate them before slicing user tokens;
        // never trust a model-specific checkpoint to be aligned implicitly.
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        for boundary_checkpoint in boundary_checkpoints {
            let Some(absolute_boundary) = successful_prefill_start
                .checked_add(boundary_checkpoint.completed_prefill_chunk_tokens)
            else {
                return Err(required_prompt_state_persistence_failure(
                    prompt_state_persistence_owner,
                    active_request,
                    "required persistent prompt-state capture",
                    "prompt-cache boundary position overflowed",
                ));
            };
            if !absolute_boundary.is_multiple_of(persistent_prompt_cache_block_token_count)
                || absolute_boundary > successful_prefill_end
            {
                return Err(required_prompt_state_persistence_failure(
                    prompt_state_persistence_owner,
                    active_request,
                    "required persistent prompt-state capture",
                    "prompt-cache boundary position is invalid",
                ));
            }
            let Some(block_start) =
                absolute_boundary.checked_sub(persistent_prompt_cache_block_token_count)
            else {
                return Err(required_prompt_state_persistence_failure(
                    prompt_state_persistence_owner,
                    active_request,
                    "required persistent prompt-state capture",
                    "prompt-cache block start underflowed",
                ));
            };
            let block_end = absolute_boundary;
            let block_tokens = &active_request.input_token_ids[block_start..block_end];
            let block_index = block_start / persistent_prompt_cache_block_token_count;
            let empty_block_causal_input = crate::PersistentPromptCacheBlockCausalInput::empty();
            let block_causal_input = if active_request
                .persistent_prompt_cache_block_causal_inputs
                .is_empty()
            {
                &empty_block_causal_input
            } else {
                let Some(block_causal_input) = active_request
                    .persistent_prompt_cache_block_causal_inputs
                    .get(block_index)
                else {
                    return Err(required_prompt_state_persistence_failure(
                        prompt_state_persistence_owner,
                        active_request,
                        "required persistent prompt-state capture",
                        "prompt-cache causal input plan does not cover captured block",
                    ));
                };
                block_causal_input
            };
            // Each block binds only new causal inputs; descendants inherit them through ancestry.
            let persistent_prompt_cache_block_key = match active_request
                .last_restored_persistent_prompt_cache_block_key
                .as_ref()
            {
                None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
                    &persistent_prompt_cache.model_contract,
                    block_tokens,
                    block_causal_input,
                ),
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key
                        .for_child_block_with_causal_input(block_tokens, block_causal_input)
                }
            };
            let Ok(persistent_prompt_cache_block_key) = persistent_prompt_cache_block_key else {
                return Err(required_prompt_state_persistence_failure(
                    prompt_state_persistence_owner,
                    active_request,
                    "required persistent prompt-state capture",
                    "prompt-cache block identity construction failed",
                ));
            };
            // Extraction returns MLX arrays referencing exact decoder state. Keep
            // these same arrays alive through a possible retry; recapturing after
            // reclamation could observe mutated request state or duplicate work.
            let kv_block_tensors = match active_request.measure_operation_with_request(
                PerformanceOperation::PersistentPromptCacheStateExtraction,
                |active_request| {
                    active_request
                        .request_decoder_state
                        .extract_persistent_prompt_cache_kv_block_tensors(
                            model.runtime(),
                            block_start,
                            block_end,
                            persistent_prompt_cache_block_token_count,
                        )
                },
            ) {
                Ok(kv_block_tensors) => kv_block_tensors,
                Err(error) => {
                    tracing::warn!(block_start, block_end, %error, "prompt-cache KV extraction failed");
                    return Err(required_prompt_state_persistence_failure(
                        prompt_state_persistence_owner,
                        active_request,
                        "required persistent prompt-state capture",
                        error,
                    ));
                }
            };
            // Synchronize before clearing allocator cache because submitted GPU
            // work may still own cached buffers. Publication needs contiguous
            // workspace for its largest tensor materialization.
            let memory_snapshot_before_cleanup = model.runtime().memory_snapshot().ok();
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
                let allocator_bytes_cleared = memory_snapshot_before_cleanup
                    .as_ref()
                    .map_or(0, |memory_snapshot| {
                        u64::try_from(memory_snapshot.allocator_cache_memory_bytes())
                            .unwrap_or(u64::MAX)
                    })
                    .saturating_sub(memory_snapshot_after_cleanup.as_ref().map_or(
                        0,
                        |memory_snapshot| {
                            u64::try_from(memory_snapshot.allocator_cache_memory_bytes())
                                .unwrap_or(u64::MAX)
                        },
                    ));
                persistent_prompt_cache_diagnostics.allocator_bytes_cleared_for_publication =
                    persistent_prompt_cache_diagnostics
                        .allocator_bytes_cleared_for_publication
                        .saturating_add(allocator_bytes_cleared);
            }
            // The disk-store API needs mutable attribution while the request also
            // remains mutably borrowed. Move the owner out temporarily and put it
            // back on every return path before interpreting publication outcome.
            let mut request_performance_attribution = std::mem::replace(
                &mut active_request.performance_attribution,
                PerformanceAttribution::disabled(),
            );
            let save_outcome = persistent_prompt_cache.publish_block_with_performance_attribution(
                model.runtime(),
                &persistent_prompt_cache_block_key,
                active_request
                    .last_restored_persistent_prompt_cache_block_key
                    .as_ref(),
                &kv_block_tensors,
                &boundary_checkpoint.recurrent_snapshot_tensors,
                &mut request_performance_attribution,
            );
            active_request.performance_attribution = request_performance_attribution;
            // Retry exactly once and only for the typed MLX active-memory limit.
            // Reusing `kv_block_tensors` and the checkpoint tensors is essential:
            // this is resource reclamation, not a second logical capture.
            let save_outcome = match save_outcome {
                Err(publication_error)
                    if publication_error.active_memory_deficit_bytes().is_some() =>
                {
                    let active_memory_deficit_bytes =
                        publication_error.active_memory_deficit_bytes().unwrap_or(0);
                    let expert_payload_bytes_before_reclamation = model
                        .expert_weight_memory_cache_statistics()
                        .resident_payload_byte_count;
                    active_request.performance_attribution.measure_operation(
                        PerformanceOperation::ExpertRetentionReclamation,
                        |_performance_attribution| {
                            reclaim_retained_experts_for_request_memory_pressure(
                                model,
                                active_memory_deficit_bytes,
                            )
                        },
                    )?;
                    let expert_payload_bytes_after_reclamation = model
                        .expert_weight_memory_cache_statistics()
                        .resident_payload_byte_count;
                    if let Some(persistent_prompt_cache_diagnostics) =
                        active_request.persistent_prompt_cache_diagnostics.as_mut()
                    {
                        persistent_prompt_cache_diagnostics
                            .expert_bytes_reclaimed_for_publication =
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
                            &persistent_prompt_cache_block_key,
                            active_request
                                .last_restored_persistent_prompt_cache_block_key
                                .as_ref(),
                            &kv_block_tensors,
                            &boundary_checkpoint.recurrent_snapshot_tensors,
                            &mut retry_performance_attribution,
                        );
                    active_request.performance_attribution = retry_performance_attribution;
                    retry_outcome
                }
                save_outcome => save_outcome,
            };
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
                }
                Err(error) => {
                    tracing::warn!(block_start, block_end, %error, "prompt-cache block save failed");
                    return Err(required_prompt_state_persistence_failure(
                        prompt_state_persistence_owner,
                        active_request,
                        "required persistent prompt-state capture",
                        error,
                    ));
                }
            }
        }
        Ok(())
    }
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
