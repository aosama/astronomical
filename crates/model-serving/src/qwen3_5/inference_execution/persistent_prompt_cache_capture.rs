use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore, PersistentPromptCacheWriteQueue,
    PersistentPromptCacheWriteQueueOutcome, Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
};

use super::engine_request::Qwen3_5EngineRequest;
use super::speculative_prefill_failure::configured_speculative_prefill_failure;
use super::{Qwen3_5EngineState, Qwen3_5Model};

/// Owns the user-visible failure contract for one required prompt-state write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PromptStatePersistenceOwner {
    PersistentPromptCache,
    SpeculativePrefill,
}

impl PromptStatePersistenceOwner {
    #[must_use]
    pub(super) const fn for_active_request(active_request: &Qwen3_5EngineRequest) -> Self {
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
        persistent_prompt_cache_write_queue: &PersistentPromptCacheWriteQueue,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        successful_prefill_start: usize,
        successful_prefill_end: usize,
        boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
        prompt_state_persistence_owner: PromptStatePersistenceOwner,
    ) -> Result<(), InferenceEngineError> {
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        for boundary_checkpoint in boundary_checkpoints {
            let Some(absolute_boundary) = successful_prefill_start
                .checked_add(boundary_checkpoint.completed_prefill_chunck_tokens)
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
            let persistent_prompt_cache_block_key = match active_request
                .last_restored_persistent_prompt_cache_block_key
                .as_ref()
            {
                None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
                    &persistent_prompt_cache.model_contract,
                    block_tokens,
                    &active_request.ordered_image_sha256_digests,
                ),
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key.for_child_block(block_tokens)
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
            let mut request_performance_attribution = std::mem::replace(
                &mut active_request.performance_attribution,
                PerformanceAttribution::disabled(),
            );
            let save_outcome = persistent_prompt_cache_write_queue.serialize_and_enqueue(
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
            match save_outcome {
                Ok(write_queue_outcome)
                    if persistent_prompt_cache_write_outcome_advances_parent_chain(
                        write_queue_outcome,
                    ) =>
                {
                    active_request.last_restored_persistent_prompt_cache_block_key =
                        Some(persistent_prompt_cache_block_key);
                }
                Ok(_) => {
                    let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
                    let expert_weight_memory_cache_statistics =
                        model.expert_weight_memory_cache_statistics();
                    tracing::error!(
                        request_id = active_request.request_id.value(),
                        block_start,
                        block_end,
                        block_index = persistent_prompt_cache_block_key.block_index(),
                        expert_memory_mode = ?model.expert_memory_mode(),
                        retained_expert_payload_bytes = expert_weight_memory_cache_statistics.resident_payload_byte_count,
                        maximum_retained_expert_payload_bytes = expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count,
                        retained_complete_expert_layer_count = expert_weight_memory_cache_statistics.complete_layer_count,
                        expert_eviction_count = expert_weight_memory_cache_statistics.eviction_count,
                        mlx_active_memory_bytes = mlx_memory_snapshot.as_ref().map(|snapshot| snapshot.active_memory_bytes()),
                        mlx_allocator_cache_memory_bytes = mlx_memory_snapshot.as_ref().map(|snapshot| snapshot.allocator_cache_memory_bytes()),
                        mlx_peak_memory_bytes = mlx_memory_snapshot.as_ref().map(|snapshot| snapshot.peak_memory_bytes()),
                        runtime_active_memory_limit_bytes = model.runtime().memory_limits().active_memory_limit_bytes(),
                        "required persistent prompt-cache capture rejected after queue admission evidence"
                    );
                    return Err(required_prompt_state_persistence_failure(
                        prompt_state_persistence_owner,
                        active_request,
                        "required persistent prompt-state capture",
                        "prompt-cache writer cannot accept more blocks",
                    ));
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
pub fn persistent_prompt_cache_write_outcome_advances_parent_chain(
    write_queue_outcome: PersistentPromptCacheWriteQueueOutcome,
) -> bool {
    matches!(
        write_queue_outcome,
        PersistentPromptCacheWriteQueueOutcome::Queued
            | PersistentPromptCacheWriteQueueOutcome::AlreadyQueued
    )
}
