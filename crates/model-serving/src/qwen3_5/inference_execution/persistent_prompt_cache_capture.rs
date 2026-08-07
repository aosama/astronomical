use crate::{
    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore, PersistentPromptCacheWriteQueue,
    PersistentPromptCacheWriteQueueOutcome, Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
};

use super::engine_request::Qwen3_5EngineRequest;
use super::{Qwen3_5EngineState, Qwen3_5Model};

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
    ) {
        for boundary_checkpoint in boundary_checkpoints {
            if active_request.persistent_prompt_cache_capture_has_stopped {
                break;
            }
            let Some(absolute_boundary) = successful_prefill_start
                .checked_add(boundary_checkpoint.completed_prefill_chunck_tokens)
            else {
                stop_capture(active_request, "prompt-cache boundary position overflowed");
                break;
            };
            if !absolute_boundary.is_multiple_of(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
                || absolute_boundary > successful_prefill_end
            {
                stop_capture(active_request, "prompt-cache boundary position is invalid");
                break;
            }
            let Some(block_start) =
                absolute_boundary.checked_sub(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
            else {
                stop_capture(active_request, "prompt-cache block start underflowed");
                break;
            };
            let block_end = absolute_boundary;
            let block_tokens = &active_request.input_token_ids[block_start..block_end];
            let persistent_prompt_cache_block_key = match active_request
                .last_restored_persistent_prompt_cache_block_key
                .as_ref()
            {
                None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
                    persistent_prompt_cache.model_contract.model_id(),
                    persistent_prompt_cache.model_contract.model_revision(),
                    block_tokens,
                    &active_request.ordered_image_sha256_digests,
                ),
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key.for_child_block(block_tokens)
                }
            };
            let Ok(persistent_prompt_cache_block_key) = persistent_prompt_cache_block_key else {
                stop_capture(
                    active_request,
                    "prompt-cache block identity construction failed",
                );
                break;
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
                        )
                },
            ) {
                Ok(kv_block_tensors) => kv_block_tensors,
                Err(error) => {
                    tracing::warn!(block_start, block_end, %error, "prompt-cache KV extraction failed");
                    stop_capture(active_request, "prompt-cache KV extraction failed");
                    break;
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
                    stop_capture(
                        active_request,
                        "prompt-cache writer cannot accept more blocks",
                    );
                    break;
                }
                Err(error) => {
                    tracing::warn!(block_start, block_end, %error, "prompt-cache block save failed");
                    stop_capture(active_request, "prompt-cache block save failed");
                    break;
                }
            }
        }
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

fn stop_capture(active_request: &mut Qwen3_5EngineRequest, reason: &'static str) {
    active_request.persistent_prompt_cache_capture_has_stopped = true;
    tracing::info!(
        reason,
        "persistent prompt-cache capture stopped for this request"
    );
}
