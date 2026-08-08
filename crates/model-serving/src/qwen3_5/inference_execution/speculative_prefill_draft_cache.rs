use crate::{
    PerformanceAttribution, PerformanceOperation, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCachePrefixLookup,
    PersistentPromptCacheWriteQueueOutcome, Qwen3_5ExecutionError,
};

use super::super::RequestDecoderStateStack;
use super::super::model::{Qwen3_5Model, Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock};
use super::Qwen3_5EngineState;

pub(super) struct SpeculativePrefillDraftPersistentPrefixRestoreOutcome {
    pub(super) restored_token_count: usize,
    pub(super) last_restored_persistent_prompt_cache_block_key: PersistentPromptCacheBlockKey,
}

impl Qwen3_5EngineState {
    /// Restores the drafter's longest dense SSD prefix independently of target state.
    ///
    /// The scorer forwards the remaining suffix at the returned position and
    /// then scores the complete restored-plus-new prompt key range.
    pub(super) fn restore_speculative_prefill_draft_persistent_prefix(
        &self,
        draft_model: &Qwen3_5Model,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        draft_request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<SpeculativePrefillDraftPersistentPrefixRestoreOutcome>, Qwen3_5ExecutionError>
    {
        let Some(draft_persistent_prompt_cache) = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
        else {
            return Ok(None);
        };
        let persistent_prompt_cache_prefix_lookup_result = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCachePrefixLookup,
                |_performance_attribution| {
                    PersistentPromptCachePrefixLookup::for_prompt_with_image_digests(
                        draft_persistent_prompt_cache
                            .model_contract_ref()
                            .model_id(),
                        draft_persistent_prompt_cache
                            .model_contract_ref()
                            .model_revision(),
                        prompt_token_ids,
                        ordered_image_sha256_digests,
                        |block_hash| draft_persistent_prompt_cache.has_kv_block(block_hash),
                        |block_hash| {
                            draft_persistent_prompt_cache.has_recurrent_snapshot(block_hash)
                        },
                    )
                },
            );
        let restored_token_count =
            persistent_prompt_cache_prefix_lookup_result.restored_token_count();
        if restored_token_count == 0 {
            return Ok(None);
        }

        let complete_block_count =
            restored_token_count / crate::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
        let mut persistent_prompt_cache_kv_block_tensors = Vec::with_capacity(complete_block_count);
        let mut last_restored_persistent_prompt_cache_block_key = None;
        for block_index in 0..complete_block_count {
            let persistent_prompt_cache_block_key = draft_prompt_cache_block_key(
                draft_persistent_prompt_cache,
                prompt_token_ids,
                ordered_image_sha256_digests,
                block_index,
                last_restored_persistent_prompt_cache_block_key.as_ref(),
            )?;
            let loaded_kv_block_tensors = performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheKvBlockRead,
                    |performance_attribution| {
                        draft_persistent_prompt_cache.load_kv_block(
                            draft_model.runtime(),
                            &persistent_prompt_cache_block_key,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                )?
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill drafter KV block disappeared during restore",
                })?;
            persistent_prompt_cache_kv_block_tensors.push(loaded_kv_block_tensors);
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
        }

        let recurrent_snapshot_block_key = last_restored_persistent_prompt_cache_block_key
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter restore lost its boundary key",
            })?;
        let persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
                |performance_attribution| {
                    draft_persistent_prompt_cache.load_recurrent_snapshot(
                        draft_model.runtime(),
                        recurrent_snapshot_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )?
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description:
                    "speculative-prefill drafter recurrent snapshot disappeared during restore",
            })?;

        draft_request_decoder_state.restore_from_persistent_prompt_cache_blocks(
            draft_model.runtime(),
            &persistent_prompt_cache_kv_block_tensors,
            &persistent_prompt_cache_recurrent_snapshot_tensors,
        )?;
        draft_request_decoder_state
            .materialize_restored_persistent_prompt_cache_state(draft_model.runtime())?;
        let last_restored_persistent_prompt_cache_block_key =
            last_restored_persistent_prompt_cache_block_key.ok_or(
                Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill drafter restore lost its final block key",
                },
            )?;
        Ok(Some(
            SpeculativePrefillDraftPersistentPrefixRestoreOutcome {
                restored_token_count,
                last_restored_persistent_prompt_cache_block_key,
            },
        ))
    }

    /// Enqueues dense drafter state blocks captured during draft prompt scoring.
    pub(super) fn save_speculative_prefill_draft_persistent_prompt_cache_blocks(
        &self,
        draft_model: &Qwen3_5Model,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        persistent_prompt_cache_blocks: Vec<
            Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let (Some(draft_persistent_prompt_cache), Some(draft_persistent_prompt_cache_write_queue)) = (
            self.speculative_prefill_draft_persistent_prompt_cache
                .as_ref(),
            self.speculative_prefill_draft_persistent_prompt_cache_write_queue
                .as_ref(),
        ) else {
            return;
        };

        let mut previous_persistent_prompt_cache_block_key =
            parent_persistent_prompt_cache_block_key.cloned();
        for persistent_prompt_cache_block in persistent_prompt_cache_blocks {
            let block_token_count = persistent_prompt_cache_block
                .block_end_tokens
                .saturating_sub(persistent_prompt_cache_block.block_start_tokens);
            if block_token_count != crate::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT
                || persistent_prompt_cache_block.block_end_tokens > prompt_token_ids.len()
                || persistent_prompt_cache_block.block_start_tokens
                    != persistent_prompt_cache_block
                        .block_end_tokens
                        .saturating_sub(crate::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
            {
                tracing::warn!(
                    block_start_tokens = persistent_prompt_cache_block.block_start_tokens,
                    block_end_tokens = persistent_prompt_cache_block.block_end_tokens,
                    "skipping invalid speculative-prefill drafter cache block"
                );
                break;
            }
            if persistent_prompt_cache_block.block_start_tokens > 0
                && previous_persistent_prompt_cache_block_key.is_none()
            {
                tracing::info!(
                    block_start_tokens = persistent_prompt_cache_block.block_start_tokens,
                    "stopping speculative-prefill drafter cache capture without a parent block"
                );
                break;
            }
            let persistent_prompt_cache_block_key =
                match previous_persistent_prompt_cache_block_key.as_ref() {
                    Some(parent_persistent_prompt_cache_block_key) => {
                        parent_persistent_prompt_cache_block_key.for_child_block(
                            &prompt_token_ids[persistent_prompt_cache_block.block_start_tokens
                                ..persistent_prompt_cache_block.block_end_tokens],
                        )
                    }
                    None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
                        draft_persistent_prompt_cache
                            .model_contract_ref()
                            .model_id(),
                        draft_persistent_prompt_cache
                            .model_contract_ref()
                            .model_revision(),
                        &prompt_token_ids[persistent_prompt_cache_block.block_start_tokens
                            ..persistent_prompt_cache_block.block_end_tokens],
                        ordered_image_sha256_digests,
                    ),
                };
            let persistent_prompt_cache_block_key = match persistent_prompt_cache_block_key {
                Ok(persistent_prompt_cache_block_key) => persistent_prompt_cache_block_key,
                Err(block_key_error) => {
                    tracing::warn!(
                        error = %block_key_error,
                        "stopping speculative-prefill drafter cache capture after block identity failure"
                    );
                    break;
                }
            };
            let mut block_performance_attribution =
                std::mem::replace(performance_attribution, PerformanceAttribution::disabled());
            let write_queue_outcome = draft_persistent_prompt_cache_write_queue
                .serialize_and_enqueue(
                    draft_model.runtime(),
                    &persistent_prompt_cache_block_key,
                    previous_persistent_prompt_cache_block_key.as_ref(),
                    &persistent_prompt_cache_block.kv_block_tensors,
                    &persistent_prompt_cache_block.recurrent_snapshot_tensors,
                    &mut block_performance_attribution,
                );
            *performance_attribution = block_performance_attribution;
            match write_queue_outcome {
                Ok(PersistentPromptCacheWriteQueueOutcome::Queued)
                | Ok(PersistentPromptCacheWriteQueueOutcome::AlreadyQueued) => {
                    previous_persistent_prompt_cache_block_key =
                        Some(persistent_prompt_cache_block_key);
                }
                Ok(write_queue_outcome) => {
                    tracing::info!(
                        outcome = ?write_queue_outcome,
                        "stopping speculative-prefill drafter cache capture"
                    );
                    break;
                }
                Err(write_error) => {
                    tracing::warn!(
                        error = %write_error,
                        "stopping speculative-prefill drafter cache capture after write failure"
                    );
                    break;
                }
            }
        }
    }
}

fn draft_prompt_cache_block_key(
    draft_persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    draft_prompt_prefix_token_ids: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
    block_index: usize,
    parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<PersistentPromptCacheBlockKey, Qwen3_5ExecutionError> {
    let block_start_tokens = block_index
        .checked_mul(crate::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block start overflowed",
        })?;
    let block_end_tokens = block_start_tokens
        .checked_add(crate::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block end overflowed",
        })?;
    let block_tokens = draft_prompt_prefix_token_ids
        .get(block_start_tokens..block_end_tokens)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block exceeds the prefix",
        })?;
    match parent_persistent_prompt_cache_block_key {
        Some(parent_persistent_prompt_cache_block_key) => parent_persistent_prompt_cache_block_key
            .for_child_block(block_tokens)
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter child cache block identity failed",
            }),
        None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
            draft_persistent_prompt_cache
                .model_contract_ref()
                .model_id(),
            draft_persistent_prompt_cache
                .model_contract_ref()
                .model_revision(),
            block_tokens,
            ordered_image_sha256_digests,
        )
        .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter root cache block identity failed",
        }),
    }
}
