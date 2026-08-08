use astronomical_ipc_protocol::RequestId;

use crate::{
    InferenceEngineError, PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, PerformanceAttribution,
    PerformanceOperation, PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCachePrefixLookup,
};

use super::super::RequestDecoderStateStack;
use super::super::model::memory_admission::{
    persistent_prompt_cache_restore_temporary_workspace_bytes, validate_context_memory_admission,
};
use super::{Qwen3_5EngineState, fatal_engine_error};
/// The cache-specific portion of a newly admitted request's starting state.
///
/// `restored_token_count` drives the prefill cursor, while the u32 field is
/// reported in the public generation usage contract. Keeping the final block
/// key lets later cold-prefilled blocks extend the same content-addressed
/// chain instead of creating an unrelated root block.
pub(super) struct PersistentPromptCacheRestoreOutcome {
    pub(super) persistent_prompt_cache_token_count: u32,
    pub(super) restored_token_count: usize,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
}

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_persistent_prompt_cache_prefix(
        &mut self,
        request_id: RequestId,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheRestoreOutcome, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        // First decide the longest usable prefix using only the store's small
        // in-memory index. No MLX array is allocated until a whole contiguous
        // prefix is known to be available.
        let lookup_result = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePrefixLookup,
            |_performance_attribution| {
                PersistentPromptCachePrefixLookup::for_prompt_with_image_digests(
                    persistent_prompt_cache.model_contract.model_id(),
                    persistent_prompt_cache.model_contract.model_revision(),
                    prompt_token_ids,
                    ordered_image_sha256_digests,
                    |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                    |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
                )
            },
        );
        let restored_token_count = lookup_result.restored_token_count();
        let prompt_token_count = prompt_token_ids.len();
        let lookup_diagnostics = lookup_result.diagnostics();
        if restored_token_count == 0 {
            tracing::info!(
                request_id = request_id.value(),
                prompt_token_count,
                restored_token_count,
                restored_block_count = 0usize,
                complete_prompt_block_count = lookup_diagnostics.complete_prompt_block_count(),
                maximum_restorable_block_count = lookup_diagnostics.maximum_restorable_block_count(),
                matched_sequence_state_block_count = lookup_diagnostics
                    .matched_sequence_state_block_count(),
                first_missing_sequence_state_block_index = ?lookup_diagnostics
                    .first_missing_sequence_state_block_index(),
                newest_boundary_state_snapshot_block_index = ?lookup_diagnostics
                    .newest_boundary_state_snapshot_block_index(),
                miss_reason = ?lookup_diagnostics.miss_reason(),
                "persistent prompt-cache miss: no restorable prefix found"
            );
            self.persistent_prompt_cache_counters.record_cache_miss();
            // A miss is normal, not an error. Returning an explicit cold state
            // keeps the caller's fallback path identical for short prompts,
            // first requests, and a directory with no matching prefix.
            return Ok(PersistentPromptCacheRestoreOutcome {
                persistent_prompt_cache_token_count: 0,
                restored_token_count: 0,
                last_restored_persistent_prompt_cache_block_key: None,
            });
        }

        let persistent_prompt_cache_restore_temporary_workspace_bytes =
            persistent_prompt_cache_restore_temporary_workspace_bytes(
                self.context_memory_reservation_bytes_per_token,
                restored_token_count,
            )
            .ok_or_else(|| {
                fatal_engine_error(
                    "persistent prompt-cache restore workspace reservation overflowed",
                )
            })?;
        performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                validate_context_memory_admission(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    total_context_tokens,
                    persistent_prompt_cache_restore_temporary_workspace_bytes,
                    self.speculative_prefill_draft_maximum_expert_page_reservation_bytes(),
                )
            },
        )?;

        let mut last_restored_persistent_prompt_cache_block_key = lookup_result
            .last_restored_persistent_prompt_cache_block_key()
            .cloned();
        let complete_block_count = restored_token_count / PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
        // Load every matched block before mutating request state. If a later
        // disk read fails, this function returns an error and the caller drops
        // the entire attempt instead of running with a partially restored
        // chain of recurrent and attention state.
        let mut persistent_prompt_cache_kv_block_tensors = Vec::with_capacity(complete_block_count);
        for block_index in 0..complete_block_count {
            let block_start = block_index * PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
            let block_end = block_start + PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
            let persistent_prompt_cache_block_key = restored_persistent_prompt_cache_block_key(
                prompt_token_ids,
                block_start,
                block_end,
                block_index,
                ordered_image_sha256_digests,
                last_restored_persistent_prompt_cache_block_key.as_ref(),
                &persistent_prompt_cache.model_contract,
            )?;
            let loaded_kv_block_tensors = performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheKvBlockRead,
                    |performance_attribution| {
                        persistent_prompt_cache.load_kv_block(
                            model.runtime(),
                            &persistent_prompt_cache_block_key,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                )
                .map_err(|persistent_prompt_cache_error| {
                    fatal_engine_error(format!(
                        "failed to load persistent prompt-cache KV block {block_index}: \
                         {persistent_prompt_cache_error}"
                    ))
                })?
                .ok_or_else(|| {
                    fatal_engine_error(
                        "persistent prompt-cache KV block was reported as present \
                         but load returned None",
                    )
                })?;
            persistent_prompt_cache_kv_block_tensors.push(loaded_kv_block_tensors);
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
        }
        let recurrent_snapshot_block_key = last_restored_persistent_prompt_cache_block_key
            .as_ref()
            .ok_or_else(|| {
                fatal_engine_error("persistent prompt-cache restore lost snapshot key")
            })?;
        let persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
                |performance_attribution| {
                    persistent_prompt_cache.load_recurrent_snapshot(
                        model.runtime(),
                        recurrent_snapshot_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to load persistent prompt-cache recurrent snapshot: \
                     {persistent_prompt_cache_error}"
                ))
            })?
            .ok_or_else(|| {
                fatal_engine_error(
                    "persistent prompt-cache recurrent snapshot was reported as present \
                     but load returned None",
                )
            })?;
        // Request decoder state owns the model-specific reconstruction rules. The disk
        // package owns file concerns only; it must not decide how a Qwen3.5
        // attention or recurrent tensor becomes live model state.
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateReconstruction,
                |_performance_attribution| {
                    request_decoder_state.restore_from_persistent_prompt_cache_blocks(
                        model.runtime(),
                        &persistent_prompt_cache_kv_block_tensors,
                        &persistent_prompt_cache_recurrent_snapshot_tensors,
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to restore request decoder state from {complete_block_count} \
                     persistent prompt-cache blocks: {persistent_prompt_cache_error}"
                ))
            })?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateMaterializationSynchronizationWait,
                |_performance_attribution| {
                    request_decoder_state
                        .materialize_restored_persistent_prompt_cache_state(model.runtime())
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to materialize restored request decoder state from {complete_block_count} \
                     persistent prompt-cache blocks: {persistent_prompt_cache_error}"
                ))
            })?;
        drop(persistent_prompt_cache_kv_block_tensors);
        drop(persistent_prompt_cache_recurrent_snapshot_tensors);
        performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| model.runtime().clear_allocator_cache(),
            )
            .map_err(|runtime_error| {
                fatal_engine_error(format!(
                    "failed to clear allocator memory after persistent prompt-cache restore: \
                     {runtime_error}"
                ))
            })?;
        model.resume_expert_retention_after_request_memory_pressure();
        let remaining_context_token_count = total_context_tokens
            .checked_sub(restored_token_count)
            .ok_or_else(|| {
                fatal_engine_error(
                    "persistent prompt-cache restore exceeded the generation context",
                )
            })?;
        performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                validate_context_memory_admission(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    remaining_context_token_count,
                    0,
                    self.speculative_prefill_draft_maximum_expert_page_reservation_bytes(),
                )
            },
        )?;
        tracing::info!(
            request_id = request_id.value(),
            prompt_token_count = prompt_token_ids.len(),
            restored_token_count,
            restored_block_count = complete_block_count,
            complete_prompt_block_count = lookup_diagnostics.complete_prompt_block_count(),
            maximum_restorable_block_count = lookup_diagnostics.maximum_restorable_block_count(),
            matched_sequence_state_block_count = lookup_diagnostics
                .matched_sequence_state_block_count(),
            first_missing_sequence_state_block_index = ?lookup_diagnostics
                .first_missing_sequence_state_block_index(),
            newest_boundary_state_snapshot_block_index = ?lookup_diagnostics
                .newest_boundary_state_snapshot_block_index(),
            unrestored_matched_sequence_state_block_count = lookup_diagnostics
                .matched_sequence_state_block_count()
                .saturating_sub(complete_block_count),
            "persistent prompt-cache hit: restored prefix from disk"
        );
        let persistent_prompt_cache_token_count =
            u32::try_from(restored_token_count).map_err(|_| {
                fatal_engine_error("persistent prompt-cache token count exceeds the u32 range")
            })?;
        self.persistent_prompt_cache_counters
            .record_cache_hit(restored_token_count);
        Ok(PersistentPromptCacheRestoreOutcome {
            persistent_prompt_cache_token_count,
            restored_token_count,
            last_restored_persistent_prompt_cache_block_key,
        })
    }
}

fn restored_persistent_prompt_cache_block_key(
    prompt_token_ids: &[u32],
    block_start: usize,
    block_end: usize,
    block_index: usize,
    ordered_image_sha256_digests: &[[u8; 32]],
    last_restored_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_model_contract: &crate::PersistentPromptCacheModelContract,
) -> Result<PersistentPromptCacheBlockKey, InferenceEngineError> {
    if block_index == 0 {
        // Root blocks are explicitly bound to the pinned model identity. Child
        // keys inherit that binding through their parent hash chain.
        return PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
            persistent_prompt_cache_model_contract.model_id(),
            persistent_prompt_cache_model_contract.model_revision(),
            &prompt_token_ids[block_start..block_end],
            ordered_image_sha256_digests,
        )
        .map_err(|_| {
            fatal_engine_error(
                "persistent prompt-cache block identity construction failed during restore",
            )
        });
    }
    last_restored_persistent_prompt_cache_block_key
        // A non-root restore requires every earlier prefix block. Allowing a
        // child to continue without its parent would make its file identity
        // ambiguous and could join unrelated prompt histories.
        .ok_or_else(|| {
            fatal_engine_error(
                "persistent prompt-cache block identity chain was lost during restore",
            )
        })?
        .for_child_block(&prompt_token_ids[block_start..block_end])
        .map_err(|_| {
            fatal_engine_error(
                "persistent prompt-cache block identity construction failed during restore",
            )
        })
}
