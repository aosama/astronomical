//! Restores the longest complete persistent prompt-cache prefix into live Qwen state.
//!
//! Lookup is allocation-free. After a complete chain is identified, all sequence
//! blocks and the newest required boundary are loaded, reconstructed, synchronized,
//! and released before memory admission is repeated for only the remaining context.

use astronomical_ipc_protocol::{
    RequestId, WorkerPersistentPromptCacheExpectedBlockHashPrefix,
    WorkerPersistentPromptCacheLookupOutcome, WorkerPersistentPromptCacheMissReason,
    WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCachePrefixLookup,
};

use super::super::RequestDecoderStateStack;
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
    pub(super) persistent_prompt_cache_diagnostics: WorkerPersistentPromptCacheRequestDiagnostics,
}

impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_persistent_prompt_cache_prefix(
        &mut self,
        request_id: RequestId,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        prompt_token_ids: &[u32],
        block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheRestoreOutcome, InferenceEngineError> {
        // First decide the longest usable prefix using only the store's small
        // in-memory index. No MLX array is allocated until a whole contiguous
        // prefix is known to be available.
        let lookup_result = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePrefixLookup,
            |_performance_attribution| {
                if block_causal_inputs.is_empty() {
                    PersistentPromptCachePrefixLookup::for_prompt(
                        &persistent_prompt_cache.model_contract,
                        prompt_token_ids,
                        |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                        |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
                    )
                } else {
                    PersistentPromptCachePrefixLookup::for_prompt_with_block_causal_inputs(
                        &persistent_prompt_cache.model_contract,
                        prompt_token_ids,
                        block_causal_inputs,
                        |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                        |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
                    )
                }
            },
        );
        let restored_token_count = lookup_result.restored_token_count();
        let prompt_token_count = prompt_token_ids.len();
        let lookup_diagnostics = lookup_result.diagnostics();
        let mut persistent_prompt_cache_diagnostics = persistent_prompt_cache_request_diagnostics(
            persistent_prompt_cache.model_contract.block_token_count(),
            lookup_diagnostics,
            restored_token_count,
        );
        if restored_token_count == 0 {
            if matches!(
                lookup_diagnostics.miss_reason(),
                Some(
                    crate::PersistentPromptCacheMissReason::RootSequenceStateBlockMissing
                        | crate::PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
                )
            ) {
                persistent_prompt_cache_diagnostics.startup_cleanup_evidence =
                    persistent_prompt_cache
                        .take_startup_cleanup_evidence()
                        .map(worker_startup_cleanup_evidence);
            }
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
                startup_cleanup_evidence = ?persistent_prompt_cache_diagnostics
                    .startup_cleanup_evidence,
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
                persistent_prompt_cache_diagnostics,
            });
        }

        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        let complete_block_count = restored_token_count / persistent_prompt_cache_block_token_count;
        let mut restored_persistent_prompt_cache_block_keys =
            Vec::with_capacity(complete_block_count);
        let mut last_restored_persistent_prompt_cache_block_key = None;
        let mut persistent_prompt_cache_restore_temporary_workspace_bytes = 0_usize;
        for block_index in 0..complete_block_count {
            let block_start = block_index * persistent_prompt_cache_block_token_count;
            let block_end = block_start + persistent_prompt_cache_block_token_count;
            let persistent_prompt_cache_block_key = restored_persistent_prompt_cache_block_key(
                prompt_token_ids,
                block_start,
                block_end,
                block_index,
                block_causal_inputs,
                last_restored_persistent_prompt_cache_block_key.as_ref(),
                &persistent_prompt_cache.model_contract,
            )?;
            let sequence_state_block_file_size_bytes = persistent_prompt_cache
                .sequence_state_block_file_size_bytes(
                    &persistent_prompt_cache_block_key.block_hash(),
                )
                .unwrap_or(0);
            persistent_prompt_cache_restore_temporary_workspace_bytes =
                persistent_prompt_cache_restore_temporary_workspace_bytes.saturating_add(
                    usize::try_from(sequence_state_block_file_size_bytes).unwrap_or(usize::MAX),
                );
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key.clone());
            restored_persistent_prompt_cache_block_keys.push(persistent_prompt_cache_block_key);
        }
        if let Some(recurrent_snapshot_block_key) =
            last_restored_persistent_prompt_cache_block_key.as_ref()
        {
            let recurrent_snapshot_file_size_bytes = persistent_prompt_cache
                .recurrent_snapshot_file_size_bytes(&recurrent_snapshot_block_key.block_hash())
                .unwrap_or(0);
            persistent_prompt_cache_restore_temporary_workspace_bytes =
                persistent_prompt_cache_restore_temporary_workspace_bytes.saturating_add(
                    usize::try_from(recurrent_snapshot_file_size_bytes).unwrap_or(usize::MAX),
                );
        }
        let additional_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        let target_expert_payload_bytes_reclaimed_before_restore = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                total_context_tokens,
                persistent_prompt_cache_restore_temporary_workspace_bytes,
                additional_maximum_expert_page_reservation_bytes,
                performance_attribution,
            )?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;

        // Lookup already proved every block exists. Load and absorb one block at
        // a time so a later read failure can drop this attempt without pinning
        // the complete prefix beside seated experts.
        for (block_index, persistent_prompt_cache_block_key) in
            restored_persistent_prompt_cache_block_keys
                .into_iter()
                .enumerate()
        {
            let mut loaded_kv_block_tensors = performance_attribution
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
            performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheStateReconstruction,
                    |_performance_attribution| {
                        request_decoder_state.absorb_persistent_prompt_cache_kv_block(
                            model.runtime(),
                            &mut loaded_kv_block_tensors,
                        )
                    },
                )
                .map_err(|persistent_prompt_cache_error| {
                    fatal_engine_error(format!(
                        "failed to absorb persistent prompt-cache KV block {block_index}: \
                         {persistent_prompt_cache_error}"
                    ))
                })?;
            drop(loaded_kv_block_tensors);
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
        }
        // Sequence state is append-only across every matched block, but recurrent
        // state needs only the newest boundary corresponding to the restored end.
        let recurrent_snapshot_block_key = last_restored_persistent_prompt_cache_block_key
            .as_ref()
            .ok_or_else(|| {
                fatal_engine_error("persistent prompt-cache restore lost snapshot key")
            })?;
        let mut persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
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
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateReconstruction,
                |_performance_attribution| {
                    request_decoder_state.absorb_persistent_prompt_cache_recurrent_snapshot(
                        model.runtime(),
                        &mut persistent_prompt_cache_recurrent_snapshot_tensors,
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to restore request decoder state from {complete_block_count} \
                     persistent prompt-cache blocks: {persistent_prompt_cache_error}"
                ))
            })?;
        drop(persistent_prompt_cache_recurrent_snapshot_tensors);
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
        // The initial admission was conservative because no prefix was restored
        // yet. Re-admit against only uncached context after temporary load memory
        // is gone, allowing the ordinary request path to use reclaimed capacity.
        let remaining_context_token_count = total_context_tokens
            .checked_sub(restored_token_count)
            .ok_or_else(|| {
                fatal_engine_error(
                    "persistent prompt-cache restore exceeded the generation context",
                )
            })?;
        let target_expert_payload_bytes_reclaimed_after_restore = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                remaining_context_token_count,
                0,
                additional_maximum_expert_page_reservation_bytes,
                performance_attribution,
            )?;
        let target_expert_payload_bytes_reclaimed_during_restore =
            target_expert_payload_bytes_reclaimed_before_restore
                .saturating_add(target_expert_payload_bytes_reclaimed_after_restore);
        persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_restore =
            target_expert_payload_bytes_reclaimed_during_restore;
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
            target_expert_payload_bytes_reclaimed_during_restore,
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
            persistent_prompt_cache_diagnostics,
        })
    }
}

fn persistent_prompt_cache_request_diagnostics(
    persistent_prompt_cache_block_token_count: usize,
    lookup_diagnostics: &crate::PersistentPromptCacheLookupDiagnostics,
    restored_token_count: usize,
) -> WorkerPersistentPromptCacheRequestDiagnostics {
    // Diagnostics are bounded scalar evidence copied over IPC. Never include full
    // hashes, prompts, local paths, or model tensor details in the public log.
    let restored_block_count = restored_token_count / persistent_prompt_cache_block_token_count;
    WorkerPersistentPromptCacheRequestDiagnostics {
        lookup_outcome: if restored_token_count == 0 {
            WorkerPersistentPromptCacheLookupOutcome::Miss
        } else {
            WorkerPersistentPromptCacheLookupOutcome::Hit
        },
        block_token_count: u64::try_from(persistent_prompt_cache_block_token_count)
            .unwrap_or(u64::MAX),
        complete_prompt_block_count: u64::try_from(
            lookup_diagnostics.complete_prompt_block_count(),
        )
        .unwrap_or(u64::MAX),
        maximum_restorable_block_count: u64::try_from(
            lookup_diagnostics.maximum_restorable_block_count(),
        )
        .unwrap_or(u64::MAX),
        matched_sequence_state_block_count: u64::try_from(
            lookup_diagnostics.matched_sequence_state_block_count(),
        )
        .unwrap_or(u64::MAX),
        restored_block_count: u64::try_from(restored_block_count).unwrap_or(u64::MAX),
        first_missing_sequence_state_block_index: lookup_diagnostics
            .first_missing_sequence_state_block_index()
            .map(|block_index| u64::try_from(block_index).unwrap_or(u64::MAX)),
        miss_reason: lookup_diagnostics.miss_reason().map(worker_miss_reason),
        expected_block_hash_prefix: lookup_diagnostics
            .first_missing_sequence_state_block_hash()
            .map(WorkerPersistentPromptCacheExpectedBlockHashPrefix::from_block_hash),
        startup_cleanup_evidence: None,
        published_block_count: 0,
        allocator_bytes_cleared_for_publication: 0,
        expert_bytes_reclaimed_for_publication: 0,
        expert_bytes_reclaimed_for_restore: 0,
    }
}

fn worker_startup_cleanup_evidence(
    startup_cleanup_evidence: crate::PersistentPromptCacheStartupCleanupEvidence,
) -> WorkerPersistentPromptCacheStartupCleanupEvidence {
    WorkerPersistentPromptCacheStartupCleanupEvidence {
        interrupted_transaction_recovery: worker_startup_cleanup_category(
            startup_cleanup_evidence.interrupted_transaction_recovery,
        ),
        obsolete_format: worker_startup_cleanup_category(startup_cleanup_evidence.obsolete_format),
        corrupt_current_format: worker_startup_cleanup_category(
            startup_cleanup_evidence.corrupt_current_format,
        ),
        quota_eviction: worker_startup_cleanup_category(startup_cleanup_evidence.quota_eviction),
    }
}

fn worker_startup_cleanup_category(
    startup_cleanup_category: crate::PersistentPromptCacheStartupCleanupCategory,
) -> WorkerPersistentPromptCacheStartupCleanupCategory {
    WorkerPersistentPromptCacheStartupCleanupCategory {
        artifact_count: startup_cleanup_category.artifact_count,
        block_count: startup_cleanup_category.block_count,
        byte_count: startup_cleanup_category.byte_count,
    }
}

fn worker_miss_reason(
    miss_reason: crate::PersistentPromptCacheMissReason,
) -> WorkerPersistentPromptCacheMissReason {
    match miss_reason {
        crate::PersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache => {
            WorkerPersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache
        }
        crate::PersistentPromptCacheMissReason::RootSequenceStateBlockMissing => {
            WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing
        }
        crate::PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing => {
            WorkerPersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
        }
    }
}

fn restored_persistent_prompt_cache_block_key(
    prompt_token_ids: &[u32],
    block_start: usize,
    block_end: usize,
    block_index: usize,
    block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
    last_restored_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_model_contract: &crate::PersistentPromptCacheModelContract,
) -> Result<PersistentPromptCacheBlockKey, InferenceEngineError> {
    let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
    let block_causal_input = if block_causal_inputs.is_empty() {
        &empty_block_causal_input
    } else {
        block_causal_inputs.get(block_index).ok_or_else(|| {
            fatal_engine_error(
                "persistent prompt-cache causal input plan does not cover restored block",
            )
        })?
    };
    if block_index == 0 {
        // Root blocks are explicitly bound to the pinned model identity. Child
        // keys inherit that binding through their parent hash chain.
        return PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
            persistent_prompt_cache_model_contract,
            &prompt_token_ids[block_start..block_end],
            block_causal_input,
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
        .for_child_block_with_causal_input(
            &prompt_token_ids[block_start..block_end],
            block_causal_input,
        )
        .map_err(|_| {
            fatal_engine_error(
                "persistent prompt-cache block identity construction failed during restore",
            )
        })
}
