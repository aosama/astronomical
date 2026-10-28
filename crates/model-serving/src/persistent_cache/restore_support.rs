//! Shared restore-side support for the persistent prompt cache: the IPC
//! diagnostics mapping, restore block-key derivation, prefix enumeration, and
//! the load-and-reconstruct pipeline that the streaming and resident engines
//! previously carried as two large near-identical functions.
//!
//! The engines keep only what genuinely differs: memory admission policy,
//! expert retention, and the resident engine's in-memory prefix adoption.
//! Everything here takes the disk store and runtime by reference and holds no
//! engine state, so one correction lands for both engines at once.

use std::collections::HashMap;

use astronomical_ipc_protocol::{
    RequestId, WorkerPersistentPromptCacheExpectedBlockHashPrefix,
    WorkerPersistentPromptCacheLookupOutcome, WorkerPersistentPromptCacheMissReason,
    WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::block_key::PersistentPromptCacheBlockKey;
use crate::persistent_cache::model_contract::PersistentPromptCacheModelContract;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheDiskStore, PersistentPromptCacheLookupDiagnostics,
    PersistentPromptCacheMissReason, PersistentPromptCacheStartupCleanupEvidence,
};

/// Builds the bounded, IPC-safe request diagnostics for one restore lookup.
///
/// Diagnostics are bounded scalar evidence copied over IPC. Never include full
/// hashes, prompts, local paths, or model tensor details in the public log.
pub(crate) fn persistent_prompt_cache_request_diagnostics(
    persistent_prompt_cache_block_token_count: usize,
    lookup_diagnostics: &PersistentPromptCacheLookupDiagnostics,
    restored_token_count: usize,
    restored_partial_tail_block_token_count: Option<usize>,
) -> WorkerPersistentPromptCacheRequestDiagnostics {
    let restored_block_count = restored_token_count / persistent_prompt_cache_block_token_count
        + usize::from(restored_partial_tail_block_token_count.is_some());
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
        partial_tail_block_token_count: restored_partial_tail_block_token_count.map(
            |partial_tail_block_token_count| {
                u64::try_from(partial_tail_block_token_count).unwrap_or(u64::MAX)
            },
        ),
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
        in_memory_prefix_adoption: None,
    }
}

/// Whether a lookup miss should attach the store's startup cleanup evidence:
/// only the two miss reasons a startup sweep can actually cure.
pub(crate) fn miss_reason_warrants_startup_cleanup_evidence(
    lookup_diagnostics: &PersistentPromptCacheLookupDiagnostics,
) -> bool {
    matches!(
        lookup_diagnostics.miss_reason(),
        Some(
            PersistentPromptCacheMissReason::RootSequenceStateBlockMissing
                | PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
        )
    )
}

/// Maps one startup cleanup evidence record into its IPC wire form.
pub(crate) fn worker_startup_cleanup_evidence(
    startup_cleanup_evidence: PersistentPromptCacheStartupCleanupEvidence,
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
    miss_reason: PersistentPromptCacheMissReason,
) -> WorkerPersistentPromptCacheMissReason {
    match miss_reason {
        PersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache => {
            WorkerPersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache
        }
        PersistentPromptCacheMissReason::RootSequenceStateBlockMissing => {
            WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing
        }
        PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing => {
            WorkerPersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
        }
    }
}

/// Logs the structured miss line for a restore that found no usable prefix.
pub(crate) fn log_persistent_prompt_cache_restore_miss(
    request_id: &RequestId,
    prompt_token_count: usize,
    lookup_diagnostics: &PersistentPromptCacheLookupDiagnostics,
    startup_cleanup_evidence: Option<&WorkerPersistentPromptCacheStartupCleanupEvidence>,
) {
    tracing::info!(
        request_id = request_id.value(),
        prompt_token_count,
        restored_token_count = 0usize,
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
        startup_cleanup_evidence = ?startup_cleanup_evidence,
        "persistent prompt-cache miss: no restorable prefix found"
    );
}

/// Logs the structured hit line for a restore that returned a prefix from disk.
pub(crate) fn log_persistent_prompt_cache_restore_hit(
    request_id: &RequestId,
    prompt_token_count: usize,
    restored_token_count: usize,
    complete_block_count: usize,
    restored_partial_tail_block_key: Option<&PersistentPromptCacheBlockKey>,
    lookup_diagnostics: &PersistentPromptCacheLookupDiagnostics,
    target_expert_payload_bytes_reclaimed_during_restore: u64,
) {
    tracing::info!(
        request_id = request_id.value(),
        prompt_token_count,
        restored_token_count,
        restored_block_count = complete_block_count
            + usize::from(restored_partial_tail_block_key.is_some()),
        restored_partial_tail_block_token_count = ?lookup_diagnostics
            .restored_partial_tail_block_token_count(),
        restored_partial_tail_block_hash = ?restored_partial_tail_block_key
            .map(PersistentPromptCacheBlockKey::block_hash),
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
}

/// The block set and byte totals of one restorable prefix, before any load.
#[derive(Debug)]
pub(crate) struct RestoredPrefixBlockEnumeration {
    pub(crate) restored_persistent_prompt_cache_block_keys: Vec<PersistentPromptCacheBlockKey>,
    pub(crate) total_sequence_block_file_size_bytes: u64,
    pub(crate) recurrent_snapshot_file_size_bytes: u64,
}

/// Derives every block key of the restored prefix and totals the transient
/// file sizes both restore phases will read.
///
/// The concat restore holds the complete source block set beside the
/// final-length destination, so the transient workspace is the sum of every
/// block, not one in-flight block. The snapshot loads only after the
/// concatenated KV materializes and releases the source blocks, so the two
/// phases never hold both at once; the workspace is the larger of the two.
pub(crate) fn enumerate_restored_prefix_blocks(
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
    restored_token_count: usize,
    restored_partial_tail_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<RestoredPrefixBlockEnumeration, InferenceEngineError> {
    let persistent_prompt_cache_block_token_count =
        persistent_prompt_cache.model_contract.block_token_count();
    let complete_block_count = restored_token_count / persistent_prompt_cache_block_token_count;
    let mut restored_persistent_prompt_cache_block_keys = Vec::with_capacity(
        complete_block_count + usize::from(restored_partial_tail_block_key.is_some()),
    );
    let mut last_restored_persistent_prompt_cache_block_key = None;
    let mut total_sequence_block_file_size_bytes = 0_u64;
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
            .sequence_state_block_file_size_bytes(&persistent_prompt_cache_block_key.block_hash())
            .unwrap_or(0);
        total_sequence_block_file_size_bytes = total_sequence_block_file_size_bytes
            .saturating_add(sequence_state_block_file_size_bytes);
        last_restored_persistent_prompt_cache_block_key =
            Some(persistent_prompt_cache_block_key.clone());
        restored_persistent_prompt_cache_block_keys.push(persistent_prompt_cache_block_key);
    }
    // The tail loads like any other sequence block but never becomes a
    // chain parent: capture keeps chaining from the last complete block.
    if let Some(restored_partial_tail_block_key) = restored_partial_tail_block_key {
        let sequence_state_block_file_size_bytes = persistent_prompt_cache
            .sequence_state_block_file_size_bytes(&restored_partial_tail_block_key.block_hash())
            .unwrap_or(0);
        total_sequence_block_file_size_bytes = total_sequence_block_file_size_bytes
            .saturating_add(sequence_state_block_file_size_bytes);
        restored_persistent_prompt_cache_block_keys.push(restored_partial_tail_block_key.clone());
    }
    // Recurrent state needs the newest boundary at the restored end, which
    // is the tail's snapshot when a tail was restored and otherwise the
    // last complete block's snapshot.
    let recurrent_snapshot_source_block_key = restored_partial_tail_block_key
        .or(last_restored_persistent_prompt_cache_block_key.as_ref());
    let recurrent_snapshot_file_size_bytes = match recurrent_snapshot_source_block_key {
        Some(recurrent_snapshot_block_key) => persistent_prompt_cache
            .recurrent_snapshot_file_size_bytes(&recurrent_snapshot_block_key.block_hash())
            .unwrap_or(0),
        None => 0,
    };
    Ok(RestoredPrefixBlockEnumeration {
        restored_persistent_prompt_cache_block_keys,
        total_sequence_block_file_size_bytes,
        recurrent_snapshot_file_size_bytes,
    })
}

/// One restored block's identity: root-bound for the first block, chained from
/// its parent for every later one.
pub(crate) fn restored_persistent_prompt_cache_block_key(
    prompt_token_ids: &[u32],
    block_start: usize,
    block_end: usize,
    block_index: usize,
    block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
    last_restored_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<PersistentPromptCacheBlockKey, InferenceEngineError> {
    let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
    let block_causal_input = if block_causal_inputs.is_empty() {
        &empty_block_causal_input
    } else {
        block_causal_inputs
            .get(block_index)
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "persistent prompt-cache causal input plan does not cover restored block"
                    .to_owned(),
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
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "persistent prompt-cache block identity construction failed during restore"
                .to_owned(),
        });
    }
    last_restored_persistent_prompt_cache_block_key
        // A non-root restore requires every earlier prefix block. Allowing a
        // child to continue without its parent would make its file identity
        // ambiguous and could join unrelated prompt histories.
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "persistent prompt-cache block identity chain was lost during restore"
                .to_owned(),
        })?
        .for_child_block_with_causal_input(
            &prompt_token_ids[block_start..block_end],
            block_causal_input,
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "persistent prompt-cache block identity construction failed during restore"
                .to_owned(),
        })
}

/// The loaded KV tensors of a restorable prefix together with the
/// capture-parent chain head the load pass re-derived.
pub(crate) struct RestoredKvBlockLoad {
    pub(crate) restored_persistent_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>>,
    pub(crate) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
}

/// Loads every restored block's KV tensors under per-block read attribution
/// and returns the capture-parent chain head alongside them.
pub(crate) fn load_restored_kv_blocks(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    restored_persistent_prompt_cache_block_keys: Vec<PersistentPromptCacheBlockKey>,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<RestoredKvBlockLoad, InferenceEngineError> {
    let mut last_restored_persistent_prompt_cache_block_key = None;
    // Lookup already proved every block exists. Load the complete block
    // set, then reconstruct each full-attention layer with one
    // concatenation, O(restored tokens); a per-block slice_update into the
    // growing final-length destination recopied the whole prefix per block
    // and forced a GPU synchronization per block.
    let mut restored_persistent_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>> =
        Vec::with_capacity(restored_persistent_prompt_cache_block_keys.len());
    for (block_index, persistent_prompt_cache_block_key) in
        restored_persistent_prompt_cache_block_keys
            .into_iter()
            .enumerate()
    {
        let loaded_kv_block_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheKvBlockRead,
                |performance_attribution| {
                    persistent_prompt_cache.load_kv_block(
                        runtime,
                        &persistent_prompt_cache_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(
                |persistent_prompt_cache_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "failed to load persistent prompt-cache KV block {block_index}: \
                         {persistent_prompt_cache_error}"
                    ),
                },
            )?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "persistent prompt-cache KV block was reported as present \
                         but load returned None"
                    .to_owned(),
            })?;
        restored_persistent_prompt_cache_kv_block_tensors.push(loaded_kv_block_tensors);
        // The tail occupies the slot after the complete blocks; only
        // complete blocks extend the capture-parent chain.
        if block_index < complete_block_count {
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
        }
    }
    Ok(RestoredKvBlockLoad {
        restored_persistent_prompt_cache_kv_block_tensors,
        last_restored_persistent_prompt_cache_block_key,
    })
}

/// Reconstructs the full-attention KV cache from the loaded blocks with one
/// concatenation per layer, then releases the source blocks: the snapshot
/// phase must never hold the block set and the destination at once.
pub(crate) fn restore_concatenated_full_attention_state(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
    restored_persistent_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>>,
    restored_token_count: usize,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let reconstruction_result = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateReconstruction,
        |_performance_attribution| {
            request_decoder_state.restore_full_attention_kv_concat(
                runtime,
                &restored_persistent_prompt_cache_kv_block_tensors,
                restored_token_count,
            )
        },
    );
    drop(restored_persistent_prompt_cache_kv_block_tensors);
    reconstruction_result.map_err(
        |persistent_prompt_cache_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to restore request decoder state from {complete_block_count} \
                 persistent prompt-cache blocks: {persistent_prompt_cache_error}"
            ),
        },
    )
}

/// Loads the newest recurrent snapshot and absorbs it into live state.
///
/// Sequence state is append-only across every matched block, but recurrent
/// state needs only the newest boundary corresponding to the restored end.
pub(crate) fn absorb_newest_recurrent_snapshot(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    request_decoder_state: &mut RequestDecoderStateStack,
    recurrent_snapshot_block_key: &PersistentPromptCacheBlockKey,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let mut persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
            |performance_attribution| {
                persistent_prompt_cache.load_recurrent_snapshot(
                    runtime,
                    recurrent_snapshot_block_key,
                    performance_attribution.positional_file_read_metrics(),
                )
            },
        )
        .map_err(
            |persistent_prompt_cache_error| InferenceEngineError::Fatal {
                reason: format!(
                    "failed to load persistent prompt-cache recurrent snapshot: \
                     {persistent_prompt_cache_error}"
                ),
            },
        )?
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "persistent prompt-cache recurrent snapshot was reported as present \
                     but load returned None"
                .to_owned(),
        })?;
    let absorb_result = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateReconstruction,
        |_performance_attribution| {
            request_decoder_state.absorb_persistent_prompt_cache_recurrent_snapshot(
                runtime,
                &mut persistent_prompt_cache_recurrent_snapshot_tensors,
            )
        },
    );
    drop(persistent_prompt_cache_recurrent_snapshot_tensors);
    absorb_result.map_err(
        |persistent_prompt_cache_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to restore request decoder state from {complete_block_count} \
                 persistent prompt-cache blocks: {persistent_prompt_cache_error}"
            ),
        },
    )
}

/// Materializes the restored state on the GPU and releases every transient
/// buffer before the caller re-runs memory admission.
pub(crate) fn materialize_and_release_restored_state(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheStateMaterializationSynchronizationWait,
            |_performance_attribution| {
                request_decoder_state.materialize_restored_persistent_prompt_cache_state(runtime)
            },
        )
        .map_err(
            |persistent_prompt_cache_error| InferenceEngineError::Fatal {
                reason: format!(
                    "failed to materialize restored request decoder state from \
                     {complete_block_count} persistent prompt-cache blocks: \
                     {persistent_prompt_cache_error}"
                ),
            },
        )?;
    performance_attribution
        .measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| runtime.clear_allocator_cache(),
        )
        .map_err(|runtime_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to clear allocator memory after persistent prompt-cache restore: \
                     {runtime_error}"
            ),
        })
}
