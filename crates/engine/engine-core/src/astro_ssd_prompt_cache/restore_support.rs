//! Shared restore-side support for the `astro_ssd_prompt_cache`: the IPC
//! diagnostics mapping, restore block-key derivation, prefix enumeration, and
//! the load-and-reconstruct pipeline the text engine runs.
//!
//! The engine keeps only memory admission policy and expert retention.
//! Everything here takes the disk store and runtime by reference and holds no
//! engine state.

use std::collections::HashMap;

use ipc_protocol::{
    RequestId, WorkerAstroSsdPromptCacheExpectedBlockHashPrefix,
    WorkerAstroSsdPromptCacheLookupOutcome, WorkerAstroSsdPromptCacheMissReason,
    WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerAstroSsdPromptCacheStartupCleanupCategory,
    WorkerAstroSsdPromptCacheStartupCleanupEvidence,
};
use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use crate::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use crate::astro_ssd_prompt_cache::block_key::AstroSsdPromptCacheBlockKey;
use crate::astro_ssd_prompt_cache::model_contract::AstroSsdPromptCacheModelContract;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheLookupDiagnostics,
    AstroSsdPromptCacheMissReason, AstroSsdPromptCacheStartupCleanupEvidence, InferenceEngineError,
};
use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};

/// Builds the bounded, IPC-safe request diagnostics for one restore lookup.
///
/// Diagnostics are bounded scalar evidence copied over IPC. Never include full
/// hashes, prompts, local paths, or model tensor details in the public log.
pub fn astro_ssd_prompt_cache_request_diagnostics(
    astro_ssd_prompt_cache_block_token_count: usize,
    lookup_diagnostics: &AstroSsdPromptCacheLookupDiagnostics,
    restored_token_count: usize,
    restored_partial_tail_block_token_count: Option<usize>,
) -> WorkerAstroSsdPromptCacheRequestDiagnostics {
    let restored_block_count = restored_token_count / astro_ssd_prompt_cache_block_token_count
        + usize::from(restored_partial_tail_block_token_count.is_some());
    WorkerAstroSsdPromptCacheRequestDiagnostics {
        lookup_outcome: if restored_token_count == 0 {
            WorkerAstroSsdPromptCacheLookupOutcome::Miss
        } else {
            WorkerAstroSsdPromptCacheLookupOutcome::Hit
        },
        block_token_count: u64::try_from(astro_ssd_prompt_cache_block_token_count)
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
            .map(WorkerAstroSsdPromptCacheExpectedBlockHashPrefix::from_block_hash),
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
pub fn miss_reason_warrants_startup_cleanup_evidence(
    lookup_diagnostics: &AstroSsdPromptCacheLookupDiagnostics,
) -> bool {
    matches!(
        lookup_diagnostics.miss_reason(),
        Some(
            AstroSsdPromptCacheMissReason::RootSequenceStateBlockMissing
                | AstroSsdPromptCacheMissReason::BoundaryStateSnapshotMissing
        )
    )
}

/// Maps one startup cleanup evidence record into its IPC wire form.
pub fn worker_startup_cleanup_evidence(
    startup_cleanup_evidence: AstroSsdPromptCacheStartupCleanupEvidence,
) -> WorkerAstroSsdPromptCacheStartupCleanupEvidence {
    WorkerAstroSsdPromptCacheStartupCleanupEvidence {
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
    startup_cleanup_category: crate::AstroSsdPromptCacheStartupCleanupCategory,
) -> WorkerAstroSsdPromptCacheStartupCleanupCategory {
    WorkerAstroSsdPromptCacheStartupCleanupCategory {
        artifact_count: startup_cleanup_category.artifact_count,
        block_count: startup_cleanup_category.block_count,
        byte_count: startup_cleanup_category.byte_count,
    }
}

fn worker_miss_reason(
    miss_reason: AstroSsdPromptCacheMissReason,
) -> WorkerAstroSsdPromptCacheMissReason {
    match miss_reason {
        AstroSsdPromptCacheMissReason::PromptTooShortForAstroSsdPromptCache => {
            WorkerAstroSsdPromptCacheMissReason::PromptTooShortForAstroSsdPromptCache
        }
        AstroSsdPromptCacheMissReason::RootSequenceStateBlockMissing => {
            WorkerAstroSsdPromptCacheMissReason::RootSequenceStateBlockMissing
        }
        AstroSsdPromptCacheMissReason::BoundaryStateSnapshotMissing => {
            WorkerAstroSsdPromptCacheMissReason::BoundaryStateSnapshotMissing
        }
    }
}

/// Logs the structured miss line for a restore that found no usable prefix.
pub fn log_astro_ssd_prompt_cache_restore_miss(
    request_id: &RequestId,
    prompt_token_count: usize,
    lookup_diagnostics: &AstroSsdPromptCacheLookupDiagnostics,
    startup_cleanup_evidence: Option<&WorkerAstroSsdPromptCacheStartupCleanupEvidence>,
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
        "persistent astro-ssd-prompt-cache miss: no restorable prefix found"
    );
}

/// Logs the structured hit line for a restore that returned a prefix from disk.
pub fn log_astro_ssd_prompt_cache_restore_hit(
    request_id: &RequestId,
    prompt_token_count: usize,
    restored_token_count: usize,
    complete_block_count: usize,
    restored_partial_tail_block_key: Option<&AstroSsdPromptCacheBlockKey>,
    lookup_diagnostics: &AstroSsdPromptCacheLookupDiagnostics,
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
            .map(AstroSsdPromptCacheBlockKey::block_hash),
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
        "persistent astro-ssd-prompt-cache hit: restored prefix from disk"
    );
}

/// The block set and byte totals of one restorable prefix, before any load.
#[derive(Debug)]
pub struct RestoredPrefixBlockEnumeration {
    pub restored_astro_ssd_prompt_cache_block_keys: Vec<AstroSsdPromptCacheBlockKey>,
    pub total_sequence_block_file_size_bytes: u64,
    pub recurrent_snapshot_file_size_bytes: u64,
}

/// Derives every block key of the restored prefix and totals the transient
/// file sizes both restore phases will read.
///
/// The concat restore holds the complete source block set beside the
/// final-length destination, so the transient workspace is the sum of every
/// block, not one in-flight block. The snapshot loads only after the
/// concatenated KV materializes and releases the source blocks, so the two
/// phases never hold both at once; the workspace is the larger of the two.
pub fn enumerate_restored_prefix_blocks(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    block_causal_inputs: &[AstroSsdPromptCacheBlockCausalInput],
    restored_token_count: usize,
    restored_partial_tail_block_key: Option<&AstroSsdPromptCacheBlockKey>,
) -> Result<RestoredPrefixBlockEnumeration, InferenceEngineError> {
    let astro_ssd_prompt_cache_block_token_count =
        astro_ssd_prompt_cache.model_contract.block_token_count();
    let complete_block_count = restored_token_count / astro_ssd_prompt_cache_block_token_count;
    let mut restored_astro_ssd_prompt_cache_block_keys = Vec::with_capacity(
        complete_block_count + usize::from(restored_partial_tail_block_key.is_some()),
    );
    let mut last_restored_astro_ssd_prompt_cache_block_key = None;
    let mut total_sequence_block_file_size_bytes = 0_u64;
    for block_index in 0..complete_block_count {
        let block_start = block_index * astro_ssd_prompt_cache_block_token_count;
        let block_end = block_start + astro_ssd_prompt_cache_block_token_count;
        let astro_ssd_prompt_cache_block_key = restored_astro_ssd_prompt_cache_block_key(
            prompt_token_ids,
            block_start,
            block_end,
            block_index,
            block_causal_inputs,
            last_restored_astro_ssd_prompt_cache_block_key.as_ref(),
            &astro_ssd_prompt_cache.model_contract,
        )?;
        let sequence_state_block_file_size_bytes = astro_ssd_prompt_cache
            .sequence_state_block_file_size_bytes(&astro_ssd_prompt_cache_block_key.block_hash())
            .unwrap_or(0);
        total_sequence_block_file_size_bytes = total_sequence_block_file_size_bytes
            .saturating_add(sequence_state_block_file_size_bytes);
        last_restored_astro_ssd_prompt_cache_block_key =
            Some(astro_ssd_prompt_cache_block_key.clone());
        restored_astro_ssd_prompt_cache_block_keys.push(astro_ssd_prompt_cache_block_key);
    }
    // The tail loads like any other sequence block but never becomes a
    // chain parent: capture keeps chaining from the last complete block.
    if let Some(restored_partial_tail_block_key) = restored_partial_tail_block_key {
        let sequence_state_block_file_size_bytes = astro_ssd_prompt_cache
            .sequence_state_block_file_size_bytes(&restored_partial_tail_block_key.block_hash())
            .unwrap_or(0);
        total_sequence_block_file_size_bytes = total_sequence_block_file_size_bytes
            .saturating_add(sequence_state_block_file_size_bytes);
        restored_astro_ssd_prompt_cache_block_keys.push(restored_partial_tail_block_key.clone());
    }
    // Recurrent state needs the newest boundary at the restored end, which
    // is the tail's snapshot when a tail was restored and otherwise the
    // last complete block's snapshot.
    let recurrent_snapshot_source_block_key =
        restored_partial_tail_block_key.or(last_restored_astro_ssd_prompt_cache_block_key.as_ref());
    let recurrent_snapshot_file_size_bytes = match recurrent_snapshot_source_block_key {
        Some(recurrent_snapshot_block_key) => astro_ssd_prompt_cache
            .recurrent_snapshot_file_size_bytes(&recurrent_snapshot_block_key.block_hash())
            .unwrap_or(0),
        None => 0,
    };
    Ok(RestoredPrefixBlockEnumeration {
        restored_astro_ssd_prompt_cache_block_keys,
        total_sequence_block_file_size_bytes,
        recurrent_snapshot_file_size_bytes,
    })
}

/// One restored block's identity: root-bound for the first block, chained from
/// its parent for every later one.
pub fn restored_astro_ssd_prompt_cache_block_key(
    prompt_token_ids: &[u32],
    block_start: usize,
    block_end: usize,
    block_index: usize,
    block_causal_inputs: &[AstroSsdPromptCacheBlockCausalInput],
    last_restored_astro_ssd_prompt_cache_block_key: Option<&AstroSsdPromptCacheBlockKey>,
    astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
) -> Result<AstroSsdPromptCacheBlockKey, InferenceEngineError> {
    let empty_block_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
    let block_causal_input = if block_causal_inputs.is_empty() {
        &empty_block_causal_input
    } else {
        block_causal_inputs
            .get(block_index)
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "persistent astro-ssd-prompt-cache causal input plan does not cover restored block"
                    .to_owned(),
            })?
    };
    if block_index == 0 {
        // Root blocks are explicitly bound to the pinned model identity. Child
        // keys inherit that binding through their parent hash chain.
        return AstroSsdPromptCacheBlockKey::for_root_block_with_causal_input(
            astro_ssd_prompt_cache_model_contract,
            &prompt_token_ids[block_start..block_end],
            block_causal_input,
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "persistent astro-ssd-prompt-cache block identity construction failed during restore"
                .to_owned(),
        });
    }
    last_restored_astro_ssd_prompt_cache_block_key
        // A non-root restore requires every earlier prefix block. Allowing a
        // child to continue without its parent would make its file identity
        // ambiguous and could join unrelated prompt histories.
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "persistent astro-ssd-prompt-cache block identity chain was lost during restore"
                .to_owned(),
        })?
        .for_child_block_with_causal_input(
            &prompt_token_ids[block_start..block_end],
            block_causal_input,
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "persistent astro-ssd-prompt-cache block identity construction failed during restore"
                .to_owned(),
        })
}

/// The loaded KV tensors of a restorable prefix together with the
/// capture-parent chain head the load pass re-derived.
pub struct RestoredKvBlockLoad {
    pub restored_astro_ssd_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>>,
    pub last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
}

/// Loads every restored block's KV tensors under per-block read attribution
/// and returns the capture-parent chain head alongside them.
pub fn load_restored_kv_blocks(
    runtime: &MlxRuntime,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    restored_astro_ssd_prompt_cache_block_keys: Vec<AstroSsdPromptCacheBlockKey>,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<RestoredKvBlockLoad, InferenceEngineError> {
    let mut last_restored_astro_ssd_prompt_cache_block_key = None;
    // Lookup already proved every block exists. Load the complete block
    // set, then reconstruct each full-attention layer with one
    // concatenation, O(restored tokens); a per-block slice_update into the
    // growing final-length destination recopied the whole prefix per block
    // and forced a GPU synchronization per block.
    let mut restored_astro_ssd_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>> =
        Vec::with_capacity(restored_astro_ssd_prompt_cache_block_keys.len());
    for (block_index, astro_ssd_prompt_cache_block_key) in
        restored_astro_ssd_prompt_cache_block_keys
            .into_iter()
            .enumerate()
    {
        let loaded_kv_block_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCacheKvBlockRead,
                |performance_attribution| {
                    astro_ssd_prompt_cache.load_kv_block(
                        runtime,
                        &astro_ssd_prompt_cache_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|astro_ssd_prompt_cache_error| InferenceEngineError::Fatal {
                reason: format!(
                    "failed to load persistent astro-ssd-prompt-cache KV block {block_index}: \
                         {astro_ssd_prompt_cache_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "persistent astro-ssd-prompt-cache KV block was reported as present \
                         but load returned None"
                    .to_owned(),
            })?;
        restored_astro_ssd_prompt_cache_kv_block_tensors.push(loaded_kv_block_tensors);
        // The tail occupies the slot after the complete blocks; only
        // complete blocks extend the capture-parent chain.
        if block_index < complete_block_count {
            last_restored_astro_ssd_prompt_cache_block_key = Some(astro_ssd_prompt_cache_block_key);
        }
    }
    Ok(RestoredKvBlockLoad {
        restored_astro_ssd_prompt_cache_kv_block_tensors,
        last_restored_astro_ssd_prompt_cache_block_key,
    })
}
