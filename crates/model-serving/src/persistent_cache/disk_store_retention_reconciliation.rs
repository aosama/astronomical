//! Startup-only repair of retention state after an interrupted publication.
//!
//! Publication normally removes a redundant parent boundary after committing
//! its child. A crash between those operations leaves a valid but oversized
//! chain. Startup first evicts unrelated content, then finishes only this safe
//! compaction, and finally retries quota enforcement while protecting the chain.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::SystemTime;

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, remove_cache_owned_file_or_confirm_absent, synchronize_directory,
};
use super::retention_policy::persistent_prompt_cache_boundary_is_common_prefix_checkpoint;
use super::startup_cleanup_evidence::PersistentPromptCacheStartupCleanupEvidence;

impl PersistentPromptCacheDiskStore {
    pub(super) fn reconcile_startup_retention_and_global_quota(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // Every block validated into the active model index is protected during
        // the first quota pass. This lets unrelated models and stale artifacts
        // absorb pressure before any repair touches the active chain.
        let protected_active_block_directory_paths = self
            .lock_tracked_files()
            .tracked_blocks()
            .into_iter()
            .map(|(_, tracked_block)| tracked_block.block_directory_path)
            .collect::<Vec<_>>();
        match self
            .enforce_startup_global_prompt_cache_quota(&protected_active_block_directory_paths)
        {
            Ok(()) => return Ok(()),
            Err(PersistentPromptCacheDiskStoreError::GlobalPromptCacheQuotaNotSatisfied {
                ..
            }) => {}
            Err(global_quota_error) => return Err(global_quota_error),
        }
        // Reaching this point means unprotected eviction was insufficient. The
        // only bytes now eligible inside the chain are redundant non-checkpoint
        // parent boundaries left by a crash after child commit.
        self.compact_active_model_parent_boundaries_for_startup_quota()?;
        self.enforce_startup_global_prompt_cache_quota(&protected_active_block_directory_paths)
    }

    fn compact_active_model_parent_boundaries_for_startup_quota(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        if !self.model_contract.has_sequence_state() || !self.model_contract.has_boundary_state() {
            return Ok(());
        }
        let mut committed_size_bytes = self.total_size_bytes();
        let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
        if committed_size_bytes <= self.global_prompt_cache_maximum_size_bytes {
            return Ok(());
        }

        let tracked_blocks = self.lock_tracked_files().tracked_blocks();
        // A boundary is redundant only when a durable child exists. Leaf
        // boundaries remain the required restart point for that prompt prefix.
        let block_hashes_with_committed_children = tracked_blocks
            .iter()
            .filter_map(|(_, tracked_block)| tracked_block.parent_block_hash)
            .collect::<HashSet<_>>();
        let mut reclaimable_boundaries = Vec::new();
        for (block_hash, tracked_block) in tracked_blocks {
            let Some(boundary_state_file) = tracked_block.boundary_state_file else {
                continue;
            };
            // Common-prefix checkpoints deliberately retain intermediate restart
            // points even when they have children; all other parent boundaries
            // may be reconstructed from their sequence chain and newer boundary.
            if !block_hashes_with_committed_children.contains(&block_hash)
                || persistent_prompt_cache_boundary_is_common_prefix_checkpoint(
                    tracked_block.block_index,
                )
            {
                continue;
            }
            let boundary_file_metadata = std::fs::symlink_metadata(&boundary_state_file.file_path)
                .map_err(
                    |source| PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                        block_file_path: boundary_state_file.file_path.clone(),
                        source,
                    },
                )?;
            let modified_at = boundary_file_metadata.modified().map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: boundary_state_file.file_path.clone(),
                    source,
                }
            })?;
            reclaimable_boundaries.push(StartupBoundaryReclaimCandidate {
                block_hash,
                block_directory_path: tracked_block.block_directory_path,
                boundary_file_path: boundary_state_file.file_path,
                boundary_file_size_bytes: boundary_state_file.file_size_bytes,
                modified_at,
            });
        }
        reclaimable_boundaries.sort_by(|left_candidate, right_candidate| {
            left_candidate
                .modified_at
                .cmp(&right_candidate.modified_at)
                .then_with(|| {
                    left_candidate
                        .boundary_file_path
                        .cmp(&right_candidate.boundary_file_path)
                })
        });

        for reclaimable_boundary in reclaimable_boundaries {
            if committed_size_bytes <= self.global_prompt_cache_maximum_size_bytes {
                break;
            }
            remove_cache_owned_file_or_confirm_absent(&reclaimable_boundary.boundary_file_path)?;
            synchronize_directory(&reclaimable_boundary.block_directory_path)?;
            self.lock_tracked_files().remove_file(
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                &reclaimable_boundary.block_hash,
            );
            committed_size_bytes =
                committed_size_bytes.saturating_sub(reclaimable_boundary.boundary_file_size_bytes);
            startup_cleanup_evidence
                .interrupted_transaction_recovery
                .record_artifact(reclaimable_boundary.boundary_file_size_bytes);
        }
        self.record_startup_cleanup_evidence(startup_cleanup_evidence);
        self.refresh_global_prompt_cache_accounting()
    }
}

struct StartupBoundaryReclaimCandidate {
    block_hash: [u8; 32],
    block_directory_path: PathBuf,
    boundary_file_path: PathBuf,
    boundary_file_size_bytes: u64,
    modified_at: SystemTime,
}
