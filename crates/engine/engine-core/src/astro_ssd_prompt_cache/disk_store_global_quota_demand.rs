//! The demand one commit places on the global astro SSD prompt cache quota,
//! and the eviction decisions that follow from it.
//!
//! Kept as plain arithmetic over scan-produced candidates so the decisions can
//! be tested without a disk store: the store applies them while it evicts.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::disk_store_global_quota_candidate::GlobalAstroSsdPromptCacheEvictionCandidate;

#[cfg(test)]
#[path = "disk_store_global_quota_demand_tests.rs"]
mod disk_store_global_quota_demand_tests;

/// The running byte totals of the astro SSD prompt cache while eviction
/// proceeds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GlobalQuotaTotals {
    pub(super) total_size_bytes: u64,
    pub(super) visual_embedding_total_size_bytes: u64,
}

impl GlobalQuotaTotals {
    /// Takes a removed candidate's bytes out of both totals, stopping at zero
    /// so a stale scan can never make a total wrap around.
    pub(super) fn subtract_candidate(
        &mut self,
        removed_candidate: &GlobalAstroSsdPromptCacheEvictionCandidate,
    ) {
        self.total_size_bytes = self
            .total_size_bytes
            .saturating_sub(removed_candidate.file_size_bytes());
        self.visual_embedding_total_size_bytes = self
            .visual_embedding_total_size_bytes
            .saturating_sub(removed_candidate.visual_embedding_size_bytes());
    }
}

/// What the commit in progress asks of the quota, with the ceiling it is
/// measured against.
pub(super) struct CommitQuotaDemand<'demand> {
    /// Bytes the commit adds once it lands.
    pub(super) additional_committed_size_bytes: u64,
    /// Bytes the commit makes redundant once it is durable. They are
    /// subtracted only in arithmetic; their actual file stays on disk until
    /// the commit becomes durable.
    pub(super) post_commit_reclaimable_size_bytes: u64,
    /// Block directories of the chain being extended, which no committed
    /// candidate may be removed from.
    pub(super) protected_block_directory_paths: &'demand [PathBuf],
    pub(super) maximum_size_bytes: u64,
    pub(super) global_astro_ssd_prompt_cache_root_directory: &'demand Path,
}

impl CommitQuotaDemand<'_> {
    /// The size the astro SSD prompt cache will have once the commit lands,
    /// starting from `current_committed_size_bytes`.
    pub(super) fn committed_size_after(
        &self,
        current_committed_size_bytes: u64,
    ) -> Result<u64, AstroSsdPromptCacheDiskStoreError> {
        current_committed_size_bytes
            .checked_add(self.additional_committed_size_bytes)
            .map(|size_before_post_commit_reclamation| {
                size_before_post_commit_reclamation
                    .saturating_sub(self.post_commit_reclaimable_size_bytes)
            })
            .ok_or_else(|| {
                AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheSizeOverflow {
                    global_astro_ssd_prompt_cache_root_directory: self
                        .global_astro_ssd_prompt_cache_root_directory
                        .to_path_buf(),
                }
            })
    }

    /// Whether the quota pass skips `eviction_candidate` instead of removing
    /// it: it was already removed along with an ancestor, or it is committed
    /// content that is protected or no longer needs to go because the
    /// post-commit projection already fits. Apart from the already-removed
    /// case, interrupted transactions and obsolete formats are never kept;
    /// they are removed even when the committed projection already fits.
    pub(super) fn keeps_eviction_candidate(
        &self,
        eviction_candidate: &GlobalAstroSsdPromptCacheEvictionCandidate,
        removed_eviction_paths: &HashSet<PathBuf>,
        current_committed_size_bytes: u64,
    ) -> Result<bool, AstroSsdPromptCacheDiskStoreError> {
        if eviction_candidate_was_already_removed(eviction_candidate, removed_eviction_paths) {
            return Ok(true);
        }
        if eviction_candidate.is_unconditionally_removable() {
            return Ok(false);
        }
        if eviction_candidate
            .contains_protected_block_directory(self.protected_block_directory_paths)
        {
            return Ok(true);
        }
        // Stop deleting committed value once the post-commit projection fits.
        Ok(self.committed_size_after(current_committed_size_bytes)? <= self.maximum_size_bytes)
    }

    /// Fails when the commit still would not fit under the ceiling after the
    /// pass has removed everything it may.
    pub(super) fn ensure_satisfied_at(
        &self,
        current_committed_size_bytes: u64,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let final_committed_size_bytes = self.committed_size_after(current_committed_size_bytes)?;
        if final_committed_size_bytes > self.maximum_size_bytes {
            return Err(
                AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheQuotaNotSatisfied {
                    maximum_size_bytes: self.maximum_size_bytes,
                    remaining_size_bytes: final_committed_size_bytes,
                },
            );
        }
        Ok(())
    }
}

fn eviction_candidate_was_already_removed(
    global_astro_ssd_prompt_cache_eviction_candidate: &GlobalAstroSsdPromptCacheEvictionCandidate,
    removed_eviction_paths: &HashSet<PathBuf>,
) -> bool {
    // Subtree candidates overlap by construction. A removed ancestor can cover
    // a later candidate even when their root paths differ, hence the member scan.
    removed_eviction_paths
        .contains(global_astro_ssd_prompt_cache_eviction_candidate.tie_breaker_path())
        || global_astro_ssd_prompt_cache_eviction_candidate
            .block_directory_paths()
            .iter()
            .any(|block_directory_path| removed_eviction_paths.contains(block_directory_path))
}
