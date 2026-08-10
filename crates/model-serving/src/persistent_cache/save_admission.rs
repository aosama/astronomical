/// Decision for one persistent model-state capture save attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCacheBlockSaveAdmission {
    SaveWithoutEviction,
    SaveAndReclaimParentBoundary,
    SaveAndEvictOldBlocksToFit,
    SaveReclaimParentAndEvictOldBlocksToFit,
    SkipBecauseCacheIsFull,
}

#[cfg(feature = "direct-mlx")]
const COMMON_PREFIX_RECURRENT_SNAPSHOT_STRIDE_BLOCKS: u32 = 4;

/// Returns whether a block-boundary recurrent snapshot remains available for branched prompts.
#[cfg(feature = "direct-mlx")]
#[must_use]
pub fn persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(
    persistent_prompt_cache_block_index: u32,
) -> bool {
    persistent_prompt_cache_block_index == 0
        || persistent_prompt_cache_block_index
            .saturating_add(1)
            .is_multiple_of(COMMON_PREFIX_RECURRENT_SNAPSHOT_STRIDE_BLOCKS)
}

impl PersistentPromptCacheBlockSaveAdmission {
    #[must_use]
    pub const fn should_reclaim_parent_boundary(self) -> bool {
        matches!(
            self,
            Self::SaveAndReclaimParentBoundary | Self::SaveReclaimParentAndEvictOldBlocksToFit
        )
    }
}

/// Decides retention from exact tracked bytes and current quota pressure.
#[must_use]
pub fn persistent_prompt_cache_save_admission(
    tracked_persistent_prompt_cache_size_bytes: u64,
    estimated_sequence_state_bytes: u64,
    estimated_boundary_state_bytes: u64,
    reclaimable_parent_boundary_state_bytes: u64,
    maximum_persistent_prompt_cache_size_bytes: u64,
    sequence_state_is_already_tracked: bool,
    boundary_state_is_already_tracked: bool,
) -> PersistentPromptCacheBlockSaveAdmission {
    let new_sequence_state_bytes = if sequence_state_is_already_tracked {
        0
    } else {
        estimated_sequence_state_bytes
    };
    let new_boundary_state_bytes = if boundary_state_is_already_tracked {
        0
    } else {
        estimated_boundary_state_bytes
    };
    let new_capture_bytes = new_sequence_state_bytes.saturating_add(new_boundary_state_bytes);
    if new_capture_bytes == 0 {
        return PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction;
    }
    if new_capture_bytes > maximum_persistent_prompt_cache_size_bytes {
        return PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull;
    }
    // Preserve the parent whenever the new capture already fits. It is a useful restore point
    // for prompts that branch before this child, so reclamation is a pressure response rather
    // than a normal replacement policy.
    if tracked_persistent_prompt_cache_size_bytes.saturating_add(new_capture_bytes)
        <= maximum_persistent_prompt_cache_size_bytes
    {
        return PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction;
    }
    let size_after_parent_reclamation = tracked_persistent_prompt_cache_size_bytes
        .saturating_add(new_capture_bytes)
        .saturating_sub(reclaimable_parent_boundary_state_bytes);
    // A non-checkpoint parent boundary is superseded by this child. Reclaim it before global
    // eviction so extending one prompt chain does not discard unrelated reusable prefixes.
    if reclaimable_parent_boundary_state_bytes > 0
        && size_after_parent_reclamation <= maximum_persistent_prompt_cache_size_bytes
    {
        return PersistentPromptCacheBlockSaveAdmission::SaveAndReclaimParentBoundary;
    }
    if reclaimable_parent_boundary_state_bytes > 0 {
        return PersistentPromptCacheBlockSaveAdmission::SaveReclaimParentAndEvictOldBlocksToFit;
    }
    PersistentPromptCacheBlockSaveAdmission::SaveAndEvictOldBlocksToFit
}
