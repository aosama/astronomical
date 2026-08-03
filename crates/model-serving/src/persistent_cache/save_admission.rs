/// Decision for one persistent prompt-cache block save attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCacheBlockSaveAdmission {
    /// Save can append or replace the block without removing any tracked block.
    SaveWithoutEviction,
    /// Save can proceed, then the store should evict old unrelated blocks to fit.
    SaveAndEvictOldBlocksToFit,
    /// Saving this block would make the restorable prefix exceed the cache budget.
    SkipBecauseCacheIsFull,
}

const COMMON_PREFIX_RECURRENT_SNAPSHOT_STRIDE_BLOCKS: u32 = 4;

/// Returns whether a block-boundary recurrent snapshot should remain available for branched prompts.
#[must_use]
pub fn persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(
    persistent_prompt_cache_block_index: u32,
) -> bool {
    persistent_prompt_cache_block_index == 0
        || persistent_prompt_cache_block_index
            .saturating_add(1)
            .is_multiple_of(COMMON_PREFIX_RECURRENT_SNAPSHOT_STRIDE_BLOCKS)
}

/// Decides whether saving one block would preserve a useful contiguous prefix.
#[must_use]
pub fn persistent_prompt_cache_save_admission(
    tracked_persistent_prompt_cache_size_bytes: u64,
    estimated_persistent_prompt_cache_kv_block_bytes: u64,
    estimated_persistent_prompt_cache_recurrent_snapshot_bytes: u64,
    reclaimable_parent_recurrent_snapshot_bytes: u64,
    maximum_persistent_prompt_cache_size_bytes: u64,
    persistent_prompt_cache_block_index: u32,
    persistent_prompt_cache_kv_block_is_already_tracked: bool,
) -> PersistentPromptCacheBlockSaveAdmission {
    let estimated_persistent_prompt_cache_save_bytes =
        estimated_persistent_prompt_cache_kv_block_bytes
            .saturating_add(estimated_persistent_prompt_cache_recurrent_snapshot_bytes);
    if estimated_persistent_prompt_cache_save_bytes == 0
        || estimated_persistent_prompt_cache_save_bytes > maximum_persistent_prompt_cache_size_bytes
    {
        return PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull;
    }

    if persistent_prompt_cache_kv_block_is_already_tracked {
        return PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction;
    }

    let net_persistent_prompt_cache_growth_bytes = estimated_persistent_prompt_cache_save_bytes
        .saturating_sub(reclaimable_parent_recurrent_snapshot_bytes);
    if tracked_persistent_prompt_cache_size_bytes
        .saturating_add(net_persistent_prompt_cache_growth_bytes)
        <= maximum_persistent_prompt_cache_size_bytes
    {
        return PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction;
    }

    if net_persistent_prompt_cache_growth_bytes == 0 {
        return PersistentPromptCacheBlockSaveAdmission::SaveAndEvictOldBlocksToFit;
    }

    let maximum_restorable_prefix_block_count =
        maximum_persistent_prompt_cache_size_bytes / net_persistent_prompt_cache_growth_bytes;
    if u64::from(persistent_prompt_cache_block_index) < maximum_restorable_prefix_block_count {
        return PersistentPromptCacheBlockSaveAdmission::SaveAndEvictOldBlocksToFit;
    }

    PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull
}
