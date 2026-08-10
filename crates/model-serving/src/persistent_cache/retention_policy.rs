//! Sparse boundary-retention policy for branched prompt reuse.
//!
//! Sequence state remains append-only at every block. Boundary state is larger
//! fixed restart state: keep the root and every fourth boundary so common-prefix
//! branches retain useful restart points while linear parents can be compacted.

const COMMON_PREFIX_BOUNDARY_CHECKPOINT_STRIDE_BLOCKS: u32 = 4;

/// Returns whether a block-boundary snapshot remains available for branched prompts.
#[must_use]
pub(crate) fn persistent_prompt_cache_boundary_is_common_prefix_checkpoint(
    persistent_prompt_cache_block_index: u32,
) -> bool {
    persistent_prompt_cache_block_index == 0
        || persistent_prompt_cache_block_index
            .saturating_add(1)
            .is_multiple_of(COMMON_PREFIX_BOUNDARY_CHECKPOINT_STRIDE_BLOCKS)
}
