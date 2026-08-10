//! Sparse boundary-retention policy for branched prompt reuse.
//!
//! Sequence state remains append-only at every block. Boundary state is larger
//! fixed restart state: keep the root and every fourth boundary so common-prefix
//! branches retain useful restart points while linear parents can be compacted.

/// Returns whether a block-boundary snapshot remains available for branched prompts.
///
/// The stride is validated as positive when the immutable storage contract is
/// resolved. Keeping it contract-owned makes retention, startup reconciliation,
/// and topology validation apply exactly the same branch restart policy.
#[must_use]
pub(crate) fn persistent_prompt_cache_boundary_is_common_prefix_checkpoint(
    persistent_prompt_cache_block_index: u32,
    common_prefix_checkpoint_stride_blocks: u32,
) -> bool {
    persistent_prompt_cache_block_index == 0
        || persistent_prompt_cache_block_index
            .saturating_add(1)
            .is_multiple_of(common_prefix_checkpoint_stride_blocks)
}
