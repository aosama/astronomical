//! The block-aware floor applied before the greedy pass.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::sensitivity::SensitivityResult;

/// Promotes per-block components from `lowest` to `second_lowest`.
///
/// Work to port (OptiQ's `_apply_block_aware_floor`):
/// 1. Group results by transformer block index (modules outside blocks are
///    ignored), ranking each block's components by KL at the lowest width,
///    most sensitive first. Large KL at the lowest width means "this module
///    suffers most when squeezed".
/// 2. Reserve attention slots first: a full-attention block (four or more
///    attention components present) reserves two, a linear-attention or
///    single-projection block reserves one, capped at `floor_components`.
///    This is the load-bearing rule that keeps late full-attention layers
///    from collapsing to all-lowest-bit.
/// 3. Per-block budget: full-attention blocks honor `floor_components`; any
///    other block is capped at one, because a two-linear block (state-space
///    mixer `in_proj`/`out_proj`, MLP `up_proj`/`down_proj`) would otherwise
///    have both promoted and the target becomes unreachable.
/// 4. Fill the remaining slots from the block's overall ranking, skipping
///    already-picked names.
/// 5. Raise each pick that sits below `second_lowest`; return the count.
pub fn apply_block_aware_floor(
    allocation: &mut BTreeMap<String, BitWidth>,
    results: &[SensitivityResult],
    lowest: BitWidth,
    second_lowest: BitWidth,
    floor_components: usize,
) -> usize {
    let _ = (allocation, results, lowest, second_lowest, floor_components);
    todo!("rank per block, reserve attention slots, fill, and promote as documented above")
}
