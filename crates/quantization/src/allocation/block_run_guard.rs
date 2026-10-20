//! The post-greedy guard against long stretches of low-bit blocks.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::sensitivity::SensitivityResult;

/// Breaks runs of consecutive low-bit-dominated blocks longer than `maximum_run`.
///
/// Work to port (OptiQ's `_enforce_block_run_limit`):
/// 1. Group allocated names by block index; a block is low-dominated when
///    more than half of its components sit at `lowest`.
/// 2. Scan block indices in ascending order for maximal runs of
///    low-dominated blocks. A run longer than `maximum_run` is broken by
///    visiting every `max(1, maximum_run)`-th block of the run (the middle
///    blocks; the edges are already covered by boundary protection) and
///    promoting that block's single most sensitive `lowest`-width component
///    (by KL at `lowest`) to `second_lowest`.
/// 3. Return the number of promotions.
///
/// This is a backstop to the floor; it only fires when an entire stretch is
/// still misallocated after the greedy pass.
pub fn enforce_block_run_limit(
    allocation: &mut BTreeMap<String, BitWidth>,
    results: &[SensitivityResult],
    lowest: BitWidth,
    second_lowest: BitWidth,
    maximum_run: usize,
) -> usize {
    let _ = (allocation, results, lowest, second_lowest, maximum_run);
    todo!("detect low-dominated runs and promote the worst component in every stride-th block")
}
