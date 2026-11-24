//! The post-greedy guard against long stretches of low-bit blocks.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::module_taxonomy::transformer_block_index;
use crate::sensitivity::SensitivityResult;

/// Breaks runs of consecutive low-bit-dominated blocks longer than `maximum_run`.
///
/// Ported from OptiQ's `_enforce_block_run_limit`:
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
    // Group each block's allocated names by transformer block index, in the
    // sweep's result order so a KL tie within a block breaks the same way the
    // Python reference's dict-insertion order does. Modules outside any block
    // are skipped, matching OptiQ's `_block_index` returning `None`.
    let mut by_block: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let sens_map: BTreeMap<String, &SensitivityResult> = results
        .iter()
        .map(|result| (result.module_name.clone(), result))
        .collect();
    for result in results {
        let Some(index) = transformer_block_index(&result.module_name) else {
            continue;
        };
        by_block
            .entry(index)
            .or_default()
            .push(result.module_name.clone());
    }

    // `BTreeMap` keys are already ascending, matching `sorted(by_block)`.
    let block_indices: Vec<usize> = by_block.keys().copied().collect();

    // Per-block low-bit dominance: more than half the components at `lowest`.
    let mut low_dominant: Vec<bool> = Vec::with_capacity(block_indices.len());
    for index in &block_indices {
        let layers = &by_block[index];
        let low = layers
            .iter()
            .filter(|name| allocation.get(*name).copied() == Some(lowest))
            .count();
        low_dominant.push((low as f64) / (layers.len().max(1) as f64) > 0.5);
    }

    // Scan for maximal runs of consecutive low-dominated blocks. A run longer
    // than `maximum_run` is broken by promoting the single most sensitive
    // `lowest`-width component of every `max(1, maximum_run)`-th block in the
    // run's middle (the edges are already covered by boundary protection).
    let mut upgrades = 0usize;
    let mut run_start: Option<usize> = None;
    let last_index = block_indices.len().saturating_sub(1);
    for (i, dominant) in low_dominant.iter().enumerate() {
        if *dominant && run_start.is_none() {
            run_start = Some(i);
        }
        if !*dominant || i == last_index {
            // The run ends before a non-dominant block, or at the final block
            // when it is dominant (`i + 1`).
            let run_end = if !*dominant { i } else { i + 1 };
            if let Some(start) = run_start
                && run_end - start > maximum_run
            {
                let stride = maximum_run.max(1);
                let middle = &block_indices[start..run_end];
                for step in middle.iter().step_by(stride) {
                    let mut candidates: Vec<(f64, &String)> = by_block[step]
                        .iter()
                        .filter(|name| allocation.get(*name).copied() == Some(lowest))
                        .map(|name| {
                            (
                                sens_map
                                    .get(name)
                                    .and_then(|result| {
                                        result.kl_divergence_by_bit_width.get(&lowest)
                                    })
                                    .copied()
                                    .unwrap_or(0.0),
                                name,
                            )
                        })
                        .collect();
                    // Most sensitive first; stable so a KL tie keeps the
                    // block's result order, matching the reference.
                    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
                    if let Some((_, name)) = candidates.first() {
                        allocation.insert((*name).clone(), second_lowest);
                        upgrades += 1;
                    }
                }
            }
            run_start = None;
        }
    }
    upgrades
}
