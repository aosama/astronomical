//! The block-aware floor applied before the greedy pass.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::module_taxonomy::{classify_module, is_attention_component, transformer_block_index};
use crate::sensitivity::SensitivityResult;

/// Promotes per-block components from `lowest` to `second_lowest`.
///
/// Ported from OptiQ's `_apply_block_aware_floor`:
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
    // Group each block's components by its transformer block index, recording
    // the KL at the lowest width so the block can be ranked most-sensitive
    // first. Modules outside any `layers.N` block are ignored, exactly as
    // OptiQ's `_block_index` returns `None` for them. A `BTreeMap` iterates
    // blocks in index order; block processing is independent (each block only
    // touches its own names), so the final allocation is order-free.
    let mut block_layers: BTreeMap<usize, Vec<(f64, String)>> = BTreeMap::new();
    for result in results {
        let Some(index) = transformer_block_index(&result.module_name) else {
            continue;
        };
        let kl_low = result
            .kl_divergence_by_bit_width
            .get(&lowest)
            .copied()
            .unwrap_or(0.0);
        block_layers
            .entry(index)
            .or_default()
            .push((kl_low, result.module_name.clone()));
    }

    let mut upgrades = 0usize;
    for layers in block_layers.values_mut() {
        if layers.is_empty() {
            continue;
        }
        // Most sensitive first. `total_cmp` orders a NaN KL (a probe that
        // produced non-finite logits) as the largest, so a NaN component ranks
        // first and would be picked by the fill stage before any finite one;
        // that is a deliberate, deterministic choice where Python's NaN sort
        // order is undefined. The parity fixture's NaN module sits in a
        // full-attention block whose floor budget is fully consumed by the
        // reserved attention slots, so the fill stage never reaches it. The
        // sort is stable, so equal finite KL keeps the input order.
        layers.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Attention components of this block, in the ranked order above.
        let attn: Vec<&String> = layers
            .iter()
            .filter(|(_, name)| is_attention_component(classify_module(name)))
            .map(|(_, name)| name)
            .collect();
        let attn_count = attn.len();

        // Reserve attention slots first: a full-attention block (four or more
        // attention components present) reserves two, a partial-attention block
        // one, capped at the floor budget. This is the load-bearing bit that
        // prevents late full-attention layers from collapsing to the lowest
        // width.
        let attn_reserve = if attn_count >= 4 {
            2
        } else {
            usize::from(attn_count > 0)
        };
        let attn_reserve = attn_reserve.min(floor_components);

        // Per-block budget. Full-attention blocks honor the global floor; any
        // other block is capped at one, because a two-linear block (state-space
        // mixer `in_proj`/`out_proj`, MLP `up_proj`/`down_proj`) would otherwise
        // have both promoted and the target becomes unreachable.
        let block_floor = if attn_count >= 4 {
            floor_components
        } else {
            floor_components.min(1)
        };

        // Reserved attention picks first, then fill the remaining budget from
        // the overall ranking, skipping anything already picked.
        let mut picks: Vec<String> = Vec::with_capacity(block_floor);
        for name in attn.iter().take(attn_reserve) {
            picks.push((*name).clone());
        }
        for (_, name) in layers.iter() {
            if picks.len() >= block_floor {
                break;
            }
            if picks.iter().any(|picked| picked == name) {
                continue;
            }
            picks.push(name.clone());
        }

        for name in picks {
            let current = allocation.get(&name).copied().unwrap_or(lowest);
            if current < second_lowest {
                allocation.insert(name, second_lowest);
                upgrades += 1;
            }
        }
    }
    upgrades
}
