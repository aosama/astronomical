//! Rule-based structural sensitivity: no measurement at all.
//!
//! Each module is scored by architecture alone. Embedding and output head and
//! the first and last transformer block rank highest, attention and the
//! mixture-of-experts router above the dense feed-forward, routed experts
//! lowest, and the network's edges above its middle. The scores are packed as
//! a [`SensitivityTable`] so the same allocator spends the bit budget on the
//! highest-priority modules at the requested candidate widths and target.

use std::collections::BTreeMap;

use crate::bit_width::{BitWidth, CandidateBitWidths};
use crate::module_inventory::ModuleInventory;

use super::{SensitivityMethod, SensitivityResult, SensitivityTable};

#[cfg(test)]
#[path = "static_scorer_tests.rs"]
mod static_scorer_tests;

/// Architecture-only priority in `[0.05, 1.0]` for one module, OptiQ's
/// `_structural_priority`, on the lower-cased name:
/// - `embed` or `lm_head` anywhere → 1.0, done.
/// - Baseline 0.30. With a block index `d` out of `n` blocks: 0.90 for the
///   first or last block, 0.65 for the next-to-edge blocks, otherwise a taper
///   `0.30 + 0.15 * (1 - edge)` where `edge = min(d, n-1-d) / max(n/2, 1)`.
/// - `+0.30` for `self_attn` or `.attn`; `+0.40` for `router`, `gate.weight`,
///   or `.gate.`; `-0.10` for `experts` or `switch`.
/// - Clamp to `[0.05, 1.0]`.
///
/// The router rule is substring-based and deliberately looser than the
/// taxonomy's router predicate, and the block index uses OptiQ's own
/// `layers?\.(\d+)\.` pattern rather than the taxonomy's: both are kept so
/// the static method stays comparable with published static-mixed artifacts.
#[must_use]
pub fn structural_priority(module_name: &str, transformer_block_count: usize) -> f64 {
    let lower_case_name = module_name.to_lowercase();
    if lower_case_name.contains("embed") || lower_case_name.contains("lm_head") {
        return 1.0;
    }
    let mut priority = position_priority(&lower_case_name, transformer_block_count);
    if lower_case_name.contains("self_attn") || lower_case_name.contains(".attn") {
        priority += 0.30;
    }
    if lower_case_name.contains("router")
        || lower_case_name.contains("gate.weight")
        || lower_case_name.contains(".gate.")
    {
        priority += 0.40;
    }
    if lower_case_name.contains("experts") || lower_case_name.contains("switch") {
        priority -= 0.10;
    }
    priority.clamp(MINIMUM_STRUCTURAL_PRIORITY, 1.0)
}

/// Scores every module in the inventory.
///
/// The priority is encoded as a sensitivity curve the allocator reads: KL
/// falls linearly from `priority` at the lowest candidate width to 0 at the
/// highest, `priority * (highest - b) / max(highest - lowest, 1)` for each
/// candidate `b`, so upgrading a high-priority module buys the most "KL".
/// The block count comes from the same name pattern the priority uses.
#[must_use]
pub fn score_static_structural(
    inventory: &ModuleInventory,
    candidate_bit_widths: &CandidateBitWidths,
) -> SensitivityTable {
    let mut block_count = 1_usize;
    for descriptor in inventory.modules() {
        if let Some(block_index) = optiq_block_index(&descriptor.module_name.to_lowercase()) {
            block_count = block_count.max(block_index + 1);
        }
    }
    let mut results = Vec::with_capacity(inventory.module_count());
    for descriptor in inventory.modules() {
        let priority = structural_priority(&descriptor.module_name, block_count);
        results.push(SensitivityResult {
            module_name: descriptor.module_name.clone(),
            kl_divergence_by_bit_width: priority_curve(priority, candidate_bit_widths),
            logical_parameter_count: descriptor.logical_parameter_count,
        });
    }
    SensitivityTable {
        method: SensitivityMethod::StaticStructuralRules,
        results,
    }
}

/// The floor every structural priority is clamped to, so no module reads as
/// free to quantize.
const MINIMUM_STRUCTURAL_PRIORITY: f64 = 0.05;

/// The baseline from the module's position in the network: edges above the
/// middle, with a taper across the middle blocks.
fn position_priority(lower_case_name: &str, transformer_block_count: usize) -> f64 {
    let Some(block_index) = optiq_block_index(lower_case_name) else {
        return 0.30;
    };
    if transformer_block_count <= 1 {
        return 0.30;
    }
    let last_block_index = transformer_block_count - 1;
    if block_index == 0 || block_index == last_block_index {
        return 0.90;
    }
    if block_index <= 1 || block_index + 2 >= transformer_block_count {
        return 0.65;
    }
    let distance_to_edge = block_index.min(last_block_index - block_index) as f64;
    let edge = distance_to_edge / (transformer_block_count as f64 / 2.0).max(1.0);
    0.30 + 0.15 * (1.0 - edge)
}

/// OptiQ's `layers?\.(\d+)\.`: the first segment ending in `layer` or
/// `layers` followed by an all-digit segment that is not the last one.
fn optiq_block_index(lower_case_name: &str) -> Option<usize> {
    let segments: Vec<&str> = lower_case_name.split('.').collect();
    for segment_index in 0..segments.len().saturating_sub(2) {
        let is_layer_segment = segments[segment_index].ends_with("layer")
            || segments[segment_index].ends_with("layers");
        let digit_segment = segments[segment_index + 1];
        let is_digit_segment = is_all_ascii_digits(digit_segment);
        if is_layer_segment && is_digit_segment {
            return digit_segment.parse().ok();
        }
    }
    None
}

/// True for a non-empty segment of ASCII digits only, the regex's `\\d+`.
fn is_all_ascii_digits(segment: &str) -> bool {
    if segment.is_empty() {
        return false;
    }
    for character in segment.chars() {
        if !character.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// The linear priority-to-KL curve across the candidate ladder.
fn priority_curve(
    priority: f64,
    candidate_bit_widths: &CandidateBitWidths,
) -> BTreeMap<BitWidth, f64> {
    let lowest_bits = f64::from(candidate_bit_widths.lowest().bits());
    let highest_bits = f64::from(candidate_bit_widths.highest().bits());
    let span = (highest_bits - lowest_bits).max(1.0);
    let mut kl_divergence_by_bit_width = BTreeMap::new();
    for bit_width in candidate_bit_widths.as_slice() {
        let kl_divergence = priority * (highest_bits - f64::from(bit_width.bits())) / span;
        kl_divergence_by_bit_width.insert(*bit_width, kl_divergence);
    }
    kl_divergence_by_bit_width
}
