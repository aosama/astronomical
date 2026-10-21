//! Rule-based structural sensitivity: no measurement at all.
//!
//! Each module is scored by architecture alone. Embedding and output head and
//! the first and last transformer block rank highest, attention and the
//! mixture-of-experts router above the dense feed-forward, routed experts
//! lowest, and the network's edges above its middle. The scores are packed as
//! a [`SensitivityTable`] so the same allocator spends the bit budget on the
//! highest-priority modules at the requested candidate widths and target.

use crate::bit_width::CandidateBitWidths;
use crate::module_inventory::ModuleInventory;

use super::{SensitivityMethod, SensitivityTable};

#[cfg(test)]
#[path = "static_scorer_tests.rs"]
mod static_scorer_tests;

/// Architecture-only priority in `(0, 1]` for one module.
///
/// Work to port (OptiQ's `_structural_priority`), on the lower-cased name:
/// - `embed` or `lm_head` anywhere → 1.0, done.
/// - Baseline 0.30. With a block index `d` out of `n` blocks: 0.90 for the
///   first or last block, 0.65 for the next-to-edge blocks, otherwise a taper
///   `0.30 + 0.15 * (1 - edge)` where `edge = min(d, n-1-d) / max(n/2, 1)`.
/// - `+0.30` for `self_attn` or `.attn`; `+0.40` for `router`, `gate.weight`,
///   or `.gate.`; `-0.10` for `experts` or `switch`.
/// - Clamp to `[0.05, 1.0]`.
///
/// Note the router rule is substring-based and deliberately looser than the
/// taxonomy's router predicate; keep OptiQ's rule so the static method stays
/// comparable with published static-mixed artifacts.
#[must_use]
pub fn structural_priority(module_name: &str, transformer_block_count: usize) -> f64 {
    let _ = (module_name, transformer_block_count);
    todo!("apply the priority rules documented above")
}

/// Scores every module in the inventory.
///
/// Work to port: for each module, encode the priority as a sensitivity curve
/// the allocator reads: KL falls linearly from `priority` at the lowest
/// candidate width to 0 at the highest, that is
/// `priority * (highest - b) / (highest - lowest)` for each candidate `b`,
/// so upgrading a high-priority module buys the most "KL". The parameter
/// count is the inventory's logical count.
#[must_use]
pub fn score_static_structural(
    inventory: &ModuleInventory,
    candidate_bit_widths: &CandidateBitWidths,
) -> SensitivityTable {
    let _ = (inventory, candidate_bit_widths);
    let _ = SensitivityMethod::StaticStructuralRules;
    todo!("build one SensitivityResult per inventory module from structural_priority")
}
