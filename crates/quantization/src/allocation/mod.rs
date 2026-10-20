//! Mixed-precision bit allocation: turn a sensitivity table into one bit
//! width per module under a bits-per-weight target.
//!
//! The allocator is a greedy knapsack with two block-aware constraints that
//! OptiQ added after field failures:
//!
//! 1. Start every module at the lowest candidate width.
//! 2. Block-aware floor ([`block_floor`]): pre-upgrade the most sensitive
//!    components of every transformer block to the second-lowest width,
//!    reserving the first slots for attention. Without this the feed-forward
//!    projections win every per-block ranking and whole attention stacks stay
//!    at the lowest width, which collapsed instruction following on late
//!    layers of Qwen3.5-4B.
//! 3. Greedy upgrades ([`knapsack`]): repeatedly promote the module whose next
//!    step up buys the most KL reduction per bit spent, until the running
//!    bits-per-weight meets the target. The target is a floor, not a cap:
//!    slight overshoot from the discrete grid is accepted because the user
//!    asked for a quality floor.
//! 4. Block-run guard ([`block_run_guard`]): after the greedy pass, break any
//!    run of more than `maximum_low_bit_block_run` consecutive blocks whose
//!    majority of components still sits at the lowest width.
//!
//! Protected modules ([`protected_modules`]) take part in the greedy pass like
//! every other module but with their KL multiplied by 100, so they are
//! upgraded first rather than exempted from the budget.
//!
//! The allocator is pure: no MLX, no I/O, deterministic for a given table.
//! That makes it the natural place to introduce Astronomical's own policy
//! modifications, and the parity fixtures under `tests/fixtures/` pin the
//! current behavior to the Python reference so a modification is a visible,
//! intended diff rather than drift.

pub mod bits_per_weight;
pub mod block_floor;
pub mod block_run_guard;
pub mod knapsack;
pub mod predicate;
pub mod protected_modules;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::sensitivity::SensitivityTable;

pub use predicate::QuantizationPredicate;

#[cfg(test)]
#[path = "allocation_tests.rs"]
mod allocation_tests;

/// Tunables that shape the allocation beyond the target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AllocationConstraints {
    /// Average bits per weight the allocation must meet or exceed.
    pub target_bits_per_weight: f64,
    /// The ladder; inferred from the table's observed widths when `None`.
    pub candidate_bit_widths: Option<CandidateBitWidths>,
    pub group_size: GroupSize,
    /// Protect the first and last `boundary_block_count` transformer blocks.
    pub protect_boundary_blocks: bool,
    pub boundary_block_count: usize,
    /// Components per block held at or above the second-lowest width; zero
    /// disables the floor. Full-attention blocks honor the value, blocks with
    /// only two linears (state-space mixers, plain MLPs) are capped at one so
    /// the floor cannot promote an entire block and make the target
    /// unreachable.
    pub floor_components_per_block: usize,
    /// Longest tolerated run of consecutive low-bit-dominated blocks; zero
    /// disables the guard.
    pub maximum_low_bit_block_run: usize,
}

impl AllocationConstraints {
    /// OptiQ's defaults: 2 floor components, run limit 3, protect one block
    /// at each end, group size 64.
    #[must_use]
    pub fn defaults_for_target(target_bits_per_weight: f64) -> Self {
        Self {
            target_bits_per_weight,
            candidate_bit_widths: None,
            group_size: GroupSize::new(64).expect("64 is an MLX affine group size"),
            protect_boundary_blocks: true,
            boundary_block_count: 1,
            floor_components_per_block: 2,
            maximum_low_bit_block_run: 3,
        }
    }
}

/// One module's final decision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModuleQuantizationDecision {
    pub module_name: String,
    pub bit_width: BitWidth,
    pub group_size: GroupSize,
    pub logical_parameter_count: u64,
}

/// The allocation plus the bookkeeping the artifact metadata records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AllocationResult {
    pub decisions: Vec<ModuleQuantizationDecision>,
    pub target_bits_per_weight: f64,
    pub achieved_bits_per_weight: f64,
    /// Modules at the ladder's highest width.
    pub module_count_at_highest_width: usize,
    /// Modules at the ladder's lowest width.
    pub module_count_at_lowest_width: usize,
    pub total_logical_parameter_count: u64,
    pub estimated_weight_bytes: u64,
    /// How many modules the block-aware floor promoted before the greedy pass.
    pub floor_upgrade_count: usize,
    /// How many modules the run guard promoted after the greedy pass.
    pub run_guard_upgrade_count: usize,
    /// Modules whose measurements contained NaN and were treated as
    /// maximally sensitive.
    pub non_finite_module_names: Vec<String>,
}

impl AllocationResult {
    /// The floor and the run guard take precedence over the budget, so a low
    /// target can be exceeded: asking for 4.0 on a seven-component block
    /// lands near 5.3. Silently shipping an artifact a third larger than
    /// requested is how someone picks a quant to fit a machine it then does
    /// not fit, so the overshoot is surfaced for the caller to print.
    #[must_use]
    pub fn overshoot_note(&self, floor_components_per_block: usize) -> Option<String> {
        let _ = floor_components_per_block;
        todo!(
            "when achieved exceeds target by more than 0.25 bits per weight, render OptiQ's NOTE text with the percentage and the floor hint"
        )
    }

    /// Modules grouped by assigned width, for the summary line.
    #[must_use]
    pub fn module_count_by_bit_width(&self) -> BTreeMap<BitWidth, usize> {
        todo!("count decisions per bit width")
    }
}

/// Runs the full allocation.
///
/// Work to port (OptiQ's `optimize_mixed_precision`), in this order:
/// 1. Empty table → empty result with zeroed figures.
/// 2. Resolve the ladder (constraints, else `table.observed_bit_widths()`).
/// 3. Build the protected set from the module names (see
///    [`protected_modules`]).
/// 4. Initialize every module at the lowest width.
/// 5. Apply the block-aware floor; count its upgrades.
/// 6. Replace NaN measurements: for a module with any NaN, substitute
///    `max_finite * 10 * (highest / max(b, 1))` at each NaN width so lower
///    widths look far worse and the greedy pass upgrades the module; record
///    its name.
/// 7. Build the effective table: protected modules get every KL multiplied
///    by 100.
/// 8. Seed the heap with each module's next upgrade, then pop until the
///    running bits-per-weight reaches the target, re-pushing the upgraded
///    module's next step. Stale heap entries (module already moved) are
///    skipped.
/// 9. Apply the block-run guard; count its upgrades.
/// 10. Build the decisions, achieved bits-per-weight, counts, and the
///     estimated size `sum(bits * parameters) / 8`.
pub fn optimize_mixed_precision(
    table: &SensitivityTable,
    constraints: &AllocationConstraints,
) -> Result<AllocationResult, QuantizationError> {
    let _ = (table, constraints);
    todo!("compose the floor, the knapsack, and the run guard as documented above")
}
