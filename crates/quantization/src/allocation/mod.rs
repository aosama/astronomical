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

use std::collections::{BTreeMap, BinaryHeap};

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::sensitivity::{SensitivityResult, SensitivityTable};

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
        // Only an overshoot past 0.25 bpw warrants a note; a smaller one is the
        // discrete grid doing its job and does not warn.
        if self.achieved_bits_per_weight <= self.target_bits_per_weight + 0.25 {
            return None;
        }
        let achieved = self.achieved_bits_per_weight;
        let target = self.target_bits_per_weight;
        let over = (achieved - target) / target * 100.0;
        Some(format!(
            "  NOTE: achieved {0:.2} bpw against a target of {1:.2} ({2:+.0}%). The block-aware floor and run guard take precedence over the budget so no stretch of blocks is left wholly at the lowest width. Lower --n-floor-per-block (currently {3}) to trade that protection for size.",
            achieved, target, over, floor_components_per_block
        ))
    }

    /// Modules grouped by assigned width, for the summary line.
    #[must_use]
    pub fn module_count_by_bit_width(&self) -> BTreeMap<BitWidth, usize> {
        let mut counts: BTreeMap<BitWidth, usize> = BTreeMap::new();
        for decision in &self.decisions {
            *counts.entry(decision.bit_width).or_insert(0) += 1;
        }
        counts
    }
}

/// Runs the full allocation, ported from OptiQ's `optimize_mixed_precision`.
///
/// The sequence (see the module doc for the rationale of each stage):
/// 1. Empty table → empty result with zeroed figures.
/// 2. Resolve the ladder (constraints, else `table.observed_bit_widths()`).
/// 3. Build the protected set from the module names (see [`protected_modules`]).
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
    // 1. Empty table → empty result with zeroed figures.
    if table.results.is_empty() {
        return Ok(AllocationResult {
            decisions: Vec::new(),
            target_bits_per_weight: constraints.target_bits_per_weight,
            achieved_bits_per_weight: 0.0,
            module_count_at_highest_width: 0,
            module_count_at_lowest_width: 0,
            total_logical_parameter_count: 0,
            estimated_weight_bytes: 0,
            floor_upgrade_count: 0,
            run_guard_upgrade_count: 0,
            non_finite_module_names: Vec::new(),
        });
    }

    // 2. Resolve the ladder: the caller's, else the union of the table's
    //    observed widths (ascending, de-duplicated). A real sweep always yields
    //    at least one width, so this inference is non-empty; a table whose every
    //    result has no measurements is degenerate and errors here rather than
    //    falling back to OptiQ's `[low_bits, high_bits]` = `[4, 8]` shim,
    //    because an empty sweep carries no information to quantize from.
    let candidate_bit_widths = match &constraints.candidate_bit_widths {
        Some(ladder) => ladder.clone(),
        None => {
            let widths: Vec<u32> = table
                .observed_bit_widths()
                .iter()
                .map(|width| width.bits())
                .collect();
            CandidateBitWidths::new(&widths)?
        }
    };
    let lowest = candidate_bit_widths.lowest();
    let highest = candidate_bit_widths.highest();
    let second_lowest = candidate_bit_widths.second_lowest();

    // 3. The protected set: embeddings/heads/outside-blocks, MoE routers and
    //    shared experts, plus the boundary blocks when enabled.
    let module_names: Vec<&str> = table
        .results
        .iter()
        .map(|result| result.module_name.as_str())
        .collect();
    let protected = protected_modules::protected_module_names(
        &module_names,
        constraints.protect_boundary_blocks,
        constraints.boundary_block_count,
    );

    // 4. Seed every module at the lowest width.
    let mut allocation: BTreeMap<String, BitWidth> = table
        .results
        .iter()
        .map(|result| (result.module_name.clone(), lowest))
        .collect();

    // 5. Block-aware floor, before any NaN cleaning, so it ranks on the raw
    //    measurements. It only runs with a floor budget and a multi-width
    //    ladder (a single-width ladder has no `second_lowest` to promote to).
    let mut floor_upgrade_count = 0usize;
    if constraints.floor_components_per_block > 0
        && let Some(second) = second_lowest
    {
        floor_upgrade_count = block_floor::apply_block_aware_floor(
            &mut allocation,
            &table.results,
            lowest,
            second,
            constraints.floor_components_per_block,
        );
    }

    // 6. Replace NaN measurements with a large finite value so the greedy pass
    //    sees them as maximally sensitive; record the affected names.
    let (results, non_finite_module_names) = clean_non_finite(&table.results, highest.bits());

    // 7. Effective measurements: protected modules get their KL multiplied so
    //    the greedy heap upgrades them first, without exempting them.
    let effective_results: BTreeMap<String, SensitivityResult> = results
        .iter()
        .map(|result| {
            if protected.contains(&result.module_name) {
                let kl_divergence_by_bit_width = result
                    .kl_divergence_by_bit_width
                    .iter()
                    .map(|(&width, &value)| {
                        (
                            width,
                            value * protected_modules::PROTECTED_MODULE_KL_MULTIPLIER,
                        )
                    })
                    .collect();
                (
                    result.module_name.clone(),
                    SensitivityResult {
                        module_name: result.module_name.clone(),
                        kl_divergence_by_bit_width,
                        logical_parameter_count: result.logical_parameter_count,
                    },
                )
            } else {
                (result.module_name.clone(), result.clone())
            }
        })
        .collect();

    // 8. Seed the heap with each module's next upgrade.
    let mut heap: BinaryHeap<knapsack::UpgradeCandidate> = BinaryHeap::new();
    for result in &results {
        let current = *allocation
            .get(&result.module_name)
            .expect("every module is seeded at the lowest width");
        if let Some(effective) = effective_results.get(&result.module_name) {
            knapsack::push_next_upgrade(&mut heap, effective, current, &candidate_bit_widths);
        }
    }

    // Greedy: keep upgrading the most efficient module until the running
    // bits-per-weight meets the target. The target is a floor, not a cap, so
    // the discrete grid's slight overshoot is accepted.
    while !heap.is_empty()
        && bits_per_weight::bits_per_weight_of_allocation(&allocation, &results)
            < constraints.target_bits_per_weight
    {
        let candidate = heap
            .pop()
            .expect("the heap is non-empty whenever the loop condition holds");
        // A stale entry means the module already moved past `current`; skip it.
        let Some(&current) = allocation.get(&candidate.module_name) else {
            continue;
        };
        if current != candidate.current_bit_width {
            continue;
        }
        allocation.insert(candidate.module_name.clone(), candidate.next_bit_width);
        if let Some(effective) = effective_results.get(&candidate.module_name) {
            knapsack::push_next_upgrade(
                &mut heap,
                effective,
                candidate.next_bit_width,
                &candidate_bit_widths,
            );
        }
    }

    // 9. Block-run guard, the post-greedy backstop. Same single-width-ladder
    //    guard as the floor: it needs a `second_lowest` to promote to.
    let mut run_guard_upgrade_count = 0usize;
    if constraints.maximum_low_bit_block_run > 0
        && let Some(second) = second_lowest
    {
        run_guard_upgrade_count = block_run_guard::enforce_block_run_limit(
            &mut allocation,
            &results,
            lowest,
            second,
            constraints.maximum_low_bit_block_run,
        );
    }

    // 10. Build the decisions and the summary figures, in the sweep's order.
    let group_size = constraints.group_size;
    let decisions: Vec<ModuleQuantizationDecision> = results
        .iter()
        .map(|result| ModuleQuantizationDecision {
            module_name: result.module_name.clone(),
            bit_width: *allocation
                .get(&result.module_name)
                .expect("every module has a final width"),
            group_size,
            logical_parameter_count: result.logical_parameter_count,
        })
        .collect();

    let achieved_bits_per_weight = bits_per_weight::bits_per_weight_of_decisions(&decisions);
    let module_count_at_highest_width = decisions
        .iter()
        .filter(|decision| decision.bit_width == highest)
        .count();
    let module_count_at_lowest_width = decisions
        .iter()
        .filter(|decision| decision.bit_width == lowest)
        .count();
    let total_logical_parameter_count: u64 = results
        .iter()
        .map(|result| result.logical_parameter_count)
        .sum();
    let estimated_weight_bytes = bits_per_weight::estimated_weight_bytes(&decisions);

    Ok(AllocationResult {
        decisions,
        target_bits_per_weight: constraints.target_bits_per_weight,
        achieved_bits_per_weight,
        module_count_at_highest_width,
        module_count_at_lowest_width,
        total_logical_parameter_count,
        estimated_weight_bytes,
        floor_upgrade_count,
        run_guard_upgrade_count,
        non_finite_module_names,
    })
}

/// Replaces NaN KL measurements with a large finite value so the greedy pass
/// treats the module as maximally sensitive (lower widths look far worse),
/// returning the cleaned results and the names of the modules that had a NaN.
/// OptiQ's step 6.
fn clean_non_finite(
    results: &[SensitivityResult],
    max_bits: u32,
) -> (Vec<SensitivityResult>, Vec<String>) {
    let mut cleaned: Vec<SensitivityResult> = Vec::with_capacity(results.len());
    let mut non_finite_module_names: Vec<String> = Vec::new();
    for result in results {
        if !result.has_non_finite_measurement() {
            cleaned.push(result.clone());
            continue;
        }
        non_finite_module_names.push(result.module_name.clone());
        // The largest of this module's finite measurements, or 1e6 when it has
        // none (OptiQ's `default=1e6`).
        let max_finite = result
            .kl_divergence_by_bit_width
            .values()
            .copied()
            .filter(|value| !value.is_nan())
            .fold(None, |highest: Option<f64>, value| {
                Some(highest.map_or(value, |current| current.max(value)))
            })
            .unwrap_or(1e6);
        let kl_divergence_by_bit_width = result
            .kl_divergence_by_bit_width
            .iter()
            .map(|(&width, &value)| {
                if value.is_nan() {
                    // Lower widths get a higher substituted KL. `max(b, 1)`
                    // guards the divisor; every valid width is at least 2.
                    let substituted =
                        max_finite * 10.0 * (max_bits as f64 / (width.bits() as f64).max(1.0));
                    (width, substituted)
                } else {
                    (width, value)
                }
            })
            .collect();
        cleaned.push(SensitivityResult {
            module_name: result.module_name.clone(),
            kl_divergence_by_bit_width,
            logical_parameter_count: result.logical_parameter_count,
        });
    }
    (cleaned, non_finite_module_names)
}
