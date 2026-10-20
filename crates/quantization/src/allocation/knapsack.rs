//! The greedy upgrade heap.
//!
//! Each heap entry is one candidate step: module, current width, next width,
//! and the KL reduction per bit spent. The heap is a max-heap on efficiency.
//! Ties are broken by module name so the allocation is deterministic across
//! runs and platforms, which the Python tool's tuple ordering happens to
//! provide and the parity fixtures depend on.

use std::collections::BinaryHeap;

use crate::bit_width::{BitWidth, CandidateBitWidths};
use crate::sensitivity::SensitivityResult;

/// One candidate upgrade step.
#[derive(Clone, Debug, PartialEq)]
pub struct UpgradeCandidate {
    pub module_name: String,
    pub current_bit_width: BitWidth,
    pub next_bit_width: BitWidth,
    pub kl_reduction: f64,
    /// `kl_reduction / max(bit_cost, 1)` where `bit_cost = (next - current) *
    /// parameters`.
    pub efficiency: f64,
}

impl Eq for UpgradeCandidate {}

impl PartialOrd for UpgradeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UpgradeCandidate {
    /// Higher efficiency first; equal efficiency orders by module name
    /// ascending so `BinaryHeap` (a max-heap) pops the lexically smaller
    /// name first, matching Python's `heapq` on `(-efficiency, name, ...)`.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `total_cmp` gives a total order over efficiency (including any NaN
        // efficiency, ordered as the largest). On an exact tie the module
        // name decides, reversed: the smaller name becomes the `Greater`
        // entry, so the max-heap pops it first — the same deterministic
        // tie-break Python's tuple ordering provides and the parity fixtures
        // depend on.
        self.efficiency
            .total_cmp(&other.efficiency)
            .then_with(|| other.module_name.cmp(&self.module_name))
    }
}

/// The quality benefit of stepping a module from `current` to `next`.
///
/// Two reference-mode semantics, autodetected (OptiQ's `_kl_reduction`):
/// - bf16 reference: `kl[b]` is KL(bf16 ‖ b-bit); upgrading reduces it, so
///   the benefit is `kl[current] - kl[next]`.
/// - uniform-N reference: `kl[lowest]` is zero (probing N bits against an
///   N-bit reference is a no-op, observed as float noise near 1e-18) and
///   `kl[higher]` measures how far the upgrade moves the module toward
///   bf16, which is the benefit directly. Without this the heap stays empty
///   and every module is left at the minimum.
///
/// The sign of `kl[next] - kl[current]` flips between the two modes, so
/// `next > current && kl[next] > kl[current]` selects the uniform-reference
/// rule. A missing width reads as zero.
#[must_use]
pub fn kl_reduction(result: &SensitivityResult, current: BitWidth, next: BitWidth) -> f64 {
    let kl_current = result.kl_divergence_at(current).unwrap_or(0.0);
    let kl_next = result.kl_divergence_at(next).unwrap_or(0.0);
    // Under a uniform-N reference the KL at the matching bit-width is ~0 and
    // the KL at a higher width is the benefit directly, so `kl_next > kl_current`
    // while `next > current`. That sign flip is the autodetect; every other
    // case is the bf16 reference where upgrading reduces KL.
    if next.bits() > current.bits() && kl_next > kl_current {
        kl_next - kl_current
    } else {
        kl_current - kl_next
    }
}

/// Pushes the module's next step, if any, onto the heap. No push when the
/// module is at the top of the ladder, off the ladder, or when the step has
/// no benefit (`kl_reduction <= 0`).
pub fn push_next_upgrade(
    heap: &mut BinaryHeap<UpgradeCandidate>,
    effective_result: &SensitivityResult,
    current: BitWidth,
    candidate_bit_widths: &CandidateBitWidths,
) {
    // `next_above` is `None` at the top of the ladder and when `current` is
    // not on it, which together cover "already at max" and "off the grid".
    let Some(next_bit_width) = candidate_bit_widths.next_above(current) else {
        return;
    };
    let reduction = kl_reduction(effective_result, current, next_bit_width);
    if reduction <= 0.0 {
        return;
    }
    // `bit_cost = (next - current) * parameters`; efficiency is the KL benefit
    // per added bit-weight. `max(bit_cost, 1)` mirrors OptiQ and guards the
    // (pathological) zero-parameter case against division by zero.
    let bit_cost = ((next_bit_width.bits() - current.bits()) as f64)
        * (effective_result.logical_parameter_count as f64);
    let efficiency = reduction / bit_cost.max(1.0);
    heap.push(UpgradeCandidate {
        module_name: effective_result.module_name.clone(),
        current_bit_width: current,
        next_bit_width,
        kl_reduction: reduction,
        efficiency,
    });
}
