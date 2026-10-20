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
        let _ = other;
        todo!("total_cmp on efficiency, then reversed name order so the smaller name is the max")
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
    let _ = (result, current, next);
    todo!("apply the sign-detection rule documented above")
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
    let _ = (heap, effective_result, current, candidate_bit_widths);
    todo!(
        "compute next_above, the reduction, the bit cost, and push when the reduction is positive"
    )
}
