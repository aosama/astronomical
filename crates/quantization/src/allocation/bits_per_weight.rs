//! Bits-per-weight arithmetic shared by the allocator and the reports.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::sensitivity::SensitivityResult;

use super::ModuleQuantizationDecision;

/// Average bits per weight over a set of decisions, zero for an empty set.
#[must_use]
pub fn bits_per_weight_of_decisions(decisions: &[ModuleQuantizationDecision]) -> f64 {
    let _ = decisions;
    todo!("sum(bits * parameters) / sum(parameters), zero when there are no parameters")
}

/// Average bits per weight of an in-progress allocation keyed by module
/// name, weighted by each result's logical parameter count.
#[must_use]
pub fn bits_per_weight_of_allocation(
    allocation: &BTreeMap<String, BitWidth>,
    results: &[SensitivityResult],
) -> f64 {
    let _ = (allocation, results);
    todo!("sum(allocation[name] * parameters) / max(sum(parameters), 1)")
}

/// Bytes the packed weights occupy at the decided widths, ignoring scales
/// and biases (OptiQ's `estimated_size_mb` input).
#[must_use]
pub fn estimated_weight_bytes(decisions: &[ModuleQuantizationDecision]) -> u64 {
    let _ = decisions;
    todo!("sum(bits * parameters) / 8")
}
