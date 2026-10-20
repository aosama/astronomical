//! Bits-per-weight arithmetic shared by the allocator and the reports.

use std::collections::BTreeMap;

use crate::bit_width::BitWidth;
use crate::sensitivity::SensitivityResult;

use super::ModuleQuantizationDecision;

/// Average bits per weight over a set of decisions, zero for an empty set.
#[must_use]
pub fn bits_per_weight_of_decisions(decisions: &[ModuleQuantizationDecision]) -> f64 {
    // OptiQ's `compute_bpw`: sum(bits * parameters) / sum(parameters), and 0.0
    // when there are no parameters (an empty allocation has no weight to weigh).
    // Accumulated in `f64` rather than an integer sum so a very large model's
    // `bits * parameters` total cannot overflow, matching Python's exact-int
    // division to well within the parity tolerance.
    let total_params: f64 = decisions
        .iter()
        .map(|decision| decision.logical_parameter_count as f64)
        .sum();
    if total_params == 0.0 {
        return 0.0;
    }
    let total_bits: f64 = decisions
        .iter()
        .map(|decision| {
            (decision.bit_width.bits() as f64) * (decision.logical_parameter_count as f64)
        })
        .sum();
    total_bits / total_params
}

/// Average bits per weight of an in-progress allocation keyed by module
/// name, weighted by each result's logical parameter count.
#[must_use]
pub fn bits_per_weight_of_allocation(
    allocation: &BTreeMap<String, BitWidth>,
    results: &[SensitivityResult],
) -> f64 {
    // OptiQ's `_compute_allocation_bpw`, recomputed on every greedy iteration:
    // the running weight of the current `allocation` dict. `max(total, 1)`
    // keeps an empty table at 0.0 instead of dividing by zero.
    let total_params: f64 = results
        .iter()
        .map(|result| result.logical_parameter_count as f64)
        .sum();
    let total_bits: f64 = results
        .iter()
        .map(|result| {
            let bits = allocation
                .get(&result.module_name)
                .map(|width| width.bits())
                .unwrap_or(0) as f64;
            bits * (result.logical_parameter_count as f64)
        })
        .sum();
    total_bits / total_params.max(1.0)
}

/// Bytes the packed weights occupy at the decided widths, ignoring scales
/// and biases (OptiQ's `estimated_size_mb` input).
#[must_use]
pub fn estimated_weight_bytes(decisions: &[ModuleQuantizationDecision]) -> u64 {
    // OptiQ's `est_size = sum(bits * param_count) / 8`. Accumulated in `u128`
    // so a large model's running total cannot overflow before the `/ 8`, then
    // narrowed to the `u64` the artifact metadata records.
    let total_bits: u128 = decisions
        .iter()
        .map(|decision| {
            (decision.bit_width.bits() as u128) * (decision.logical_parameter_count as u128)
        })
        .sum();
    (total_bits / 8) as u64
}
