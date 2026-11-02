//! The sweep output: one KL divergence per module per candidate bit width,
//! and the method that produced it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bit_width::BitWidth;

/// Which method produced a table; recorded in the artifact metadata so a
/// consumer knows whether the bit map was measured or assumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityMethod {
    /// Exact KL sweep against a resident bf16 reference.
    ExactBfloat16Reference,
    /// Exact KL sweep against a uniform 4-bit baseline with bf16 weights
    /// streamed from disk per probe. The signal is "marginal benefit of
    /// raising this module above 4 bits".
    ExactUniform4BitReference,
    /// Rule-based structural priority, no forward passes.
    StaticStructuralRules,
}

/// One module's KL divergence at each candidate bit width.
///
/// The map is keyed by bit width and sorted, which the allocator relies on to
/// walk "the next width up". A value may be NaN when a probe produced
/// non-finite logits; the allocator treats NaN as maximally sensitive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityResult {
    pub module_name: String,
    /// KL(reference ‖ probe) averaged over calibration samples and positions,
    /// per candidate bit width. For the keep-bfloat16 tier this is exactly
    /// zero by construction and no forward is spent.
    pub kl_divergence_by_bit_width: BTreeMap<BitWidth, f64>,
    /// The logical parameter count the allocator weighs the module's cost by.
    pub logical_parameter_count: u64,
}

impl SensitivityResult {
    #[must_use]
    pub fn kl_divergence_at(&self, bit_width: BitWidth) -> Option<f64> {
        self.kl_divergence_by_bit_width.get(&bit_width).copied()
    }

    /// True when any recorded divergence is NaN.
    #[must_use]
    pub fn has_non_finite_measurement(&self) -> bool {
        self.kl_divergence_by_bit_width
            .values()
            .any(|kl_divergence| kl_divergence.is_nan())
    }
}

/// The complete sweep output, one entry per measured module.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityTable {
    pub method: SensitivityMethod,
    pub results: Vec<SensitivityResult>,
}

impl SensitivityTable {
    /// Every bit width that appears in at least one result, ascending. The
    /// allocator infers its candidate ladder from this when the caller does
    /// not pass one.
    #[must_use]
    pub fn observed_bit_widths(&self) -> Vec<BitWidth> {
        // A `BTreeSet` collects the union of every result's width keys while
        // both de-duplicating and keeping them sorted, which is exactly the
        // ascending, de-duplicated ladder the allocator walks for
        // "the next width up".
        self.results
            .iter()
            .flat_map(|result| result.kl_divergence_by_bit_width.keys().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    #[must_use]
    pub fn total_logical_parameter_count(&self) -> u64 {
        self.results
            .iter()
            .map(|result| result.logical_parameter_count)
            .sum()
    }

    #[must_use]
    pub fn result_for(&self, module_name: &str) -> Option<&SensitivityResult> {
        self.results
            .iter()
            .find(|result| result.module_name == module_name)
    }
}
