//! Per-module quantization sensitivity: the one measurement that drives
//! weight bits, key-value cache bits, and (in OptiQ) adapter ranks.
//!
//! Two methods produce the same [`SensitivityTable`]:
//!
//! - [`exact_sweep`]: for every module and every candidate bit width,
//!   simulate-quantize just that module, forward the calibration samples, and
//!   measure the KL divergence of the output logits against an unmodified
//!   reference. Cost is `modules × widths × samples` forward passes. The
//!   forward pass is supplied by a [`SensitivityProbeModel`], which
//!   model-serving implements for Qwen3.5 so this crate never depends on an
//!   engine.
//! - [`static_scorer`]: architecture-only priority with no measurement, the
//!   fallback for bases too large to sweep. OptiQ measured it at the same
//!   downstream accuracy as the exact method on small dense models and 125
//!   times faster.
//!
//! [`checkpoint`] persists a partially complete sweep so a killed run resumes
//! instead of restarting, and [`report`] prints the top-N ranking.

pub mod checkpoint;
pub mod exact_sweep;
pub mod probe;
pub mod report;
pub mod static_scorer;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bit_width::BitWidth;

pub use probe::{CalibrationForwardSample, ModuleRestoreToken, SensitivityProbeModel};

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
        todo!("collect the union of keys across results into a sorted, de-duplicated vector")
    }

    #[must_use]
    pub fn total_logical_parameter_count(&self) -> u64 {
        todo!("sum logical_parameter_count across results")
    }

    #[must_use]
    pub fn result_for(&self, module_name: &str) -> Option<&SensitivityResult> {
        self.results
            .iter()
            .find(|result| result.module_name == module_name)
    }
}
