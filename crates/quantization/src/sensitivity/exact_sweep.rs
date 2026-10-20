//! The exact per-module KL sweep driver.
//!
//! The driver is engine-agnostic: it sequences probes through a
//! [`SensitivityProbeModel`], keeps the checkpoint monotonic, reports
//! progress with an ETA, and enforces the measured-majority guard. It does
//! not know whether the reference is bf16 or a uniform baseline; the probe
//! model's `current_bit_width` answers tell it which widths are free.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bit_width::{CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::progress::ProgressReporter;

use super::probe::{CalibrationForwardSample, SensitivityProbeModel};
use super::{SensitivityMethod, SensitivityResult, SensitivityTable};

#[cfg(test)]
#[path = "exact_sweep_tests.rs"]
mod exact_sweep_tests;

/// Everything the sweep needs besides the model.
pub struct ExactSweepRequest<'request> {
    pub calibration_samples: &'request [CalibrationForwardSample],
    pub candidate_bit_widths: &'request CandidateBitWidths,
    pub group_size: GroupSize,
    /// Which reference the probe model embodies; recorded on the table.
    pub method: SensitivityMethod,
    /// Where to persist partial results after every module; `None` disables
    /// checkpointing.
    pub checkpoint_path: Option<&'request Path>,
    pub progress_reporter: &'request mut dyn ProgressReporter,
}

/// A few unmatched modules are normal (a tied head, a multi-token-prediction
/// block, a tower the adapter drops). Losing more than half is not a
/// degraded measurement; it is no measurement.
pub const MAXIMUM_SKIPPED_MODULE_FRACTION: f64 = 0.5;

/// Counting modules is not enough: a mixture-of-experts model's experts are a
/// handful of fused tensors holding most of its weights, so losing all of them
/// barely moves the module count. Nemotron-H skipped 28% of modules, under
/// the count rule, and they were the 128-expert projections.
pub const MAXIMUM_SKIPPED_PARAMETER_FRACTION: f64 = 0.25;

/// Runs the sweep and returns the table.
///
/// Work to port (OptiQ's `_exact_with_bf16_reference` and
/// `_exact_with_quantized_baseline`, merged):
/// 1. Load the checkpoint if present; a module whose entry covers every
///    candidate width is complete and is not re-probed.
/// 2. Forward every calibration sample once through the unmodified model and
///    keep the reference logits (the probe may store them as float16).
/// 3. For each module from `quantizable_modules`, in order:
///    - skip and count it when the probe cannot resolve a source for it
///      (`simulate_quantize_module` returns `ProbeModelFailure` naming the
///      missing source); the first five skips are logged;
///    - for each candidate width: zero KL without a forward when the width is
///      the keep-bfloat16 tier or equals `current_bit_width`; otherwise
///      simulate-quantize, forward every sample, average the KL against the
///      stored reference, restore the module;
///    - the keep-bfloat16 tier against a uniform reference is an error
///      (`KeepBfloat16TierNeedsBfloat16Reference`), not a silent zero;
///    - record the result, release transient memory, advance progress with
///      the module name and the per-width KL values as the detail, and merge
///      the checkpoint.
/// 4. Apply [`require_measured_majority`].
///
/// A NaN KL is recorded as NaN; the allocator decides what to do with it.
pub fn run_exact_sweep<ProbeModel: SensitivityProbeModel>(
    probe_model: &mut ProbeModel,
    request: ExactSweepRequest<'_>,
) -> Result<SensitivityTable, QuantizationError> {
    let _ = (probe_model, request);
    todo!("sequence the probes as documented above")
}

/// Refuses a sweep that measured almost nothing, by count or by parameter mass.
pub fn require_measured_majority(
    skipped_module_count: usize,
    total_module_count: usize,
    skipped_parameter_count: u64,
    total_parameter_count: u64,
) -> Result<(), QuantizationError> {
    let _ = (
        skipped_module_count,
        total_module_count,
        skipped_parameter_count,
        total_parameter_count,
    );
    todo!("fail with SensitivitySweepMeasuredTooLittle when either fraction exceeds its maximum")
}

/// Averages per-sample KL values; the per-module, per-width score.
#[must_use]
pub fn mean_kl_divergence(per_sample_kl_divergences: &[f64]) -> f64 {
    let _ = per_sample_kl_divergences;
    todo!("arithmetic mean; zero for an empty slice")
}

/// Builds the result for one module from its per-width means.
#[must_use]
pub fn sensitivity_result(
    module_name: &str,
    logical_parameter_count: u64,
    kl_divergence_by_bit_width: BTreeMap<crate::bit_width::BitWidth, f64>,
) -> SensitivityResult {
    SensitivityResult {
        module_name: module_name.to_owned(),
        kl_divergence_by_bit_width,
        logical_parameter_count,
    }
}
