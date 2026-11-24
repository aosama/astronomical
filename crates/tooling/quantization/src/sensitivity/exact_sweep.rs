//! The exact per-module KL sweep driver.
//!
//! The driver is engine-agnostic: it sequences probes through a
//! [`SensitivityProbeModel`], keeps the checkpoint monotonic, reports
//! progress with an ETA, and enforces the measured-majority guard. It does
//! not know whether the reference is bf16 or a uniform baseline; the probe
//! model's `current_bit_width` answers tell it which widths are free.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::module_inventory::QuantizableModuleDescriptor;
use crate::progress::{ProgressReporter, StageProgressTracker};

use super::checkpoint::{complete_entries_for, load_checkpoint, merge_and_save_checkpoint};
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

/// Runs the sweep and returns the table, OptiQ's
/// `_exact_with_bf16_reference` and `_exact_with_quantized_baseline` merged:
/// 1. A module whose checkpoint entry covers every candidate width is
///    complete and is not re-probed.
/// 2. Every calibration sample is forwarded once through the unmodified
///    model and its reference logits kept.
/// 3. Each remaining module is probed at each width: zero KL with no forward
///    for the keep-bfloat16 tier or the width the baseline already holds;
///    otherwise simulate-quantize, forward every sample, average the KL
///    against the reference, and restore. A module the probe cannot resolve
///    a source for (`ProbeModelFailure` from the simulate step) is skipped
///    and counted. After each module, transient memory is released,
///    progress advances with the per-width KL values, and the checkpoint is
///    merged.
/// 4. [`require_measured_majority`] refuses a sweep that measured too little.
///
/// The keep-bfloat16 tier against a uniform reference is refused up front
/// (`KeepBfloat16TierNeedsBfloat16Reference`) rather than recorded as a
/// silent zero. A NaN KL is recorded as NaN; the allocator decides what to
/// do with it.
pub fn run_exact_sweep<ProbeModel: SensitivityProbeModel>(
    probe_model: &mut ProbeModel,
    request: ExactSweepRequest<'_>,
) -> Result<SensitivityTable, QuantizationError> {
    let ladder = request.candidate_bit_widths.as_slice();
    let has_keep_bfloat16_tier = request.candidate_bit_widths.highest().keeps_bfloat16();
    if has_keep_bfloat16_tier && request.method == SensitivityMethod::ExactUniform4BitReference {
        return Err(QuantizationError::KeepBfloat16TierNeedsBfloat16Reference);
    }
    let completed_results = completed_results_by_name(request.checkpoint_path, ladder);
    let modules = probe_model.quantizable_modules();
    let mut reference_logits = Vec::with_capacity(request.calibration_samples.len());
    for calibration_sample in request.calibration_samples {
        reference_logits.push(probe_model.forward(calibration_sample)?);
    }
    let mut sweep_tally = SweepTally::default();
    let mut progress_tracker = StageProgressTracker::new(
        "sensitivity sweep",
        modules.len() as u64,
        count_completed(&modules, &completed_results),
        StageProgressTracker::DEFAULT_REPORT_INTERVAL,
        request.progress_reporter,
    );
    let probe_context = ProbeContext {
        calibration_samples: request.calibration_samples,
        reference_logits: &reference_logits,
        ladder,
        group_size: request.group_size,
    };
    for module in &modules {
        sweep_tally.total_parameter_count += module.logical_parameter_count;
        if let Some(completed_result) = completed_results.get(&module.module_name) {
            sweep_tally.results.push(completed_result.clone());
            continue;
        }
        let progress_detail =
            sweep_one_module(probe_model, &probe_context, module, &mut sweep_tally)?;
        progress_tracker.advance(&progress_detail);
        if let Some(checkpoint_path) = request.checkpoint_path {
            merge_and_save_checkpoint(checkpoint_path, &sweep_tally.results)?;
        }
    }
    progress_tracker.report_now("sweep complete");
    require_measured_majority(
        sweep_tally.skipped_module_count,
        modules.len(),
        sweep_tally.skipped_parameter_count,
        sweep_tally.total_parameter_count,
    )?;
    Ok(SensitivityTable {
        method: request.method,
        results: sweep_tally.results,
    })
}

/// Refuses a sweep that measured almost nothing, by count or by parameter
/// mass: either fraction above its maximum fails, because either alone
/// misses a real case (muse_glimmer by count, Nemotron-H by mass).
pub fn require_measured_majority(
    skipped_module_count: usize,
    total_module_count: usize,
    skipped_parameter_count: u64,
    total_parameter_count: u64,
) -> Result<(), QuantizationError> {
    let skipped_too_many_modules = total_module_count > 0
        && skipped_module_count as f64
            > total_module_count as f64 * MAXIMUM_SKIPPED_MODULE_FRACTION;
    let skipped_too_much_mass = total_parameter_count > 0
        && skipped_parameter_count as f64
            > total_parameter_count as f64 * MAXIMUM_SKIPPED_PARAMETER_FRACTION;
    if !skipped_too_many_modules && !skipped_too_much_mass {
        return Ok(());
    }
    let skipped_parameter_share_percent = if total_parameter_count > 0 {
        100.0 * skipped_parameter_count as f64 / total_parameter_count as f64
    } else {
        0.0
    };
    Err(QuantizationError::SensitivitySweepMeasuredTooLittle {
        measured_module_count: total_module_count - skipped_module_count.min(total_module_count),
        total_module_count,
        skipped_parameter_share_percent,
    })
}

/// Averages per-sample KL values; the per-module, per-width score. Zero for
/// an empty slice.
#[must_use]
pub fn mean_kl_divergence(per_sample_kl_divergences: &[f64]) -> f64 {
    if per_sample_kl_divergences.is_empty() {
        return 0.0;
    }
    let mut kl_divergence_sum = 0.0;
    for kl_divergence in per_sample_kl_divergences {
        kl_divergence_sum += kl_divergence;
    }
    kl_divergence_sum / per_sample_kl_divergences.len() as f64
}

/// How many modules OptiQ logs individually before only counting skips.
const LOGGED_SKIP_LIMIT: usize = 5;

/// What one module's probes need from the sweep.
struct ProbeContext<'sweep, Logits> {
    calibration_samples: &'sweep [CalibrationForwardSample],
    reference_logits: &'sweep [Logits],
    ladder: &'sweep [BitWidth],
    group_size: GroupSize,
}

/// The sweep's running results and skip accounting.
#[derive(Default)]
struct SweepTally {
    results: Vec<SensitivityResult>,
    skipped_module_count: usize,
    skipped_parameter_count: u64,
    total_parameter_count: u64,
}

/// Checkpointed results covering every width of the ladder, by module name.
fn completed_results_by_name(
    checkpoint_path: Option<&Path>,
    ladder: &[BitWidth],
) -> BTreeMap<String, SensitivityResult> {
    let mut completed_results = BTreeMap::new();
    let Some(checkpoint_path) = checkpoint_path else {
        return completed_results;
    };
    for completed_result in complete_entries_for(load_checkpoint(checkpoint_path), ladder) {
        completed_results.insert(completed_result.module_name.clone(), completed_result);
    }
    completed_results
}

fn count_completed(
    modules: &[QuantizableModuleDescriptor],
    completed_results: &BTreeMap<String, SensitivityResult>,
) -> u64 {
    let mut completed_count = 0;
    for module in modules {
        if completed_results.contains_key(&module.module_name) {
            completed_count += 1;
        }
    }
    completed_count
}

/// Probes one module at every width, recording its result or its skip, and
/// returns the progress detail.
fn sweep_one_module<ProbeModel: SensitivityProbeModel>(
    probe_model: &mut ProbeModel,
    probe_context: &ProbeContext<'_, ProbeModel::Logits>,
    module: &QuantizableModuleDescriptor,
    sweep_tally: &mut SweepTally,
) -> Result<String, QuantizationError> {
    let mut kl_divergence_by_bit_width = BTreeMap::new();
    for bit_width in probe_context.ladder {
        match probe_one_width(probe_model, probe_context, &module.module_name, *bit_width) {
            Ok(kl_divergence) => {
                kl_divergence_by_bit_width.insert(*bit_width, kl_divergence);
            }
            Err(QuantizationError::ProbeModelFailure { description }) => {
                record_skip(sweep_tally, module, &description);
                probe_model.release_transient_memory()?;
                return Ok(format!("{} skipped", module.module_name));
            }
            Err(probe_error) => return Err(probe_error),
        }
    }
    probe_model.release_transient_memory()?;
    let progress_detail = describe_module_result(&module.module_name, &kl_divergence_by_bit_width);
    sweep_tally.results.push(sensitivity_result(
        &module.module_name,
        module.logical_parameter_count,
        kl_divergence_by_bit_width,
    ));
    Ok(progress_detail)
}

/// The mean KL at one width, or zero with no forward when the width is free.
/// The module is restored even when a forward fails mid-probe, so an error
/// never leaves the model altered.
fn probe_one_width<ProbeModel: SensitivityProbeModel>(
    probe_model: &mut ProbeModel,
    probe_context: &ProbeContext<'_, ProbeModel::Logits>,
    module_name: &str,
    bit_width: BitWidth,
) -> Result<f64, QuantizationError> {
    let is_baseline_width = probe_model.current_bit_width(module_name) == Some(bit_width);
    if bit_width.keeps_bfloat16() || is_baseline_width {
        return Ok(0.0);
    }
    let restore_token =
        probe_model.simulate_quantize_module(module_name, bit_width, probe_context.group_size)?;
    let measured_outcome = measure_against_reference(probe_model, probe_context);
    probe_model.restore_module(restore_token)?;
    Ok(mean_kl_divergence(&measured_outcome?))
}

/// One KL per calibration sample against its stored reference logits.
fn measure_against_reference<ProbeModel: SensitivityProbeModel>(
    probe_model: &mut ProbeModel,
    probe_context: &ProbeContext<'_, ProbeModel::Logits>,
) -> Result<Vec<f64>, QuantizationError> {
    let mut per_sample_kl_divergences = Vec::with_capacity(probe_context.calibration_samples.len());
    let paired_samples = probe_context
        .calibration_samples
        .iter()
        .zip(probe_context.reference_logits);
    for (calibration_sample, reference_logits) in paired_samples {
        let probe_logits = probe_model.forward(calibration_sample)?;
        per_sample_kl_divergences.push(probe_model.kl_divergence(reference_logits, &probe_logits)?);
    }
    Ok(per_sample_kl_divergences)
}

fn record_skip(
    sweep_tally: &mut SweepTally,
    module: &QuantizableModuleDescriptor,
    description: &str,
) {
    sweep_tally.skipped_module_count += 1;
    sweep_tally.skipped_parameter_count += module.logical_parameter_count;
    if sweep_tally.skipped_module_count <= LOGGED_SKIP_LIMIT {
        tracing::warn!(module_name = %module.module_name, %description, "skipping a module the probe cannot resolve");
    }
}

/// `module kl@4=1.000e-02 kl@8=2.000e-03`, the progress line's detail.
fn describe_module_result(
    module_name: &str,
    kl_divergence_by_bit_width: &BTreeMap<BitWidth, f64>,
) -> String {
    let mut progress_detail = module_name.to_owned();
    for (bit_width, kl_divergence) in kl_divergence_by_bit_width {
        progress_detail.push_str(&format!(" kl@{}={kl_divergence:.3e}", bit_width.bits()));
    }
    progress_detail
}

/// Builds the result for one module from its per-width means.
#[must_use]
pub fn sensitivity_result(
    module_name: &str,
    logical_parameter_count: u64,
    kl_divergence_by_bit_width: BTreeMap<BitWidth, f64>,
) -> SensitivityResult {
    SensitivityResult {
        module_name: module_name.to_owned(),
        kl_divergence_by_bit_width,
        logical_parameter_count,
    }
}
