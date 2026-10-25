//! The `astronomical quantize` verb bodies.
//!
//! `plan` and `inspect` are pure and testable with stubbed documents in the
//! integration lane. `convert` is behind the `quantize-convert` feature because
//! it links MLX and the model-serving engine; without the feature it prints
//! how to obtain a build that has it and exits with failure, so the verb is
//! never silently absent from a default binary.
//!
//! Progress for `convert` goes to standard error through
//! [`StandardErrorProgressReporter`] every five seconds with a percentage
//! and an estimated time to completion; standard output carries only the
//! final summary (or JSON when requested) so the verb composes in scripts.

use std::io::Write;
use std::time::Duration;

use astronomical_quantization::progress::{EtaEstimate, ProgressReporter, StageProgress};

use crate::quantize_arguments::{
    QuantizeConvertArguments, QuantizeInspectArguments, QuantizePlanArguments,
};

/// Failures of the quantize verb.
#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("{0}")]
    Quantization(#[from] astronomical_quantization::QuantizationError),
    #[error("could not write the quantize report to standard output: {cause}")]
    StdoutUnwritable { cause: String },
    #[error(
        "astronomical quantize convert is not compiled into this binary; build the CLI with the quantize-convert feature (cargo build -p astronomical-cli --features quantize-convert) to run conversions in-process"
    )]
    ConvertNotCompiledIn,
}

/// Collaborators injected so the journeys can capture output.
pub struct QuantizeDependencies<'a> {
    pub stdout: &'a mut dyn Write,
    pub stderr: &'a mut dyn Write,
    /// Cadence for progress lines on standard error.
    pub progress_interval: Duration,
}

/// Writes `StageProgress` snapshots as one line each on standard error.
pub struct StandardErrorProgressReporter<'a> {
    pub stderr: &'a mut dyn Write,
}

impl ProgressReporter for StandardErrorProgressReporter<'_> {
    fn report(&mut self, progress: &StageProgress) {
        let estimate_text = match &progress.estimate {
            EtaEstimate::NotYetAvailable => "eta pending".to_owned(),
            EtaEstimate::Remaining(remaining) => format!("eta {}s", remaining.as_secs()),
        };
        let _ = writeln!(
            self.stderr,
            "[{}] {}/{} ({:.0}%) elapsed {}s {} {}",
            progress.stage_name,
            progress.completed_work_units,
            progress.total_work_units,
            progress.percent_complete(),
            progress.elapsed.as_secs(),
            estimate_text,
            progress.detail
        );
    }
}

/// `quantize plan`: read the sensitivity document, allocate, and print the
/// per-module bit map plus the summary and any overshoot note.
///
/// Work to implement: `OptiQSensitivityDocument::read` → `to_table` →
/// `AllocationConstraints` from the arguments → `plan_from_sensitivity_table`
/// → render as a table (module, bits, parameters) sorted by module name,
/// then `Target / Achieved bits per weight`, per-width counts, and the
/// overshoot note; or as one JSON document with `--json`.
pub fn run_quantize_plan(
    plan_arguments: &QuantizePlanArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let _ = (plan_arguments, dependencies);
    todo!("read, allocate, render")
}

/// `quantize inspect`: print an artifact's quantization provenance.
///
/// Work to implement: resolve `optiq/metadata.json` (or the legacy root
/// name), parse it as `OptiQArtifactMetadata`, and render method,
/// reference, base model, target and achieved bits per weight, producer,
/// and the per-width module counts from `per_layer`; a directory with no
/// metadata reports "not an OptiQ-format artifact" and succeeds.
pub fn run_quantize_inspect(
    inspect_arguments: &QuantizeInspectArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let _ = (inspect_arguments, dependencies);
    todo!("resolve, parse, render")
}

/// `quantize convert`: the full pipeline in-process.
///
/// Work to implement (feature `quantize-convert`): build a
/// `QuantizationPipelineRequest` from the arguments; read available memory
/// through the model-serving memory package (never a hardwired figure);
/// construct the probe model loader from
/// `astronomical_model_serving::quantization_probe::load_qwen3_5_probe_model`;
/// run `run_quantization_pipeline` with a
/// [`StandardErrorProgressReporter`]; print the outcome summary.
#[cfg(feature = "quantize-convert")]
pub fn run_quantize_convert(
    convert_arguments: &QuantizeConvertArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let _ = (
        convert_arguments,
        dependencies,
        astronomical_model_serving::quantization_probe::load_qwen3_5_probe_model,
    );
    todo!("build the request, load the probe model, run the pipeline, print the summary")
}

/// Without the feature the verb explains how to get a build that has it.
#[cfg(not(feature = "quantize-convert"))]
pub fn run_quantize_convert(
    convert_arguments: &QuantizeConvertArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let _ = (convert_arguments, dependencies);
    Err(QuantizeError::ConvertNotCompiledIn)
}
