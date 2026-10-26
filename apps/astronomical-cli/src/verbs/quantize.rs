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

// Arguments for `astronomical quantize`, an ephemeral in-process verb with
// three sub-verbs.
//
// - `quantize plan --sensitivity FILE --target-bits-per-weight N [...]`:
//   allocate from an existing `optiq/sensitivity.json` and print the bit
//   map. Pure; runs on any host; the way to re-target an already-measured
//   base without a sweep.
// - `quantize inspect ARTIFACT_DIR`: print an artifact's quantization
//   provenance (method, reference, target and achieved bits per weight,
//   per-width module counts). Pure.
// - `quantize convert SOURCE_DIR --output DIR [...]`: the full pipeline.
//   Needs the `quantize-convert` build feature.
//
// Flag names spell every word out, per the repository's naming rule:
// `--target-bits-per-weight`, never `--target-bpw`.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::arguments::UsageError;

/// Which quantize sub-verb was requested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizeCommand {
    Plan(QuantizePlanArguments),
    Inspect(QuantizeInspectArguments),
    Convert(QuantizeConvertArguments),
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizePlanArguments {
    pub sensitivity_document_path: PathBuf,
    pub target_bits_per_weight: f64,
    /// `None` infers the ladder from the document.
    pub candidate_bits: Option<Vec<u32>>,
    pub floor_components_per_block: usize,
    pub maximum_low_bit_block_run: usize,
    pub render_json: bool,
}

impl Eq for QuantizePlanArguments {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizeInspectArguments {
    pub artifact_directory: PathBuf,
    pub render_json: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizeConvertArguments {
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    pub base_model_name: Option<String>,
    /// `exact` (default) or `static`.
    pub method: QuantizeMethodArgument,
    /// `auto` (default), `bf16`, or `uniform-4-bit`.
    pub reference: QuantizeReferenceArgument,
    pub target_bits_per_weight: f64,
    pub candidate_bits: Vec<u32>,
    pub group_size: u32,
    pub calibration_sequence_count: usize,
    pub calibration_mix_path: Option<PathBuf>,
    pub floor_components_per_block: usize,
    pub range_search_enabled: bool,
    pub range_search_maximum_bits: u32,
    pub overwrite_output: bool,
}

impl Eq for QuantizeConvertArguments {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizeMethodArgument {
    Exact,
    Static,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizeReferenceArgument {
    Auto,
    Bfloat16,
    Uniform4Bit,
}

/// OptiQ's command-line defaults, kept so a Rust conversion with no flags
/// matches a Python conversion with no flags.
pub const DEFAULT_TARGET_BITS_PER_WEIGHT: f64 = 5.0;
pub const DEFAULT_CANDIDATE_BITS: [u32; 2] = [4, 8];
pub const DEFAULT_GROUP_SIZE: u32 = 64;
pub const DEFAULT_CALIBRATION_SEQUENCE_COUNT: usize = 24;
pub const DEFAULT_FLOOR_COMPONENTS_PER_BLOCK: usize = 2;
pub const DEFAULT_MAXIMUM_LOW_BIT_BLOCK_RUN: usize = 3;
pub const DEFAULT_RANGE_SEARCH_MAXIMUM_BITS: u32 = 3;

/// Parses everything after `astronomical quantize`.
///
/// Work to implement: dispatch on the first argument (`plan`, `inspect`,
/// `convert`; missing → `MissingCommand`; other →
/// `UnknownQuantizeTarget`), then parse the sub-verb's flags with the same
/// repeated-flag and missing-value discipline as the other verbs. A
/// `--candidate-bits` value is a comma-separated list of integers; a
/// non-integer or an unsupported width is `InvalidQuantizeValue`. Floats
/// for `--target-bits-per-weight` must be finite and positive.
pub fn parse_quantize_command(
    remaining_arguments: &[OsString],
) -> Result<QuantizeCommand, UsageError> {
    let _ = remaining_arguments;
    todo!("dispatch on the sub-verb and parse its flags")
}

/// Parses a comma-separated bit-width list such as `2,4,8`.
pub fn parse_candidate_bits(value: &str) -> Result<Vec<u32>, UsageError> {
    let _ = value;
    todo!(
        "split on commas, parse each as u32, validate through astronomical_quantization::BitWidth"
    )
}

#[cfg(test)]
mod quantize_tests;
