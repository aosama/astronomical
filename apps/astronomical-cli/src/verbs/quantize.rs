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

use astronomical_quantization::allocation::AllocationConstraints;
use astronomical_quantization::artifact::json_file::read_json;
use astronomical_quantization::artifact::{
    OptiQArtifactMetadata, OptiQSensitivityDocument, resolve_metadata_path,
};
use astronomical_quantization::bit_width::{CandidateBitWidths, GroupSize};
use astronomical_quantization::pipeline::plan_from_sensitivity_table;
use astronomical_quantization::progress::{EtaEstimate, ProgressReporter, StageProgress};

mod quantize_arguments;
#[cfg(feature = "quantize-convert")]
mod quantize_convert;
mod quantize_reports;

pub use quantize_arguments::{
    DEFAULT_CALIBRATION_SEQUENCE_COUNT, DEFAULT_CANDIDATE_BITS, DEFAULT_FLOOR_COMPONENTS_PER_BLOCK,
    DEFAULT_GROUP_SIZE, DEFAULT_MAXIMUM_LOW_BIT_BLOCK_RUN, DEFAULT_RANGE_SEARCH_MAXIMUM_BITS,
    DEFAULT_TARGET_BITS_PER_WEIGHT, QuantizeCommand, QuantizeConvertArguments,
    QuantizeInspectArguments, QuantizeMethodArgument, QuantizePlanArguments,
    QuantizeReferenceArgument, parse_candidate_bits, parse_quantize_command,
};
#[cfg(feature = "quantize-convert")]
pub use quantize_convert::run_quantize_convert;

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
    #[error("the artifact metadata at {metadata_path} could not be read: {description}")]
    MetadataUnreadable {
        metadata_path: std::path::PathBuf,
        description: String,
    },
    #[error("the machine's physical memory size could not be read: {description}")]
    PhysicalMemoryUnreadable { description: String },
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
/// per-module bit map plus the summary and any overshoot note, or one JSON
/// allocation document with `--json`. The ladder comes from the arguments
/// or, when absent, from the document.
pub fn run_quantize_plan(
    plan_arguments: &QuantizePlanArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let sensitivity_document =
        OptiQSensitivityDocument::read(&plan_arguments.sensitivity_document_path)?;
    let sensitivity_table = sensitivity_document.to_table()?;
    let candidate_bits = match &plan_arguments.candidate_bits {
        Some(candidate_bits) => candidate_bits.clone(),
        None => sensitivity_document.candidate_bits.clone(),
    };
    let constraints = AllocationConstraints {
        target_bits_per_weight: plan_arguments.target_bits_per_weight,
        candidate_bit_widths: Some(CandidateBitWidths::new(&candidate_bits)?),
        group_size: GroupSize::new(sensitivity_document.group_size)?,
        protect_boundary_blocks: true,
        boundary_block_count: 1,
        floor_components_per_block: plan_arguments.floor_components_per_block,
        maximum_low_bit_block_run: plan_arguments.maximum_low_bit_block_run,
    };
    let allocation = plan_from_sensitivity_table(&sensitivity_table, &constraints)?;
    let render_outcome = if plan_arguments.render_json {
        write_json_line(&allocation, dependencies.stdout)
    } else {
        quantize_reports::render_plan_text(
            &allocation,
            plan_arguments.floor_components_per_block,
            dependencies.stdout,
        )
    };
    render_outcome.map_err(QuantizeError::stdout_unwritable)
}

/// `quantize inspect`: print an artifact's quantization provenance. A
/// directory with no metadata is reported as "not an OptiQ-format
/// artifact" and succeeds, because inspecting a plain model is a valid
/// question with a clear answer.
pub fn run_quantize_inspect(
    inspect_arguments: &QuantizeInspectArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let Some(metadata_path) = resolve_metadata_path(&inspect_arguments.artifact_directory) else {
        return writeln!(
            dependencies.stdout,
            "{} is not an OptiQ-format artifact (no optiq/metadata.json)",
            inspect_arguments.artifact_directory.display()
        )
        .map_err(QuantizeError::stdout_unwritable);
    };
    let metadata: OptiQArtifactMetadata = match read_json(&metadata_path) {
        Ok(metadata) => metadata,
        Err(description) => {
            return Err(QuantizeError::MetadataUnreadable {
                metadata_path,
                description,
            });
        }
    };
    let render_outcome = if inspect_arguments.render_json {
        write_json_line(&metadata, dependencies.stdout)
    } else {
        quantize_reports::render_inspect_text(&metadata, dependencies.stdout)
    };
    render_outcome.map_err(QuantizeError::stdout_unwritable)
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

impl QuantizeError {
    /// Named so `map_err` takes a function rather than a closure.
    fn stdout_unwritable(write_error: std::io::Error) -> Self {
        Self::StdoutUnwritable {
            cause: write_error.to_string(),
        }
    }
}

/// One pretty JSON document and a newline.
fn write_json_line<Document: serde::Serialize>(
    document: &Document,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    serde_json::to_writer_pretty(&mut *output, document)?;
    writeln!(output)
}

// The only unit test pins the default build's refusal of `convert`, so the
// module exists only where that refusal is compiled in.
#[cfg(all(test, not(feature = "quantize-convert")))]
mod quantize_tests;
