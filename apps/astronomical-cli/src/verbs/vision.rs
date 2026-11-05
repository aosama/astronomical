//! The `astronomical vision` verb bodies: `inspect` and `match` ask questions about
//! a model's vision tower without downloading its weights, and `add` builds a new
//! model from a text model and a tower. None of them need a daemon or a GPU.
//!
//! Progress goes to standard error at the given interval, with a percentage and an
//! estimated time to completion; standard output carries only the final report (or
//! JSON when requested) so the verb composes in scripts.

use std::io::Write;
use std::time::{Duration, Instant};

use astronomical_vision_tower::{
    ArtifactReference, ArtifactSource, BaseMatchProgress, BaseMatchVerdict, TowerAdditionProgress,
    TowerAdditionReport, TowerAdditionRequest, TowerAdditionStage, TowerStatus, VisionTowerError,
    add_tower, inspect_artifact, match_tower_to_text,
};

mod vision_arguments;
mod vision_reports;

pub use vision_arguments::{
    VisionAddArguments, VisionCommand, VisionInspectArguments, VisionMatchArguments,
    parse_vision_command,
};

const COMPARE_STAGE_NAME: &str = "compare";
const BYTES_PER_GIBIBYTE: f64 = 1_073_741_824.0;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("{0}")]
    VisionTower(#[from] VisionTowerError),
    #[error("could not write the vision report to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

/// Collaborators injected so the journeys can capture output.
pub struct VisionDependencies<'a> {
    pub stdout: &'a mut dyn Write,
    pub stderr: &'a mut dyn Write,
    /// Cadence for progress lines on standard error.
    pub progress_interval: Duration,
}

/// `vision inspect`: whether a model has a tower. Returns the status because "no
/// tower" is a successful answer, and the caller decides what exit that means.
pub async fn run_vision_inspect(
    inspect_arguments: &VisionInspectArguments,
    dependencies: &mut VisionDependencies<'_>,
) -> Result<TowerStatus, VisionError> {
    let source = open_model(&inspect_arguments.model).await?;
    let inspection = inspect_artifact(&source).await?;
    let render_outcome = if inspect_arguments.render_json {
        write_json_line(&inspection, dependencies.stdout)
    } else {
        vision_reports::render_inspection_text(&inspection, dependencies.stdout)
    };
    render_outcome.map_err(VisionError::stdout_unwritable)?;
    Ok(inspection.status)
}

/// `vision match`: whether a tower shares a text model's base.
pub async fn run_vision_match(
    match_arguments: &VisionMatchArguments,
    dependencies: &mut VisionDependencies<'_>,
) -> Result<BaseMatchVerdict, VisionError> {
    let text_source = open_model(&match_arguments.text_model).await?;
    let tower_source = open_model(&match_arguments.tower_model).await?;
    let mut progress =
        StandardErrorProgress::new(&mut *dependencies.stderr, dependencies.progress_interval);
    let report = match_tower_to_text(&text_source, &tower_source, &mut progress).await?;
    let render_outcome = if match_arguments.render_json {
        write_json_line(&report, dependencies.stdout)
    } else {
        vision_reports::render_match_text(&report, dependencies.stdout)
    };
    render_outcome.map_err(VisionError::stdout_unwritable)?;
    Ok(report.verdict)
}

/// `vision add`: build a new model from a local text model and a tower. A refusal
/// is an error, so the verb exits with failure and leaves nothing at the output.
pub async fn run_vision_add(
    add_arguments: &VisionAddArguments,
    dependencies: &mut VisionDependencies<'_>,
) -> Result<TowerAdditionReport, VisionError> {
    let text_source = open_model(&add_arguments.text_model).await?;
    let tower_source = open_model(&add_arguments.tower_model).await?;
    let mut progress =
        StandardErrorProgress::new(&mut *dependencies.stderr, dependencies.progress_interval);
    let report = add_tower(
        TowerAdditionRequest {
            text_source: &text_source,
            tower_source: &tower_source,
            output_directory: &add_arguments.output_directory,
        },
        &mut progress,
    )
    .await?;
    let render_outcome = if add_arguments.render_json {
        write_json_line(&report, dependencies.stdout)
    } else {
        vision_reports::render_addition_text(&report, dependencies.stdout)
    };
    render_outcome.map_err(VisionError::stdout_unwritable)?;
    Ok(report)
}

async fn open_model(model_text: &str) -> Result<ArtifactSource, VisionTowerError> {
    ArtifactSource::open(&ArtifactReference::parse(model_text)?).await
}

/// Reports at most once per interval, so a fast run prints nothing and a slow one
/// still shows it is alive. The clock restarts with each stage, so one stage's pace
/// never distorts the next one's estimate.
struct StandardErrorProgress<'a> {
    stderr: &'a mut dyn Write,
    progress_interval: Duration,
    stage_name: &'static str,
    stage_started_at: Instant,
    last_report_at: Instant,
}

impl<'a> StandardErrorProgress<'a> {
    fn new(stderr: &'a mut dyn Write, progress_interval: Duration) -> Self {
        Self {
            stderr,
            progress_interval,
            stage_name: COMPARE_STAGE_NAME,
            stage_started_at: Instant::now(),
            last_report_at: Instant::now(),
        }
    }

    fn enter_stage(&mut self, stage_name: &'static str) {
        if self.stage_name != stage_name {
            self.stage_name = stage_name;
            self.stage_started_at = Instant::now();
        }
    }

    fn report_if_due(&mut self, completed_units: f64, total_units: f64, detail: &str) {
        if self.last_report_at.elapsed() < self.progress_interval || completed_units <= 0.0 {
            return;
        }
        self.last_report_at = Instant::now();
        let elapsed_seconds = self.stage_started_at.elapsed().as_secs_f64();
        let remaining_seconds = elapsed_seconds / completed_units * (total_units - completed_units);
        let _ = writeln!(
            self.stderr,
            "[{}] {detail} ({:.0}%) elapsed {elapsed_seconds:.0}s eta {remaining_seconds:.0}s",
            self.stage_name,
            100.0 * completed_units / total_units,
        );
    }
}

impl BaseMatchProgress for StandardErrorProgress<'_> {
    fn tensor_compared(&mut self, completed_count: usize, total_count: usize) {
        self.enter_stage(COMPARE_STAGE_NAME);
        self.report_if_due(
            completed_count as f64,
            total_count as f64,
            &format!("{completed_count}/{total_count}"),
        );
    }
}

impl TowerAdditionProgress for StandardErrorProgress<'_> {
    fn stage_bytes(&mut self, stage: TowerAdditionStage, completed_bytes: u64, total_bytes: u64) {
        self.enter_stage(match stage {
            TowerAdditionStage::FetchingTower => "tower",
            TowerAdditionStage::HashingFiles => "hash",
        });
        self.report_if_due(
            completed_bytes as f64,
            total_bytes as f64,
            &format!(
                "{:.2}/{:.2} GiB",
                completed_bytes as f64 / BYTES_PER_GIBIBYTE,
                total_bytes as f64 / BYTES_PER_GIBIBYTE
            ),
        );
    }
}

impl VisionError {
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

#[cfg(test)]
mod vision_tests;
