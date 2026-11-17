//! What `decision calibrate` and `decision train` share: their collaborators,
//! their failures, and the steps both take around their own work (reading
//! the examples, preparing the model, checking the output folder, writing
//! and announcing the new folder).

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_config::{
    AstronomicalInstancePaths, AstronomicalRuntimeInstance, is_discoverable_model_folder,
};
use astronomical_decision_contract::training_record::TrainingRecord;
use astronomical_decision_training::calibrated_model_folder::{
    CalibratedFolderError, FolderAdditions, SourceModelFolder, canonical_output_path,
    check_output_location, write_calibrated_model_folder,
};
use astronomical_decision_training::calibration_plan::CalibrationPlanError;
use astronomical_decision_training::labelled_example::{
    LabelledExample, LabelledExampleError, parse_labelled_examples,
};
use astronomical_decision_training::scored_question::ScoredQuestionError;
use astronomical_decision_training::training_record_file::read_training_record_file;
use thiserror::Error;

use super::model_folder_lookup::{ModelFolderLookup, ModelFolderLookupError};
use super::new_model_visibility::{make_new_model_visible, written_folder_text};
use super::progress_reporting::{CopyProgressLine, TimedProgress};
use crate::verbs::decide::{DecideError, DecisionWaitBounds};
use crate::{DaemonProbe, ModelLifecycle, RequiredCapability};

/// Collaborators both verbs need, injected so tests can stub them.
pub struct DecisionToolDependencies<'a> {
    pub candidate_socket_paths: Vec<PathBuf>,
    pub runtime_instance: AstronomicalRuntimeInstance,
    /// The daemon's loopback HTTP address, for the reload that makes the
    /// new model servable.
    pub loopback_address: SocketAddr,
    /// The instance whose models are looked up, to find the model's folder.
    pub instance_paths: AstronomicalInstancePaths,
    /// Where the report goes.
    pub stdout: &'a mut dyn Write,
    /// Where progress goes.
    pub stderr: &'a mut dyn Write,
    pub wait_bounds: DecisionWaitBounds,
    pub download_stall_bound: Duration,
    pub download_poll_interval: Duration,
    /// How often a long step reports its progress and time left.
    pub progress_interval: Duration,
}

/// Failures of either verb after its arguments have parsed.
#[derive(Debug, Error)]
pub enum DecisionToolError {
    #[error("Could not read {examples_file_path}: {cause}")]
    ExamplesUnreadable {
        examples_file_path: PathBuf,
        cause: String,
    },
    #[error("{0}")]
    Examples(#[from] LabelledExampleError),
    #[error(transparent)]
    Decide(#[from] DecideError),
    #[error(transparent)]
    ModelFolder(#[from] ModelFolderLookupError),
    #[error(
        "{output_name} is already the name of an installed model; choose another output folder name"
    )]
    OutputNameTaken { output_name: String },
    #[error(transparent)]
    Folder(#[from] CalibratedFolderError),
    #[error("example on line {line_number}: {source}")]
    Scoring {
        line_number: usize,
        source: ScoredQuestionError,
    },
    #[error("example on line {line_number}: {asked} questions were asked but {answered} answered")]
    AnswerCountMismatch {
        line_number: usize,
        asked: usize,
        answered: usize,
    },
    #[error(transparent)]
    Plan(#[from] CalibrationPlanError),
    #[error("Could not write the report: {cause}")]
    ReportUnwritable { cause: String },
    #[error("{model_id} is not a decision model training can change")]
    NotTrainable { model_id: String },
    #[error("the training did not finish: {0}")]
    TrainingFailed(String),
    #[error(
        "the written model answered question {question_name} {largest_gap:.5} away from the trained model; the new folder is kept so you can look at it, but it may not answer as the report describes"
    )]
    SavedModelDisagrees {
        question_name: String,
        largest_gap: f32,
    },
    #[error(
        "the written model could not be checked ({0}); the new folder is kept, so ask it a question to try it"
    )]
    SavedModelUnchecked(String),
    #[error(
        "{}",
        output_not_discoverable_text(output_directory, scanned_model_roots)
    )]
    OutputNotDiscoverable {
        output_directory: PathBuf,
        scanned_model_roots: Vec<PathBuf>,
    },
    #[error("the built-in check set cannot be read: {0}")]
    CheckSetUnreadable(String),
    #[error("the model folder's earlier calibration cannot be read: {0}")]
    EarlierCalibrationUnreadable(String),
    #[error("the model folder's earlier training record cannot be read: {0}")]
    EarlierRecordUnreadable(String),
}

impl DecisionToolError {
    /// Whether the failure is the invocation's fault, so the process exits 2.
    #[must_use]
    pub fn is_usage_error(&self) -> bool {
        match self {
            Self::ExamplesUnreadable { .. }
            | Self::Examples(_)
            | Self::OutputNameTaken { .. }
            | Self::NotTrainable { .. }
            | Self::OutputNotDiscoverable { .. }
            | Self::ModelFolder(ModelFolderLookupError::ModelNotFound { .. })
            | Self::Folder(
                CalibratedFolderError::OutputExists { .. }
                | CalibratedFolderError::OutputOverlapsSource,
            ) => true,
            Self::Decide(decide_error) => decide_error.is_usage_error(),
            _ => false,
        }
    }
}

pub(super) fn read_labelled_examples(
    examples_file_path: &Path,
) -> Result<Vec<LabelledExample>, DecisionToolError> {
    let examples_text = match std::fs::read_to_string(examples_file_path) {
        Ok(examples_text) => examples_text,
        Err(read_error) => {
            return Err(DecisionToolError::ExamplesUnreadable {
                examples_file_path: examples_file_path.to_path_buf(),
                cause: read_error.to_string(),
            });
        }
    };
    Ok(parse_labelled_examples(&examples_text)?)
}

pub(super) fn model_lifecycle(dependencies: &DecisionToolDependencies<'_>) -> ModelLifecycle {
    ModelLifecycle {
        daemon_probe: DaemonProbe {
            candidate_socket_paths: dependencies.candidate_socket_paths.clone(),
            request_timeout: dependencies.wait_bounds.request_timeout,
            runtime_instance: dependencies.runtime_instance,
        },
        download_stall_bound: dependencies.download_stall_bound,
        download_poll_interval: dependencies.download_poll_interval,
    }
}

/// Resolves the decision model, downloading it when this Mac lacks it.
pub(super) async fn prepare_decision_model(
    requested_model_id: Option<&str>,
    model_lifecycle: &ModelLifecycle,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<String, DecisionToolError> {
    let stderr = &mut *dependencies.stderr;
    let mut progress_started = false;
    let prepared_model_id = model_lifecycle
        .prepare_model_id(
            requested_model_id,
            RequiredCapability::Decisions,
            &mut |progress_line: &str| {
                progress_started = true;
                let _ = write!(stderr, "\r{progress_line}");
                let _ = stderr.flush();
            },
        )
        .await;
    if progress_started {
        let _ = writeln!(stderr);
    }
    match prepared_model_id {
        Ok(decision_model_id) => Ok(decision_model_id),
        Err(lifecycle_error) => Err(DecisionToolError::Decide(DecideError::from(
            lifecycle_error,
        ))),
    }
}

pub(super) fn check_output_is_new(
    model_folder: &ModelFolderLookup,
    output_directory: &Path,
) -> Result<(), DecisionToolError> {
    check_output_location(&model_folder.model_directory, output_directory)?;
    check_output_is_discoverable(&model_folder.scanned_model_roots, output_directory)?;
    let output_name = output_model_id(output_directory);
    if model_folder.installed_model_ids.contains(&output_name) {
        return Err(DecisionToolError::OutputNameTaken { output_name });
    }
    Ok(())
}

/// Refuses an output folder Astronomical would not find, before any example
/// is read. The written folder is served and checked at the end, so one that
/// discovery never finds would cost the whole run and then fail its check.
pub(super) fn check_output_is_discoverable(
    scanned_model_roots: &[PathBuf],
    output_directory: &Path,
) -> Result<(), DecisionToolError> {
    let canonical_output = canonical_output_path(output_directory)?;
    for model_root in scanned_model_roots {
        // A configured folder that does not exist yet holds no models.
        let Ok(canonical_root) = std::fs::canonicalize(model_root) else {
            continue;
        };
        if is_discoverable_model_folder(&canonical_root, &canonical_output) {
            return Ok(());
        }
    }
    Err(DecisionToolError::OutputNotDiscoverable {
        output_directory: output_directory.to_path_buf(),
        scanned_model_roots: scanned_model_roots.to_vec(),
    })
}

fn output_not_discoverable_text(
    output_directory: &Path,
    scanned_model_roots: &[PathBuf],
) -> String {
    let mut root_list = String::new();
    for model_root in scanned_model_roots {
        root_list.push_str(&format!("\n  {}", model_root.display()));
    }
    let suggested_name = output_model_id(output_directory);
    let suggested_root = scanned_model_roots
        .first()
        .map_or_else(|| PathBuf::from("MODELS"), PathBuf::clone);
    format!(
        "Astronomical would not find a model in {}, so the new model could not be served or checked. Choose a folder inside one of the folders it looks in, at most 4 folders down and not inside another model, for example --output {}:{root_list}",
        output_directory.display(),
        suggested_root
            .join("calibrated")
            .join(suggested_name)
            .display()
    )
}

/// The training record a source folder already carries, when it was itself
/// made by `astronomical decision`; its findings still hold for the parts
/// the new folder copies, so the new record builds on it.
pub(super) fn earlier_training_record(
    model_directory: &Path,
) -> Result<Option<TrainingRecord>, DecisionToolError> {
    read_training_record_file(model_directory).map_err(|record_error| {
        DecisionToolError::EarlierRecordUnreadable(record_error.to_string())
    })
}

/// The ID the daemon serves a folder under: its name.
pub(super) fn output_model_id(output_directory: &Path) -> String {
    output_directory
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Copies the model with its additions and proves the copy, reporting progress.
pub(super) fn write_model_folder(
    output_directory: &Path,
    model_folder: &ModelFolderLookup,
    folder_additions: FolderAdditions<'_>,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let mut copy_progress = CopyProgressLine {
        stderr: &mut *dependencies.stderr,
        timed_progress: TimedProgress::new(
            "Copying and checking the model",
            "MB",
            dependencies.progress_interval,
        ),
    };
    write_calibrated_model_folder(
        &SourceModelFolder {
            model_directory: &model_folder.model_directory,
            other_model_subdirectories: &model_folder.nested_model_directories,
        },
        output_directory,
        folder_additions,
        &mut copy_progress,
    )?;
    Ok(())
}

/// Has the daemon look for the new folder, then says how to ask for it.
pub(super) async fn announce_model_folder(
    output_directory: &Path,
    model_folder: &ModelFolderLookup,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let new_model_id = output_model_id(output_directory);
    let visibility =
        make_new_model_visible(dependencies.loopback_address, daemon_probe, &new_model_id).await;
    write_report(
        dependencies.stdout,
        &written_folder_text(
            output_directory,
            &new_model_id,
            visibility,
            &model_folder.scanned_model_roots,
        ),
    )
}

pub(super) fn write_report(
    stdout: &mut dyn Write,
    report_text: &str,
) -> Result<(), DecisionToolError> {
    let write_outcome = match stdout.write_all(report_text.as_bytes()) {
        Ok(()) => stdout.flush(),
        Err(write_error) => Err(write_error),
    };
    match write_outcome {
        Ok(()) => Ok(()),
        Err(write_error) => Err(DecisionToolError::ReportUnwritable {
            cause: write_error.to_string(),
        }),
    }
}

#[cfg(test)]
#[path = "decision_tool_tests.rs"]
mod tests;
