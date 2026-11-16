//! Sending one training to the daemon and following it to its end.
//!
//! The worker reads every example and trains on the GPU; this side only
//! sends the job, turns each progress frame into a timed progress line, and
//! returns what the training produced. Each frame is awaited on its own, so
//! the silence bound restarts whenever the worker reports a step: a long
//! training is never mistaken for a daemon that stopped answering.

use std::io::Write;
use std::time::Duration;

use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, DecisionTrainingStage,
};
use astronomical_ipc_protocol::{DaemonRequest, DaemonResponse};
use tokio::time::timeout;

use super::decision_tool::{DecisionToolDependencies, DecisionToolError};
use super::progress_reporting::TimedProgress;
use crate::DaemonProbe;
use crate::daemon_frame_wait::{
    DaemonWaitBounds, DaemonWorkWatch, await_frame_watching_daemon_work,
};
use crate::verbs::decide::DecideError;

/// One progress frame's numbers.
#[derive(Clone, Copy, Debug)]
struct StageReport {
    stage: DecisionTrainingStage,
    completed: u64,
    total: u64,
}

/// The progress line of the stage the worker last reported.
struct StageProgress {
    stage: DecisionTrainingStage,
    timed_progress: TimedProgress,
}

/// Has the daemon train `decision_model_id` on `job`, printing progress, and
/// returns what the training produced.
pub(super) async fn request_training(
    decision_model_id: &str,
    job: DecisionTrainingJob,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<DecisionTrainingOutcome, DecisionToolError> {
    let wait_bounds = dependencies.wait_bounds;
    let mut daemon_client = match daemon_probe.connect().await {
        Ok(daemon_client) => daemon_client,
        Err(probe_error) => return Err(DecideError::from(probe_error).into()),
    };
    let training_request = DaemonRequest::TrainDecisionModel {
        model: Some(decision_model_id.to_owned()),
        job,
    };
    match timeout(
        wait_bounds.request_timeout,
        daemon_client.send_request(&training_request),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => return Err(DecideError::DaemonStoppedResponding.into()),
    }
    let daemon_wait_bounds = DaemonWaitBounds {
        silence_bound: wait_bounds.request_timeout,
        load_stall_bound: wait_bounds.model_load_stall_bound,
        work_poll_interval: wait_bounds.daemon_work_poll_interval,
    };
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let mut stage_progress: Option<StageProgress> = None;
    loop {
        let next_frame = await_frame_watching_daemon_work(
            daemon_client.next_response(),
            daemon_probe,
            daemon_wait_bounds,
            &mut daemon_work_watch,
            &mut *dependencies.stderr,
        )
        .await;
        let training_response = match next_frame {
            Ok(Ok(Some(training_response))) => training_response,
            Ok(Ok(None) | Err(_)) | Err(_) => {
                return Err(DecideError::DaemonStoppedResponding.into());
            }
        };
        if let DaemonResponse::DecisionTrainingProgress {
            stage,
            completed,
            total,
        } = training_response
        {
            report_stage_progress(
                &mut stage_progress,
                StageReport {
                    stage,
                    completed,
                    total,
                },
                dependencies.progress_interval,
                &mut *dependencies.stderr,
            );
            continue;
        }
        return training_outcome(training_response);
    }
}

/// Prints one progress report, starting a fresh line and clock whenever the
/// worker moves on to the next stage.
fn report_stage_progress(
    stage_progress: &mut Option<StageProgress>,
    stage_report: StageReport,
    progress_interval: Duration,
    stderr: &mut dyn Write,
) {
    let StageReport {
        stage,
        completed,
        total,
    } = stage_report;
    let is_same_stage =
        matches!(stage_progress, Some(current_stage) if current_stage.stage == stage);
    if !is_same_stage {
        let (stage_name, unit_name) = stage_wording(stage);
        *stage_progress = Some(StageProgress {
            stage,
            timed_progress: TimedProgress::new(stage_name, unit_name, progress_interval),
        });
    }
    if let Some(current_stage) = stage_progress {
        current_stage
            .timed_progress
            .report(stderr, completed, total);
    }
}

/// How each stage's progress line reads. A training's total is the most
/// steps every run could take; runs that stop improving early jump the
/// count forward, so the time left shrinks rather than grows.
fn stage_wording(stage: DecisionTrainingStage) -> (&'static str, &'static str) {
    match stage {
        DecisionTrainingStage::Reading => ("Reading examples", "examples"),
        DecisionTrainingStage::Checking => ("Checking the starting point", "questions"),
        DecisionTrainingStage::Training => ("Training", "steps"),
        DecisionTrainingStage::Scoring => ("Scoring the trained model", "questions"),
    }
}

/// The outcome the one terminal frame carries, or why there is none.
fn training_outcome(
    terminal_response: DaemonResponse,
) -> Result<DecisionTrainingOutcome, DecisionToolError> {
    match terminal_response {
        DaemonResponse::DecisionModelTrained { outcome, .. } => Ok(*outcome),
        DaemonResponse::DecisionsFailed { reason } => {
            Err(DecisionToolError::TrainingFailed(reason.to_string()))
        }
        DaemonResponse::GenerationRejected { reason } => {
            Err(DecideError::DecisionsRejected { reason }.into())
        }
        // No other frame can end a training; treat the daemon as gone
        // rather than guessing.
        _ => Err(DecideError::DaemonStoppedResponding.into()),
    }
}

#[cfg(test)]
#[path = "training_request_tests.rs"]
mod tests;
