//! Unit tests for following a head training's frames.

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::head_training_job::{RunEnding, RunSummary};

use super::*;

fn report(
    stage_progress: &mut Option<StageProgress>,
    stage: HeadTrainingStage,
    completed: u64,
    total: u64,
    progress_text: &mut Vec<u8>,
) {
    report_stage_progress(
        stage_progress,
        StageReport {
            stage,
            completed,
            total,
        },
        Duration::ZERO,
        progress_text,
    );
}

#[test]
fn should_start_a_fresh_line_for_each_stage_and_finish_when_runs_stop_early() {
    let mut progress_text = Vec::new();
    let mut stage_progress = None;
    report(
        &mut stage_progress,
        HeadTrainingStage::Reading,
        1,
        24,
        &mut progress_text,
    );
    report(
        &mut stage_progress,
        HeadTrainingStage::Reading,
        24,
        24,
        &mut progress_text,
    );
    report(
        &mut stage_progress,
        HeadTrainingStage::Training,
        1,
        120,
        &mut progress_text,
    );
    // A run that stops improving jumps the count to the end of its passes.
    report(
        &mut stage_progress,
        HeadTrainingStage::Training,
        120,
        120,
        &mut progress_text,
    );
    let progress_lines = String::from_utf8(progress_text).unwrap();
    assert!(
        progress_lines.contains("Reading examples: 24 of 24 examples (100%), done"),
        "{progress_lines}"
    );
    assert!(
        progress_lines.contains("Training: 1 of 120 passes"),
        "{progress_lines}"
    );
    assert!(
        progress_lines.ends_with("120 of 120 passes (100%), done\n"),
        "{progress_lines}"
    );
}

#[test]
fn should_return_the_outcome_the_terminal_frame_carries() {
    let outcome = HeadTrainingOutcome {
        examples: Vec::new(),
        trained_tensors: Vec::new(),
        runs: vec![RunSummary {
            seed: 1,
            best_choosing_loss: 0.4,
            best_epoch: 3,
            epochs_run: 9,
            ending: RunEnding::StoppedImproving,
        }],
        best_seed: 1,
        largest_parity_gap: 0.000_2,
        indistinguishable_question_count: 0,
    };
    let trained = training_outcome(DaemonResponse::DecisionHeadTrained {
        model: "kev".to_owned(),
        outcome: Box::new(outcome.clone()),
    });
    assert_eq!(trained.unwrap(), outcome);
}

#[test]
fn should_name_why_a_training_did_not_finish() {
    let failed = training_outcome(DaemonResponse::DecisionsFailed {
        reason: DecisionFailure::invalid_request("this model's final layers cannot be trained"),
    });
    assert!(matches!(
        failed,
        Err(DecisionToolError::TrainingFailed(reason)) if reason.contains("cannot be trained")
    ));
    let rejected = training_outcome(DaemonResponse::GenerationRejected {
        reason: "no decision model is installed".to_owned(),
    });
    assert!(matches!(rejected, Err(DecisionToolError::Decide(_))));
}
