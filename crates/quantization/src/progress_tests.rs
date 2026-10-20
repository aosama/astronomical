//! Unit tests for the progress tracker's cadence and ETA arithmetic.

use super::*;

/// Records every snapshot so a test can assert on cadence and content.
#[derive(Default)]
struct RecordingProgressReporter {
    snapshots: Vec<StageProgress>,
}

impl ProgressReporter for RecordingProgressReporter {
    fn report(&mut self, progress: &StageProgress) {
        self.snapshots.push(progress.clone());
    }
}

#[test]
fn should_report_one_hundred_percent_when_there_is_no_work() {
    let progress = StageProgress {
        stage_name: "requantize",
        completed_work_units: 0,
        total_work_units: 0,
        elapsed: Duration::ZERO,
        estimate: EtaEstimate::NotYetAvailable,
        detail: String::new(),
    };

    assert_eq!(progress.percent_complete(), 100.0);
}

#[test]
fn should_report_immediately_on_the_first_unit_and_then_only_after_the_interval() {
    let mut recording_reporter = RecordingProgressReporter::default();
    let mut tracker = StageProgressTracker::new(
        "sensitivity sweep",
        10,
        0,
        Duration::from_secs(5),
        &mut recording_reporter,
    );

    tracker.advance("model.layers.0.mlp.up_proj");
    tracker.advance("model.layers.0.mlp.down_proj");

    // Two advances inside one interval produce exactly one report: the first.
    assert_eq!(recording_reporter.snapshots.len(), 1);
    assert_eq!(recording_reporter.snapshots[0].completed_work_units, 1);
}

#[test]
fn should_extrapolate_the_eta_from_units_completed_in_this_run_only() {
    // A resumed sweep with 90 of 100 modules already checkpointed must not
    // claim the remaining 10 are nearly free: the rate comes from this run.
    let mut recording_reporter = RecordingProgressReporter::default();
    let mut tracker = StageProgressTracker::new(
        "sensitivity sweep",
        100,
        90,
        Duration::from_secs(5),
        &mut recording_reporter,
    );

    // With nothing completed in this run yet, there is no rate to extrapolate.
    let estimate = tracker.estimate(tracker.started_at());
    assert_eq!(estimate, EtaEstimate::NotYetAvailable);

    // The first advance reports immediately and the completed count continues
    // from the checkpoint (90 -> 91), proving the checkpointed units are not
    // counted as this run's work (which would reset the rate to near-complete).
    tracker.advance("model.layers.90.mlp.up_proj");
    let first = recording_reporter
        .snapshots
        .first()
        .expect("the first advance reports immediately");
    assert_eq!(first.completed_work_units, 91);
    assert_eq!(first.percent_complete(), 91.0);
}

#[test]
fn should_always_emit_a_final_report_at_completion() {
    let mut recording_reporter = RecordingProgressReporter::default();
    let mut tracker = StageProgressTracker::new(
        "requantize",
        3,
        0,
        Duration::from_secs(3600),
        &mut recording_reporter,
    );

    tracker.advance("a");
    tracker.advance("b");
    tracker.advance("c");
    tracker.report_now("done");

    let final_snapshot = recording_reporter
        .snapshots
        .last()
        .expect("the completion report should be recorded");
    assert_eq!(final_snapshot.percent_complete(), 100.0);
}
