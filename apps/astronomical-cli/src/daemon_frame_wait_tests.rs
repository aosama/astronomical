use std::time::Duration;

use astronomical_ipc_protocol::{DaemonWorkerStatus, ModelLoadInFlight};
use tokio::time::Instant;

use super::DaemonWorkWatch;
use crate::{DaemonStatusSnapshot, formatting::MODEL_LOAD_LINE};

const LOAD_STALL_BOUND: Duration = Duration::from_secs(10);

fn load_report(model_id: &str) -> ModelLoadInFlight {
    ModelLoadInFlight::for_model(model_id)
}

fn status_reporting(
    model_load_in_flight: Option<ModelLoadInFlight>,
    request_progress_count: u64,
) -> DaemonStatusSnapshot {
    DaemonStatusSnapshot {
        worker_status: DaemonWorkerStatus::Ready,
        ready_model_id: None,
        default_model_id: None,
        model_load_in_flight,
        request_progress_count,
        neural_engine_prefill: None,
    }
}

/// Whether the load counts at `heard_at`, discarding what the watch writes to stderr.
fn counts_load(
    daemon_work_watch: &mut DaemonWorkWatch,
    model_load_in_flight: Option<&ModelLoadInFlight>,
    heard_at: Instant,
) -> bool {
    daemon_work_watch.note_load_report(
        model_load_in_flight,
        heard_at,
        LOAD_STALL_BOUND,
        &mut Vec::new(),
    )
}

#[test]
fn should_count_a_load_the_first_time_it_is_reported() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let first_load = load_report("first-model");

    assert!(counts_load(
        &mut daemon_work_watch,
        Some(&first_load),
        Instant::now()
    ));
}

#[test]
fn should_not_count_a_report_that_stays_put_past_the_stall_bound() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let first_load = load_report("first-model");
    let first_heard_at = Instant::now();
    counts_load(&mut daemon_work_watch, Some(&first_load), first_heard_at);

    assert!(counts_load(
        &mut daemon_work_watch,
        Some(&first_load),
        first_heard_at + LOAD_STALL_BOUND - Duration::from_millis(1)
    ));
    assert!(!counts_load(
        &mut daemon_work_watch,
        Some(&first_load),
        first_heard_at + LOAD_STALL_BOUND
    ));
}

#[test]
fn should_restart_the_stall_bound_for_another_model() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let first_heard_at = Instant::now();
    counts_load(
        &mut daemon_work_watch,
        Some(&load_report("first-model")),
        first_heard_at,
    );

    assert!(counts_load(
        &mut daemon_work_watch,
        Some(&load_report("second-model")),
        first_heard_at + LOAD_STALL_BOUND * 2
    ));
}

#[test]
fn should_treat_a_load_after_a_cleared_report_as_new() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let repeated_load = load_report("first-model");
    let first_heard_at = Instant::now();
    counts_load(&mut daemon_work_watch, Some(&repeated_load), first_heard_at);
    counts_load(&mut daemon_work_watch, None, first_heard_at);

    assert!(counts_load(
        &mut daemon_work_watch,
        Some(&repeated_load),
        first_heard_at + LOAD_STALL_BOUND * 2
    ));
}

#[test]
fn should_draw_the_load_line_once_and_end_it_when_the_load_ends() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let first_load = load_report("first-model");
    let heard_at = Instant::now();
    let mut stderr = Vec::new();

    for reported_load in [Some(&first_load), Some(&first_load), None, None] {
        daemon_work_watch.note_load_report(reported_load, heard_at, LOAD_STALL_BOUND, &mut stderr);
    }

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        format!("\r{MODEL_LOAD_LINE}\n")
    );
}

#[test]
fn should_draw_a_new_line_for_a_load_that_clears_and_returns() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let repeated_load = load_report("first-model");
    let heard_at = Instant::now();
    let mut stderr = Vec::new();

    for reported_load in [Some(&repeated_load), None, Some(&repeated_load), None] {
        daemon_work_watch.note_load_report(reported_load, heard_at, LOAD_STALL_BOUND, &mut stderr);
    }

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        format!("\r{MODEL_LOAD_LINE}\n\r{MODEL_LOAD_LINE}\n")
    );
}

/// No poll saw the report clear, because the frame arrived inside one poll interval.
#[test]
fn should_treat_a_load_after_a_frame_as_new() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let repeated_load = load_report("first-model");
    let first_heard_at = Instant::now();
    let mut stderr = Vec::new();
    daemon_work_watch.note_load_report(
        Some(&repeated_load),
        first_heard_at,
        LOAD_STALL_BOUND,
        &mut stderr,
    );

    daemon_work_watch.forget_load_after_frame(&mut stderr);

    assert!(daemon_work_watch.note_load_report(
        Some(&repeated_load),
        first_heard_at + LOAD_STALL_BOUND * 2,
        LOAD_STALL_BOUND,
        &mut stderr,
    ));
    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        format!("\r{MODEL_LOAD_LINE}\n\r{MODEL_LOAD_LINE}")
    );
}

#[test]
fn should_hear_from_the_daemon_through_a_load_alone() {
    let mut daemon_work_watch = DaemonWorkWatch::default();

    assert!(daemon_work_watch.has_heard_from_daemon(
        &status_reporting(Some(load_report("first-model")), 41),
        Instant::now(),
        LOAD_STALL_BOUND,
        &mut Vec::new(),
    ));
}

#[test]
fn should_hear_from_the_daemon_through_progress_past_a_stalled_load() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let stalled_load = Some(load_report("first-model"));
    let first_heard_at = Instant::now();
    daemon_work_watch.has_heard_from_daemon(
        &status_reporting(stalled_load.clone(), 41),
        first_heard_at,
        LOAD_STALL_BOUND,
        &mut Vec::new(),
    );

    assert!(daemon_work_watch.has_heard_from_daemon(
        &status_reporting(stalled_load, 42),
        first_heard_at + LOAD_STALL_BOUND * 2,
        LOAD_STALL_BOUND,
        &mut Vec::new(),
    ));
}

#[test]
fn should_not_hear_from_the_daemon_when_nothing_has_changed() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let heard_at = Instant::now();

    for _status_poll in 0..2 {
        assert!(!daemon_work_watch.has_heard_from_daemon(
            &status_reporting(None, 41),
            heard_at,
            LOAD_STALL_BOUND,
            &mut Vec::new(),
        ));
    }
}

#[test]
fn should_pad_a_shorter_line_over_a_longer_one() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let mut stderr = Vec::new();

    daemon_work_watch.draw_live_line("twelve chars", &mut stderr);
    daemon_work_watch.draw_live_line("short", &mut stderr);

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        "\rtwelve chars\rshort       "
    );
}

#[test]
fn should_end_a_live_line_exactly_once() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let mut stderr = Vec::new();

    daemon_work_watch.draw_live_line("line", &mut stderr);
    daemon_work_watch.finish_live_line(&mut stderr);
    daemon_work_watch.finish_live_line(&mut stderr);

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        "\rline\n"
    );
}

#[test]
fn should_write_nothing_when_no_line_was_ever_drawn() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    let mut stderr = Vec::new();

    daemon_work_watch.finish_live_line(&mut stderr);

    assert!(
        stderr.is_empty(),
        "a fast answer must not gain a blank line"
    );
}

#[test]
fn should_not_treat_the_first_count_as_progress() {
    let mut daemon_work_watch = DaemonWorkWatch::default();

    assert!(!daemon_work_watch.reports_request_progress(41));
}

#[test]
fn should_treat_a_risen_count_as_progress() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    daemon_work_watch.reports_request_progress(41);

    assert!(daemon_work_watch.reports_request_progress(42));
}

#[test]
fn should_not_treat_an_unchanged_count_as_progress() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    daemon_work_watch.reports_request_progress(41);
    daemon_work_watch.reports_request_progress(42);

    assert!(!daemon_work_watch.reports_request_progress(42));
}

#[test]
fn should_treat_a_restarted_count_as_progress() {
    let mut daemon_work_watch = DaemonWorkWatch::default();
    daemon_work_watch.reports_request_progress(41);

    assert!(daemon_work_watch.reports_request_progress(0));
}
