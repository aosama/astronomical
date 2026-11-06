//! `respond` while the daemon loads the model: it sends nothing on the generation's
//! connection until the load ends, and a load can outlast the silence bound.
//! A load report counts as hearing from the daemon; only silence, or a report that
//! stays put past the stall bound, means it has gone away.

use std::time::Duration;

use astronomical_cli::RespondError;
use astronomical_ipc_protocol::ModelLoadInFlight;

use super::daemon_wait_support::{
    run_respond_against, run_respond_with_silence_bound, run_respond_with_stall_bound,
    stub_config_with_first_frame_delay,
};
use crate::integration::support::environment::{
    FIRST_FRAME_DELAY_PAST_THE_BOUND, FIRST_FRAME_THAT_NEVER_ARRIVES, rising_progress_counts,
};
use crate::integration::support::stub_daemon::StubDaemonConfig;

/// What the wait draws while a model loads.
const LOAD_LINE: &str = "Loading the model";

fn stub_config_loading_model(
    chat_first_frame_delay: Duration,
    model_load_reports: Vec<Option<ModelLoadInFlight>>,
) -> StubDaemonConfig {
    StubDaemonConfig {
        model_load_reports,
        ..stub_config_with_first_frame_delay(chat_first_frame_delay)
    }
}

fn stuck_load_report() -> Option<ModelLoadInFlight> {
    Some(ModelLoadInFlight::for_model("test/local-chatter"))
}

/// The status requests that see the load: the one before the wait and the wait's first.
fn load_seen_by_two_status_requests_then_ended() -> Vec<Option<ModelLoadInFlight>> {
    vec![stuck_load_report(), stuck_load_report(), None]
}

/// A load stays reported past the silence bound while the daemon is healthy: the
/// false failure this wait exists to end.
#[tokio::test]
async fn should_keep_waiting_while_a_report_stays_put_inside_the_stall_bound() {
    let journey = run_respond_against(
        "patient",
        stub_config_loading_model(FIRST_FRAME_DELAY_PAST_THE_BOUND, vec![stuck_load_report()]),
    )
    .await;

    journey
        .respond_outcome
        .expect("a load still being reported must reach its answer");
    assert_eq!(journey.stdout, "Hello");
    assert!(
        journey.stderr.contains("Loading the model"),
        "the wait should say what it is waiting for: {:?}",
        journey.stderr
    );
    assert!(
        journey.stderr.ends_with('\n'),
        "the live line should end before the answer finishes: {:?}",
        journey.stderr
    );
}

/// The load ends and prefill runs on: the line must end with the load, not stay up
/// saying the model is loading.
#[tokio::test]
async fn should_end_the_load_line_when_the_load_ends_before_the_answer() {
    let journey = run_respond_against(
        "load-ends",
        StubDaemonConfig {
            model_load_reports: load_seen_by_two_status_requests_then_ended(),
            request_progress_counts: rising_progress_counts(),
            ..stub_config_with_first_frame_delay(FIRST_FRAME_DELAY_PAST_THE_BOUND)
        },
    )
    .await;

    journey
        .respond_outcome
        .expect("progress after the load must reach the answer");
    assert_eq!(journey.stdout, "Hello");
    assert_eq!(journey.stderr, format!("\r{LOAD_LINE}\n"));
}

/// A status request that goes unanswered says nothing about the load, so it must not
/// read as the load ending, which would break the line in two.
#[tokio::test]
async fn should_keep_the_load_line_when_a_status_request_goes_unanswered() {
    let journey = run_respond_with_silence_bound(
        "load-lost",
        StubDaemonConfig {
            model_load_reports: vec![stuck_load_report()],
            hung_status_requests: 2..3,
            ..stub_config_with_first_frame_delay(FIRST_FRAME_DELAY_PAST_THE_BOUND)
        },
        // Room for the answer after the lost status request gives up, on a loaded machine.
        Duration::from_secs(1),
    )
    .await;

    journey
        .respond_outcome
        .expect("a load still being reported must reach its answer");
    assert_eq!(journey.stderr, format!("\r{LOAD_LINE}\n"));
}

#[tokio::test]
async fn should_give_up_when_a_report_stays_put_past_the_stall_bound() {
    let journey = run_respond_with_stall_bound(
        "stuck",
        stub_config_loading_model(FIRST_FRAME_THAT_NEVER_ARRIVES, vec![stuck_load_report()]),
        Duration::from_millis(300),
    )
    .await;

    assert!(
        matches!(
            journey.respond_outcome,
            Err(RespondError::DaemonStoppedResponding)
        ),
        "a daemon that repeats one report forever has wedged: {:?}",
        journey.respond_outcome
    );
    assert!(journey.stderr.ends_with('\n'), "{:?}", journey.stderr);
}

#[tokio::test]
async fn should_give_up_on_silence_when_nothing_reports_a_load() {
    let journey = run_respond_against(
        "silent",
        stub_config_with_first_frame_delay(FIRST_FRAME_THAT_NEVER_ARRIVES),
    )
    .await;

    assert!(
        matches!(
            journey.respond_outcome,
            Err(RespondError::DaemonStoppedResponding)
        ),
        "with no load reported, the bound must apply as before: {:?}",
        journey.respond_outcome
    );
    assert!(
        journey.stderr.is_empty(),
        "nothing was loading, so nothing should claim to be: {:?}",
        journey.stderr
    );
}

#[tokio::test]
async fn should_say_nothing_about_loading_for_an_answer_with_no_load() {
    let journey =
        run_respond_against("fast", stub_config_with_first_frame_delay(Duration::ZERO)).await;

    journey
        .respond_outcome
        .expect("an answer with no load should complete");
    assert_eq!(journey.stdout, "Hello");
    assert!(journey.stderr.is_empty(), "{:?}", journey.stderr);
}
