//! `respond` while the daemon works on something other than the first token. It sends
//! nothing on the generation's connection during prefill or while the request waits
//! behind another, so a request making progress counts as hearing from the daemon. The
//! count does not say whose request moved, which is why these journeys stand for both
//! cases.

use astronomical_cli::RespondError;
use std::time::Duration;

use super::daemon_wait_support::{
    run_respond_against, run_respond_with_silence_bound, stub_config_with_first_frame_delay,
};
use crate::integration::support::environment::{
    DAEMON_WORK_POLL_INTERVAL, FIRST_FRAME_DELAY_PAST_THE_BOUND, FIRST_FRAME_THAT_NEVER_ARRIVES,
    SILENCE_BOUND, rising_progress_counts,
};
use crate::integration::support::stub_daemon::StubDaemonConfig;

/// How high the count stands has no meaning, so a high one must earn nothing.
const STUCK_COUNT: u64 = 400;
/// Enough polls that the last change lands well after the journey starts, so giving
/// up later than one silence bound proves the changes extended the wait.
const COUNTS_BEFORE_THE_WORKER_STICKS: u64 = 8;
/// The status request answered before the wait starts polling.
const PREFLIGHT_STATUS_REQUESTS: usize = 1;
/// Room for the answer after a lost status request gives up, on a loaded machine.
const SILENCE_BOUND_PAST_A_LOST_STATUS: Duration = Duration::from_secs(1);

fn stub_config_with_progress(
    first_frame_delay: Duration,
    request_progress_counts: Vec<u64>,
) -> StubDaemonConfig {
    StubDaemonConfig {
        request_progress_counts,
        ..stub_config_with_first_frame_delay(first_frame_delay)
    }
}

fn rising_counts_up_to(last_count: u64) -> Vec<u64> {
    let mut rising_counts = Vec::new();
    for scripted_count in 1..=last_count {
        rising_counts.push(scripted_count);
    }
    rising_counts
}

/// A prompt long enough that prefill outlasts the silence bound: the false failure
/// this wait exists to end.
#[tokio::test]
async fn should_keep_waiting_while_requests_make_progress() {
    let journey = run_respond_against(
        "rising",
        stub_config_with_progress(FIRST_FRAME_DELAY_PAST_THE_BOUND, rising_progress_counts()),
    )
    .await;

    journey
        .respond_outcome
        .expect("a request making progress must reach its answer");
    assert_eq!(journey.stdout, "Hello");
    assert!(
        !journey.stderr.contains("Loading the model"),
        "progress is not a load: {:?}",
        journey.stderr
    );
}

/// The worker wedged part way through prefill: the daemon still answers status, and
/// the report that it has stopped must still arrive, after the wait was extended.
#[tokio::test]
async fn should_give_up_when_progress_stops() {
    let journey = run_respond_against(
        "stops",
        stub_config_with_progress(
            FIRST_FRAME_THAT_NEVER_ARRIVES,
            rising_counts_up_to(COUNTS_BEFORE_THE_WORKER_STICKS),
        ),
    )
    .await;

    assert!(matches!(
        journey.respond_outcome,
        Err(RespondError::DaemonStoppedResponding)
    ));
    let earliest_extended_give_up = SILENCE_BOUND + DAEMON_WORK_POLL_INTERVAL * 4;
    assert!(
        journey.elapsed >= earliest_extended_give_up,
        "the changes should have extended the wait, gave up at {:?}",
        journey.elapsed
    );
}

/// A count that never moves says nothing about the worker, however high it is.
#[tokio::test]
async fn should_give_up_when_the_count_never_changes() {
    let journey = run_respond_against(
        "never",
        stub_config_with_progress(FIRST_FRAME_THAT_NEVER_ARRIVES, vec![STUCK_COUNT]),
    )
    .await;

    assert!(matches!(
        journey.respond_outcome,
        Err(RespondError::DaemonStoppedResponding)
    ));
}

/// A daemon that stops answering status must not stop the bound on silence from firing.
#[tokio::test]
async fn should_give_up_when_status_never_answers() {
    let journey = run_respond_against(
        "mute",
        StubDaemonConfig {
            hung_status_requests: PREFLIGHT_STATUS_REQUESTS..usize::MAX,
            ..stub_config_with_progress(FIRST_FRAME_THAT_NEVER_ARRIVES, rising_progress_counts())
        },
    )
    .await;

    assert!(matches!(
        journey.respond_outcome,
        Err(RespondError::DaemonStoppedResponding)
    ));
}

/// One lost status answer during healthy progress must not cost the wait its next
/// look at the count, or the false failure returns.
#[tokio::test]
async fn should_keep_waiting_when_one_status_request_goes_unanswered() {
    // The first poll sets the baseline and the second is the first change, so the
    // lost answer falls on a wait that has already been extended once.
    let poll_after_the_first_change = PREFLIGHT_STATUS_REQUESTS + 2;
    let journey = run_respond_with_silence_bound(
        "lost",
        StubDaemonConfig {
            hung_status_requests: poll_after_the_first_change..poll_after_the_first_change + 1,
            ..stub_config_with_progress(FIRST_FRAME_DELAY_PAST_THE_BOUND, rising_progress_counts())
        },
        SILENCE_BOUND_PAST_A_LOST_STATUS,
    )
    .await;

    journey
        .respond_outcome
        .expect("one lost status answer must not end a healthy wait");
    assert_eq!(journey.stdout, "Hello");
}
