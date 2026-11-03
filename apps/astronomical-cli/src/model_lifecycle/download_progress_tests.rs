//! Unit tests for the live download line and the stall watch. Every test
//! passes the elapsed time in, so none of them sleeps.

use std::time::Duration;

use astronomical_ipc_protocol::DaemonDownloadJob;

use super::{
    DownloadProgressTracker, StallVerdict, StallWatch, format_rate, format_time_remaining,
};

const GIGABYTE: u64 = 1_073_741_824;
const MEGABYTE: u64 = 1_048_576;

fn job(state: &str, bytes_completed: u64, bytes_total: u64) -> DaemonDownloadJob {
    DaemonDownloadJob {
        huggingface_id: "black-forest-labs/FLUX.2-klein-4B".to_owned(),
        state: state.to_owned(),
        bytes_completed,
        bytes_total,
        error: None,
    }
}

fn seconds(whole_seconds: u64) -> Duration {
    Duration::from_secs(whole_seconds)
}

#[test]
fn should_report_no_rate_until_two_polls_have_moved_bytes() {
    let mut tracker = DownloadProgressTracker::new();
    assert_eq!(tracker.bytes_per_second(), None);

    tracker.record(seconds(2), GIGABYTE);
    assert_eq!(tracker.bytes_per_second(), None, "one sample has no rate");

    tracker.record(seconds(4), GIGABYTE);
    assert_eq!(
        tracker.bytes_per_second(),
        None,
        "no bytes moved, so no rate"
    );
}

#[test]
fn should_compute_the_rate_from_the_bytes_moved_between_polls() {
    let mut tracker = DownloadProgressTracker::new();
    tracker.record(seconds(2), 0);
    tracker.record(seconds(4), 100 * MEGABYTE);

    assert_eq!(tracker.bytes_per_second(), Some(50.0 * MEGABYTE as f64));
}

#[test]
fn should_follow_the_recent_window_and_forget_old_speed() {
    let mut tracker = DownloadProgressTracker::new();
    // 100 MB/s for the first 20 s, then a connection that drops to 10 MB/s.
    tracker.record(seconds(0), 0);
    tracker.record(seconds(20), 2_000 * MEGABYTE);
    for second in (22..=60).step_by(2) {
        let moved_since_20 = (second - 20) * 10 * MEGABYTE;
        tracker.record(seconds(second), 2_000 * MEGABYTE + moved_since_20);
    }

    let rate = tracker.bytes_per_second().expect("bytes are flowing");

    let megabytes_per_second = rate / MEGABYTE as f64;
    assert!(
        (megabytes_per_second - 10.0).abs() < 0.5,
        "{megabytes_per_second}"
    );
}

#[test]
fn should_start_the_window_over_when_the_byte_count_moves_backwards() {
    let mut tracker = DownloadProgressTracker::new();
    tracker.record(seconds(2), 5 * GIGABYTE);
    tracker.record(seconds(4), 6 * GIGABYTE);
    // The daemon restarted and reconciled against the staged files on disk.
    tracker.record(seconds(6), 4 * GIGABYTE);
    assert_eq!(tracker.bytes_per_second(), None, "no rate across the jump");

    tracker.record(seconds(8), 4 * GIGABYTE + 100 * MEGABYTE);

    assert_eq!(tracker.bytes_per_second(), Some(50.0 * MEGABYTE as f64));
}

#[test]
fn should_render_rate_and_time_remaining_while_bytes_are_flowing() {
    let mut tracker = DownloadProgressTracker::new();
    tracker.record(seconds(2), 0);
    tracker.record(seconds(4), 100 * MEGABYTE);

    let line = tracker.render_line(&job("downloading", 100 * MEGABYTE, 1_100 * MEGABYTE));

    assert!(line.contains("downloading"), "{line}");
    assert!(line.contains("(9%)"), "{line}");
    assert!(line.contains("50.0 MB/s"), "{line}");
    assert!(line.contains("about 20s left"), "{line}");
}

#[test]
fn should_keep_the_base_text_other_verbs_and_tests_rely_on() {
    let mut tracker = DownloadProgressTracker::new();

    let line = tracker.render_line(&job("downloading", GIGABYTE, 2 * GIGABYTE));

    assert!(line.contains("1 GB / 2 GB (50%)"), "{line}");
}

#[test]
fn should_show_no_rate_or_time_outside_the_downloading_state() {
    let mut tracker = DownloadProgressTracker::new();
    tracker.record(seconds(2), 0);
    tracker.record(seconds(4), 100 * MEGABYTE);

    let line = tracker.render_line(&job("verifying", 100 * MEGABYTE, 100 * MEGABYTE));

    assert!(!line.contains("MB/s") && !line.contains("left"), "{line}");
}

#[test]
fn should_pad_a_shorter_line_so_it_overwrites_the_previous_one() {
    let mut tracker = DownloadProgressTracker::new();
    tracker.record(seconds(2), 0);
    tracker.record(seconds(4), 100 * MEGABYTE);
    let long_line = tracker.render_line(&job("downloading", 100 * MEGABYTE, 1_100 * MEGABYTE));

    let short_line = tracker.render_line(&job("verifying", 1_100 * MEGABYTE, 1_100 * MEGABYTE));

    assert_eq!(short_line.chars().count(), long_line.chars().count());
    assert!(short_line.ends_with(' '));
}

#[test]
fn should_format_rates_in_megabytes_and_gigabytes_a_second() {
    assert_eq!(format_rate(0.0), "0.0 MB/s");
    assert_eq!(format_rate(5.5 * MEGABYTE as f64), "5.5 MB/s");
    assert_eq!(format_rate(999.0 * MEGABYTE as f64), "999.0 MB/s");
    assert_eq!(format_rate(2_048.0 * MEGABYTE as f64), "2.0 GB/s");
}

#[test]
fn should_format_time_remaining_at_every_scale() {
    assert_eq!(format_time_remaining(0.0), "about 0s left");
    assert_eq!(format_time_remaining(0.2), "about 1s left");
    assert_eq!(format_time_remaining(59.0), "about 59s left");
    assert_eq!(format_time_remaining(60.0), "about 1m 00s left");
    assert_eq!(format_time_remaining(220.0), "about 3m 40s left");
    assert_eq!(format_time_remaining(3_599.0), "about 59m 59s left");
    assert_eq!(format_time_remaining(3_600.0), "about 1h 00m left");
    assert_eq!(format_time_remaining(7_500.0), "about 2h 05m left");
    assert_eq!(format_time_remaining(86_400.0), "over a day left");
}

#[test]
fn should_not_invent_a_time_for_a_nonsense_estimate() {
    assert_eq!(format_time_remaining(f64::INFINITY), "time left unknown");
    assert_eq!(format_time_remaining(f64::NAN), "time left unknown");
    assert_eq!(format_time_remaining(-5.0), "time left unknown");
}

#[test]
fn should_treat_a_changing_byte_count_as_progress_however_long_it_takes() {
    let mut stall_watch = StallWatch::new(seconds(90));

    for poll in 0..200_u64 {
        let verdict = stall_watch.observe(seconds(poll * 10), "downloading", poll * MEGABYTE);

        assert_eq!(verdict, StallVerdict::Progressing, "poll {poll}");
    }
}

#[test]
fn should_stall_once_nothing_changes_for_the_bound() {
    let mut stall_watch = StallWatch::new(seconds(90));
    assert_eq!(
        stall_watch.observe(seconds(10), "downloading", 5),
        StallVerdict::Progressing
    );

    assert_eq!(
        stall_watch.observe(seconds(99), "downloading", 5),
        StallVerdict::Progressing
    );
    let verdict = stall_watch.observe(seconds(100), "downloading", 5);

    assert_eq!(
        verdict,
        StallVerdict::Stalled {
            stalled_for: seconds(90),
            state: "downloading".to_owned()
        }
    );
}

#[test]
fn should_restart_the_clock_when_bytes_move_again() {
    let mut stall_watch = StallWatch::new(seconds(90));
    stall_watch.observe(seconds(0), "downloading", 5);
    stall_watch.observe(seconds(80), "downloading", 5);

    stall_watch.observe(seconds(85), "downloading", 6);

    assert_eq!(
        stall_watch.observe(seconds(170), "downloading", 6),
        StallVerdict::Progressing
    );
    assert!(matches!(
        stall_watch.observe(seconds(175), "downloading", 6),
        StallVerdict::Stalled { .. }
    ));
}

#[test]
fn should_treat_a_state_change_as_progress_even_with_the_same_bytes() {
    let mut stall_watch = StallWatch::new(seconds(90));
    stall_watch.observe(seconds(0), "downloading", 100);
    stall_watch.observe(seconds(80), "downloading", 100);

    let verdict = stall_watch.observe(seconds(85), "verifying", 100);

    assert_eq!(verdict, StallVerdict::Progressing);
}

#[test]
fn should_give_verifying_and_publishing_ten_times_the_patience() {
    for finalizing_state in ["verifying", "publishing"] {
        let mut stall_watch = StallWatch::new(seconds(90));
        stall_watch.observe(seconds(0), finalizing_state, 100);

        assert_eq!(
            stall_watch.observe(seconds(899), finalizing_state, 100),
            StallVerdict::Progressing,
            "{finalizing_state}"
        );
        assert!(
            matches!(
                stall_watch.observe(seconds(900), finalizing_state, 100),
                StallVerdict::Stalled { .. }
            ),
            "{finalizing_state}"
        );
    }
}

#[test]
fn should_stall_a_paused_job_at_the_ordinary_bound() {
    let mut stall_watch = StallWatch::new(seconds(90));
    stall_watch.observe(seconds(0), "paused", 100);

    let verdict = stall_watch.observe(seconds(90), "paused", 100);

    assert_eq!(
        verdict,
        StallVerdict::Stalled {
            stalled_for: seconds(90),
            state: "paused".to_owned()
        }
    );
}
