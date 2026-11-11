use super::{EndOfSpeechCountdown, frame_budget};

/// Frames kept before stopping, given the frame at which the model signals.
fn frames_kept(frames_after_end: usize, signal_frame: usize, frame_limit: usize) -> usize {
    let mut countdown = EndOfSpeechCountdown::new(frames_after_end);
    for frame_index in 0..frame_limit {
        if countdown.should_stop(frame_index == signal_frame) {
            return frame_index + 1;
        }
    }
    frame_limit
}

#[test]
fn should_keep_the_signalling_frame_and_one_more() {
    assert_eq!(frames_kept(1, 10, 100), 12);
}

// Boundaries: no closing frames, and the short-text three.
#[test]
fn should_keep_zero_or_three_closing_frames_as_asked() {
    assert_eq!(frames_kept(0, 10, 100), 11);
    assert_eq!(frames_kept(3, 10, 100), 14);
}

// Negative: without a signal only the budget stops generation.
#[test]
fn should_run_to_the_budget_without_a_signal() {
    let mut countdown = EndOfSpeechCountdown::new(1);
    assert!((0..50).all(|_| !countdown.should_stop(false)));
    assert!(!countdown.has_seen_end_of_speech());
}

#[test]
fn should_budget_three_tokens_a_second_plus_two_seconds() {
    assert_eq!(frame_budget(0, 12.5), 25);
    assert_eq!(frame_budget(30, 12.5), 150);
    assert_eq!(frame_budget(31, 12.5), 155);
}
