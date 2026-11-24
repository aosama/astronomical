use super::generation_outcome;
use crate::PerformanceAttributionOutcome;

// Positive: a reply that reached its end is recorded as a success.
#[test]
fn should_record_a_completed_reply_as_a_success() {
    assert!(matches!(
        generation_outcome(true),
        PerformanceAttributionOutcome::Success
    ));
}

// Negative: a reply stopped before its end is recorded as a cancellation,
// never as a success.
#[test]
fn should_record_a_stopped_reply_as_a_cancellation() {
    assert!(matches!(
        generation_outcome(false),
        PerformanceAttributionOutcome::Cancelled
    ));
}
