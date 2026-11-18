use std::time::Duration;

use super::DraftRoundTally;

#[test]
fn should_average_tokens_and_time_over_the_rounds_recorded() {
    let mut tally = DraftRoundTally::default();
    tally.record_round(3, Duration::from_millis(30));
    tally.record_round(1, Duration::from_millis(20));
    assert_eq!(tally.round_count, 2);
    assert_eq!(tally.emitted_token_count, 4);
    assert_eq!(tally.tokens_per_round(), Some(2.0));
    assert_eq!(tally.milliseconds_per_round(), Some(25.0));
    assert_eq!(tally.tokens_per_second(), Some(80.0));
}

#[test]
fn should_report_no_average_before_the_first_round() {
    let tally = DraftRoundTally::default();
    assert_eq!(tally.tokens_per_round(), None);
    assert_eq!(tally.milliseconds_per_round(), None);
    assert_eq!(tally.tokens_per_second(), None);
}

#[test]
fn should_report_no_rate_for_rounds_that_took_no_measurable_time() {
    let mut tally = DraftRoundTally::default();
    tally.record_round(2, Duration::ZERO);
    assert_eq!(tally.tokens_per_round(), Some(2.0));
    assert_eq!(tally.tokens_per_second(), None);
}

#[test]
fn should_saturate_instead_of_wrapping_on_impossible_totals() {
    let mut tally = DraftRoundTally::default();
    tally.record_round(usize::MAX, Duration::MAX);
    tally.record_round(usize::MAX, Duration::MAX);
    assert_eq!(tally.emitted_token_count, u64::MAX);
    assert_eq!(tally.round_nanoseconds, u64::MAX);
}
