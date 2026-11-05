use std::time::Duration;

use super::{StageDurations, summarize_stages};

fn stage_sample(base_milliseconds: u64) -> StageDurations {
    StageDurations {
        input_preparation: Duration::from_millis(base_milliseconds),
        surface_write: Duration::from_millis(base_milliseconds + 1),
        engines_after_write: Duration::from_millis(base_milliseconds + 2),
        merge: Duration::from_millis(base_milliseconds + 3),
        total: Duration::from_millis(base_milliseconds + 4),
    }
}

#[test]
fn should_summarize_each_stage_from_its_own_durations() {
    let summaries = summarize_stages(&[stage_sample(30), stage_sample(10), stage_sample(20)]);

    assert_eq!(
        summaries.input_preparation.median,
        Duration::from_millis(20)
    );
    assert_eq!(summaries.surface_write.median, Duration::from_millis(21));
    assert_eq!(
        summaries.engines_after_write.median,
        Duration::from_millis(22)
    );
    assert_eq!(summaries.merge.median, Duration::from_millis(23));
    assert_eq!(summaries.total.median, Duration::from_millis(24));
}

#[test]
fn should_summarize_a_single_sample_as_itself() {
    let summaries = summarize_stages(&[stage_sample(7)]);

    assert_eq!(summaries.total.minimum, Duration::from_millis(11));
    assert_eq!(summaries.total.maximum, Duration::from_millis(11));
}

#[test]
#[should_panic(expected = "timed")]
fn should_refuse_to_summarize_no_samples() {
    summarize_stages(&[]);
}
