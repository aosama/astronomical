use std::time::Duration;

use super::{DirectExpertFetchCounts, DirectExpertFetchOutcome, DirectExpertFetchTally};

#[test]
fn should_count_each_fetching_layer_under_what_happened_to_it() {
    let tally = DirectExpertFetchTally::default();
    tally.record_outcome(DirectExpertFetchOutcome::FetchedIntoTable);
    tally.record_outcome(DirectExpertFetchOutcome::FetchedIntoTable);
    tally.record_outcome(DirectExpertFetchOutcome::LayerKeepsNothing);
    tally.record_outcome(DirectExpertFetchOutcome::TableCouldNotTakeThem);
    tally.record_outcome(DirectExpertFetchOutcome::NoDirectReader);
    let counts = tally.take();
    assert_eq!(counts.fetched_into_table_layer_count, 2);
    assert_eq!(counts.layer_keeps_nothing_count, 1);
    assert_eq!(counts.table_could_not_take_them_count, 1);
    assert_eq!(counts.no_direct_reader_count, 1);
}

#[test]
fn should_add_up_the_experts_fetched_and_the_time_they_took() {
    let tally = DirectExpertFetchTally::default();
    tally.record_fetch(
        5,
        5_000,
        Duration::from_micros(800),
        Duration::from_micros(700),
    );
    tally.record_fetch(
        3,
        3_000,
        Duration::from_micros(200),
        Duration::from_micros(150),
    );
    let counts = tally.take();
    assert_eq!(counts.fetched_into_table_expert_count, 8);
    assert_eq!(counts.fetched_into_table_bytes, 8_000);
    assert_eq!(counts.direct_fetch_nanoseconds, 1_000_000);
    assert_eq!(counts.storage_read_nanoseconds, 850_000);
}

#[test]
fn should_start_from_zero_after_the_counts_are_taken() {
    let tally = DirectExpertFetchTally::default();
    tally.record_outcome(DirectExpertFetchOutcome::FetchedIntoTable);
    tally.record_fetch(8, 8_000, Duration::from_millis(1), Duration::from_millis(1));
    let _ = tally.take();
    assert_eq!(tally.take(), DirectExpertFetchCounts::default());
}

#[test]
fn should_saturate_instead_of_wrapping_on_an_impossible_fetch() {
    let tally = DirectExpertFetchTally::default();
    tally.record_fetch(usize::MAX, u64::MAX, Duration::MAX, Duration::MAX);
    tally.record_fetch(usize::MAX, u64::MAX, Duration::MAX, Duration::MAX);
    let counts = tally.take();
    assert_eq!(counts.fetched_into_table_expert_count, u64::MAX);
    assert_eq!(counts.fetched_into_table_bytes, u64::MAX);
    assert_eq!(counts.direct_fetch_nanoseconds, u64::MAX);
    assert_eq!(counts.storage_read_nanoseconds, u64::MAX);
}
