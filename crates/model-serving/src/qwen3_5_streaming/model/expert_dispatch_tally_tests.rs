use std::time::Duration;

use super::{ExpertDispatchCounts, ExpertDispatchPath, ExpertDispatchTally};

#[test]
fn should_count_each_layer_under_the_way_it_was_served() {
    let tally = ExpertDispatchTally::default();
    tally.record_path(ExpertDispatchPath::TableWithoutHostRoute);
    tally.record_path(ExpertDispatchPath::TableWithoutHostRoute);
    tally.record_path(ExpertDispatchPath::HostRouteTableMiss);
    tally.record_path(ExpertDispatchPath::HostRouteTableHit);
    tally.record_path(ExpertDispatchPath::CompletePageInHand);
    tally.record_path(ExpertDispatchPath::Unhandled);
    let counts = tally.take();
    assert_eq!(counts.table_without_host_route_layer_count, 2);
    assert_eq!(counts.host_route_table_miss_layer_count, 1);
    assert_eq!(counts.host_route_table_hit_layer_count, 1);
    assert_eq!(counts.complete_page_in_hand_layer_count, 1);
    assert_eq!(counts.unhandled_layer_count, 1);
    assert_eq!(counts.host_route_layer_count(), 3);
}

#[test]
fn should_add_up_the_time_spent_waiting_for_routes() {
    let tally = ExpertDispatchTally::default();
    tally.record_host_route_wait(Duration::from_micros(250));
    tally.record_host_route_wait(Duration::from_micros(750));
    assert_eq!(tally.take().host_route_wait_nanoseconds, 1_000_000);
}

#[test]
fn should_start_from_zero_after_the_counts_are_taken() {
    let tally = ExpertDispatchTally::default();
    tally.record_path(ExpertDispatchPath::HostRouteTableMiss);
    tally.record_host_route_wait(Duration::from_millis(1));
    let _ = tally.take();
    assert_eq!(tally.take(), ExpertDispatchCounts::default());
}

#[test]
fn should_report_nothing_when_no_layer_ran() {
    let counts = ExpertDispatchTally::default().take();
    assert_eq!(counts, ExpertDispatchCounts::default());
    assert_eq!(counts.host_route_layer_count(), 0);
}

#[test]
fn should_saturate_instead_of_wrapping_on_an_impossible_wait() {
    let tally = ExpertDispatchTally::default();
    tally.record_host_route_wait(Duration::MAX);
    tally.record_host_route_wait(Duration::MAX);
    assert_eq!(tally.take().host_route_wait_nanoseconds, u64::MAX);
}
