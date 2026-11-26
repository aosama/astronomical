use memory_policy::astro_expert_cache_model::CapacityMoment;

use super::{MAXIMUM_CAPACITY_MOMENTS, RouteObservationCollector};
use crate::qwen3_5_text::model::routed_reply_recording::ReplyStartSnapshot;

fn moment_with_capacity(kept_expert_capacity_bytes: u64) -> CapacityMoment {
    CapacityMoment {
        kept_expert_capacity_bytes,
        ..CapacityMoment::default()
    }
}

#[test]
fn should_keep_no_capacity_moment_while_no_reply_is_recorded() {
    let mut collector = RouteObservationCollector::new();
    collector.note_capacity_moment(moment_with_capacity(7));
    assert!(collector.take_capacity_moments().is_empty());
}

#[test]
fn should_index_a_capacity_moment_from_the_recorded_replys_start() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    collector.note_capacity_moment(moment_with_capacity(7));
    let moments = collector.take_capacity_moments();
    assert_eq!(moments.len(), 1);
    assert_eq!(moments[0].reply_token_index, 0);
    assert_eq!(moments[0].kept_expert_capacity_bytes, 7);
}

#[test]
fn should_start_each_recorded_reply_with_no_moments() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    collector.note_capacity_moment(moment_with_capacity(7));
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    assert!(collector.take_capacity_moments().is_empty());
}

#[test]
fn should_keep_no_more_moments_than_the_bound() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    for _ in 0..=MAXIMUM_CAPACITY_MOMENTS {
        collector.note_capacity_moment(moment_with_capacity(7));
    }
    assert_eq!(
        collector.take_capacity_moments().len(),
        MAXIMUM_CAPACITY_MOMENTS
    );
}
