use memory_policy::astro_expert_cache_model::CapacityMoment;

use super::{MAXIMUM_CAPACITY_MOMENTS, MAXIMUM_RECORDED_MISS_TOKENS, RouteObservationCollector};
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

#[test]
fn should_keep_each_token_s_misses_by_layer_while_a_reply_is_recorded() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    collector.note_layer_misses(2, 3);
    collector.note_layer_misses(2, 1);
    collector.note_layer_misses(0, 5);
    collector.finish_token_misses();
    collector.finish_token_misses();

    assert_eq!(
        collector.take_engine_miss_counts(2),
        vec![vec![5, 0, 4], Vec::new()]
    );
}

#[test]
fn should_keep_no_misses_while_no_reply_is_recorded() {
    let mut collector = RouteObservationCollector::new();
    collector.note_layer_misses(1, 3);
    collector.finish_token_misses();

    assert!(collector.take_engine_miss_counts(1).is_empty());
}

#[test]
fn should_keep_only_the_newest_tokens_misses_when_fewer_routes_are_kept() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    for miss_count in 1..=3 {
        collector.note_layer_misses(0, miss_count);
        collector.finish_token_misses();
    }

    assert_eq!(collector.take_engine_miss_counts(2), vec![vec![2], vec![3]]);
}

#[test]
fn should_keep_no_more_tokens_misses_than_the_routes_ring_holds() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    for _ in 0..=MAXIMUM_RECORDED_MISS_TOKENS {
        collector.note_layer_misses(0, 1);
        collector.finish_token_misses();
    }

    assert_eq!(
        collector.take_engine_miss_counts(usize::MAX).len(),
        MAXIMUM_RECORDED_MISS_TOKENS
    );
}

#[test]
fn should_start_each_recorded_reply_with_no_misses() {
    let mut collector = RouteObservationCollector::new();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(0));
    collector.note_layer_misses(0, 4);
    collector.finish_token_misses();
    collector.set_reply_start_snapshot(ReplyStartSnapshot::empty_at(1));

    assert!(collector.take_engine_miss_counts(usize::MAX).is_empty());
}
