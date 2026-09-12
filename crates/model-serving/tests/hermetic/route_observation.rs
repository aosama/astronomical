//! Hermetic contracts for the decode route-observation history (#536).

use astronomical_model_serving::{
    ObservedExpertRoute, RouteObservationRecord, RouteObservationRing,
    sorted_unique_layer_routed_expert_ids,
};

fn observation(input_token_id: u32, route_marker: u16) -> RouteObservationRecord {
    RouteObservationRecord {
        input_token_id,
        previous_token_route: None,
        token_route: vec![Some(vec![route_marker])],
    }
}

#[test]
fn ring_retains_observations_in_arrival_order() {
    let mut ring = RouteObservationRing::new(4);
    for token_id in 0..3_u32 {
        ring.record_observation(observation(token_id, u16::try_from(token_id).unwrap()));
    }
    let retained_token_ids: Vec<u32> = ring.observations().map(|o| o.input_token_id).collect();
    assert_eq!(retained_token_ids, vec![0, 1, 2]);
    assert_eq!(ring.observation_count(), 3);
    assert_eq!(ring.stored_observation_count(), 3);
    assert_eq!(ring.evicted_observation_count(), 0);
}

#[test]
fn ring_overwrites_oldest_observation_when_full() {
    let mut ring = RouteObservationRing::new(3);
    for token_id in 0..7_u32 {
        ring.record_observation(observation(token_id, u16::try_from(token_id).unwrap()));
    }
    let retained_token_ids: Vec<u32> = ring.observations().map(|o| o.input_token_id).collect();
    assert_eq!(retained_token_ids, vec![4, 5, 6]);
    assert_eq!(ring.observation_count(), 3);
    assert_eq!(ring.stored_observation_count(), 7);
    assert_eq!(ring.evicted_observation_count(), 4);
}

#[test]
fn ring_preserves_complete_record_contents() {
    let previous_route: ObservedExpertRoute = vec![Some(vec![3, 9]), None, Some(vec![41])];
    let token_route: ObservedExpertRoute = vec![Some(vec![1, 2, 3]), None, Some(vec![7])];
    let record = RouteObservationRecord {
        input_token_id: 12_345,
        previous_token_route: Some(previous_route.clone()),
        token_route: token_route.clone(),
    };
    let mut ring = RouteObservationRing::new(2);
    ring.record_observation(observation(1, 1));
    ring.record_observation(record.clone());
    let retained: Vec<&RouteObservationRecord> = ring.observations().collect();
    assert_eq!(retained.len(), 2);
    assert_eq!(retained[1], &record);
    assert_eq!(retained[1].previous_token_route, Some(previous_route));
    assert_eq!(retained[1].token_route, token_route);
}

#[test]
fn single_observation_capacity_still_admits_every_newest_observation() {
    let mut ring = RouteObservationRing::new(1);
    for token_id in 0..5_u32 {
        ring.record_observation(observation(token_id, u16::try_from(token_id).unwrap()));
    }
    let retained_token_ids: Vec<u32> = ring.observations().map(|o| o.input_token_id).collect();
    assert_eq!(retained_token_ids, vec![4]);
    assert_eq!(ring.evicted_observation_count(), 4);
}

#[test]
fn zero_capacity_is_clamped_to_one_so_capture_never_stalls() {
    let mut ring = RouteObservationRing::new(0);
    ring.record_observation(observation(9, 9));
    assert_eq!(ring.capacity(), 1);
    assert_eq!(ring.observation_count(), 1);
}

#[test]
fn compaction_sorts_and_deduplicates_one_layer_route() {
    let compacted = sorted_unique_layer_routed_expert_ids(&[40, 8, 40, 15, 8]);
    assert_eq!(compacted, Some(vec![8_u16, 15, 40]));
}

#[test]
fn compaction_rejects_identifiers_beyond_the_expert_population() {
    let compacted = sorted_unique_layer_routed_expert_ids(&[8, u32::from(u16::MAX) + 1]);
    assert_eq!(compacted, None);
}

#[test]
fn compaction_of_an_empty_route_is_an_empty_layer_selection() {
    let compacted = sorted_unique_layer_routed_expert_ids(&[]);
    assert_eq!(compacted, Some(Vec::new()));
}
