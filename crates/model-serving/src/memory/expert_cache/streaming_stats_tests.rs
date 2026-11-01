//! Unit contract for session expert-streaming counters and derived views.

use super::{
    ExpertStreamingSessionCounters, RouteCoverageClass, SessionStreamingPhase,
    expert_layer_topology_from_residency,
};
use astronomical_memory_policy::residency::{CurrentExpertLayerResidency, RetainedExpertPageClass};

fn one_layer_residency(
    class: RetainedExpertPageClass,
    retained_expert_ids: Vec<usize>,
) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index: 7,
        class,
        retained_expert_ids,
        payload_bytes: 4096,
        covered_weighted_demand: 900,
    }
}

#[test]
fn should_record_route_coverage_across_all_three_classes_with_assignment_split() {
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_route_coverage(8, 0, RouteCoverageClass::FullyCovered);
    session_counters.record_route_coverage(3, 5, RouteCoverageClass::PartiallyCovered);
    session_counters.record_route_coverage(0, 8, RouteCoverageClass::Missed);
    assert_eq!(session_counters.retained_route_assignment_count, 11);
    assert_eq!(session_counters.storage_route_assignment_count, 13);
    assert_eq!(session_counters.hot_route_fully_covered_token_count, 1);
    assert_eq!(session_counters.hot_route_partially_covered_token_count, 1);
    assert_eq!(session_counters.hot_route_missed_token_count, 1);
}

#[test]
fn should_split_mandatory_source_bytes_by_phase_and_sum_avoided_bytes() {
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_mandatory_source_bytes(SessionStreamingPhase::Prefill, 2_000);
    session_counters.record_mandatory_source_bytes(SessionStreamingPhase::Decode, 300);
    session_counters.record_mandatory_source_bytes(SessionStreamingPhase::Decode, 50);
    session_counters.record_avoided_complete_layer_source_bytes(10_000);
    session_counters.record_avoided_complete_layer_source_bytes(1_000);
    assert_eq!(session_counters.mandatory_prefill_source_bytes, 2_000);
    assert_eq!(session_counters.mandatory_decode_source_bytes, 350);
    assert_eq!(session_counters.avoided_complete_layer_source_bytes, 11_000);
}

#[test]
fn should_saturate_every_counter_rather_than_wrap() {
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_avoided_complete_layer_source_bytes(u64::MAX);
    session_counters.record_avoided_complete_layer_source_bytes(1);
    assert_eq!(
        session_counters.avoided_complete_layer_source_bytes,
        u64::MAX
    );
}

#[test]
fn should_map_an_elastic_layer_residency_onto_the_observatory_topology() {
    let topology = expert_layer_topology_from_residency(&one_layer_residency(
        RetainedExpertPageClass::ElasticRoutedExperts,
        vec![3, 1, 2],
    ));
    assert_eq!(topology.layer_index, 7);
    assert_eq!(topology.retained_expert_ids, vec![3, 1, 2]);
    assert!(!topology.is_complete_layer);
    assert_eq!(topology.retained_payload_bytes, 4096);
    assert_eq!(topology.covered_demand, 900);
}

#[test]
fn should_map_a_complete_layer_residency_as_complete() {
    let complete_ids: Vec<usize> = (0..256).collect();
    let topology = expert_layer_topology_from_residency(&one_layer_residency(
        RetainedExpertPageClass::StableCompleteLayer,
        complete_ids,
    ));
    assert!(topology.is_complete_layer);
    assert_eq!(topology.retained_expert_ids.len(), 256);
}
