//! Unit contract for session expert-streaming counters and derived views.

use super::super::serving_phase::SessionStreamingPhase;
use super::{
    ExpertStreamingSessionCounters, RouteCoverageClass, expert_layer_topology_from_residency,
    no_cache_route_source_bytes, record_streamed_page_source_plan_in_both_ledgers,
};
use crate::performance_attribution::StreamedPageSourcePlan;
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

/// Component: the same sequence of prefill chunks and decode tokens flows
/// through both accounting paths — the opt-in per-request attribution and
/// the always-on observatory session counters — and the phase byte totals
/// must match exactly (issue #1715: one session's counters disagreed with
/// its attribution reports by tens of gigabytes of prefill).
/// Replays one streamed page through both ledgers so the component test
/// reads as a serving sequence instead of construction noise.
fn replay_streamed_page_in_both_ledgers(
    route_token_count: i32,
    serving_phase: SessionStreamingPhase,
    payload_byte_count: u64,
    performance_attribution: &mut crate::PerformanceAttribution,
    session_counters: &mut ExpertStreamingSessionCounters,
) {
    record_streamed_page_source_plan_in_both_ledgers(
        &StreamedPageSourcePlan {
            layer_index: 3,
            route_token_count,
            serving_phase,
            routed_expert_count: 8,
            streamed_expert_count: 8,
            source_shard_count: 2,
            payload_byte_count,
        },
        performance_attribution,
        session_counters,
    );
}

/// Component: the same sequence of prefill chunks and decode tokens flows
/// through both accounting paths — the opt-in per-request attribution and
/// the always-on observatory session counters — and the phase byte totals
/// must match exactly (issue #1715: one session's counters disagreed with
/// its attribution reports by tens of gigabytes of prefill).
#[test]
fn should_agree_between_session_counters_and_attribution_reports() {
    let mut performance_attribution = crate::PerformanceAttribution::enabled();
    let mut session_counters = ExpertStreamingSessionCounters::default();
    // Two prefill chunks, then a decode token, then a multi-token decode
    // route (speculative decode): the phase must come from the engine, not
    // from the route's token count, on both sides.
    replay_streamed_page_in_both_ledgers(
        512,
        SessionStreamingPhase::Prefill,
        1_000,
        &mut performance_attribution,
        &mut session_counters,
    );
    replay_streamed_page_in_both_ledgers(
        512,
        SessionStreamingPhase::Prefill,
        700,
        &mut performance_attribution,
        &mut session_counters,
    );
    replay_streamed_page_in_both_ledgers(
        1,
        SessionStreamingPhase::Decode,
        40,
        &mut performance_attribution,
        &mut session_counters,
    );
    replay_streamed_page_in_both_ledgers(
        4,
        SessionStreamingPhase::Decode,
        60,
        &mut performance_attribution,
        &mut session_counters,
    );
    let attributed_prefill_bytes = performance_attribution
        .counter_value(crate::PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes);
    let attributed_decode_bytes = performance_attribution
        .counter_value(crate::PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes);
    assert_eq!(attributed_prefill_bytes, 1_700);
    assert_eq!(attributed_decode_bytes, 100);
    assert_eq!(session_counters.mandatory_prefill_source_bytes, 1_700);
    assert_eq!(session_counters.mandatory_decode_source_bytes, 100);
    assert_eq!(
        session_counters.mandatory_prefill_source_bytes, attributed_prefill_bytes,
        "the session counters and the attribution reports must agree on prefill bytes"
    );
    assert_eq!(
        session_counters.mandatory_decode_source_bytes, attributed_decode_bytes,
        "the session counters and the attribution reports must agree on decode bytes"
    );
}

#[test]
fn should_record_route_coverage_across_all_three_classes_with_assignment_split() {
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_route_coverage(8, 0, RouteCoverageClass::FullyCovered);
    session_counters.record_route_coverage(3, 5, RouteCoverageClass::PartiallyCovered);
    session_counters.record_route_coverage(0, 8, RouteCoverageClass::Missed);
    assert_eq!(session_counters.retained_route_assignment_count, 11);
    assert_eq!(session_counters.storage_route_assignment_count, 13);
    assert_eq!(
        session_counters.hot_route_fully_covered_layer_token_count,
        1
    );
    assert_eq!(
        session_counters.hot_route_partially_covered_layer_token_count,
        1
    );
    assert_eq!(session_counters.hot_route_missed_layer_token_count, 1);
}

#[test]
fn should_count_decode_classifications_as_layer_tokens_not_tokens() {
    // One decode token crossing two paged layers classifies twice: once per
    // layer-token. The counters must expose that unit honestly instead of
    // pretending the page counts tokens (issue #1715).
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_route_coverage(8, 0, RouteCoverageClass::FullyCovered);
    session_counters.record_route_coverage(4, 4, RouteCoverageClass::PartiallyCovered);
    let classified_layer_token_count = session_counters.hot_route_fully_covered_layer_token_count
        + session_counters.hot_route_partially_covered_layer_token_count
        + session_counters.hot_route_missed_layer_token_count;
    assert_eq!(
        classified_layer_token_count, 2,
        "one decode token over two layers must count two layer-tokens"
    );
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
fn should_split_source_read_elapsed_time_by_phase() {
    let mut session_counters = ExpertStreamingSessionCounters::default();
    session_counters.record_phase_source_read_elapsed(SessionStreamingPhase::Prefill, 9_000);
    session_counters.record_phase_source_read_elapsed(SessionStreamingPhase::Decode, 400);
    session_counters.record_phase_source_read_elapsed(SessionStreamingPhase::Decode, 100);
    assert_eq!(
        session_counters.prefill_source_read_elapsed_nanoseconds,
        9_000
    );
    assert_eq!(
        session_counters.decode_source_read_elapsed_nanoseconds, 500,
        "decode seconds must never include prefill read time"
    );
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
    session_counters.record_phase_source_read_elapsed(SessionStreamingPhase::Decode, u64::MAX);
    session_counters.record_phase_source_read_elapsed(SessionStreamingPhase::Decode, 1);
    assert_eq!(
        session_counters.decode_source_read_elapsed_nanoseconds,
        u64::MAX
    );
}

#[test]
fn should_credit_routed_expert_bytes_not_layer_payload_for_a_resident_decode_token() {
    // A decode token routes to 8 of 256 experts; only those 8 experts' bytes
    // were avoided, not the whole seated layer's payload (issue #1715).
    let complete_layer_payload_bytes = 4_096_000_000;
    let avoided_bytes = no_cache_route_source_bytes(complete_layer_payload_bytes, 256, 8);
    assert_eq!(avoided_bytes, complete_layer_payload_bytes / 256 * 8);
    assert!(avoided_bytes < complete_layer_payload_bytes);
}

#[test]
fn should_credit_the_routed_union_for_a_resident_prefill_chunk() {
    let complete_layer_payload_bytes = 4_096_000_000;
    // Duplicate expert ids across the chunk's tokens are one storage read.
    let duplicate_union_bytes = no_cache_route_source_bytes(complete_layer_payload_bytes, 256, 12);
    assert_eq!(
        duplicate_union_bytes,
        complete_layer_payload_bytes / 256 * 12
    );
    // A route that touches every expert equals exactly one layer payload.
    let whole_layer_union_bytes =
        no_cache_route_source_bytes(complete_layer_payload_bytes, 256, 256);
    assert_eq!(whole_layer_union_bytes, complete_layer_payload_bytes);
}

#[test]
fn should_never_credit_more_avoided_bytes_than_a_no_cache_baseline() {
    let complete_layer_payload_bytes = 4_096_000_000;
    // An over-full route saturates at the layer payload instead of exceeding it.
    assert_eq!(
        no_cache_route_source_bytes(complete_layer_payload_bytes, 256, 300),
        complete_layer_payload_bytes
    );
    // Degenerate geometry credits nothing rather than dividing by zero.
    assert_eq!(
        no_cache_route_source_bytes(complete_layer_payload_bytes, 0, 8),
        0
    );
    // A zero-capacity claim with no payload stays zero.
    assert_eq!(no_cache_route_source_bytes(0, 256, 8), 0);
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
