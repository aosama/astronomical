//! Behavioral coverage for chunk-and-layer expert source attribution.

use super::*;
use crate::memory::SessionStreamingPhase;
use crate::performance_attribution::StreamedPageSourcePlan;

/// Builds one source plan record so each replay reads as evidence, not noise.
fn source_plan(
    layer_index: usize,
    route_token_count: i32,
    serving_phase: SessionStreamingPhase,
    routed_expert_count: usize,
    streamed_expert_count: usize,
    source_shard_count: usize,
    payload_byte_count: u64,
) -> StreamedPageSourcePlan {
    StreamedPageSourcePlan {
        layer_index,
        route_token_count,
        serving_phase,
        routed_expert_count,
        streamed_expert_count,
        source_shard_count,
        payload_byte_count,
    }
}

#[test]
fn should_aggregate_expert_source_plans_by_phase_and_layer_without_expert_ids() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        7,
        2_048,
        SessionStreamingPhase::Prefill,
        8,
        256,
        3,
        900,
    ));
    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        7,
        417,
        SessionStreamingPhase::Prefill,
        7,
        256,
        3,
        900,
    ));
    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        7,
        1,
        SessionStreamingPhase::Decode,
        8,
        8,
        2,
        30,
    ));

    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes),
        1_800
    );
    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes),
        30
    );

    let report = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    let source_summaries = report["expert_streaming_source_summaries"]
        .as_array()
        .expect("generation attribution should expose bounded expert source summaries");

    assert_eq!(source_summaries.len(), 2);
    assert_eq!(source_summaries[0]["phase"], "prefill");
    assert_eq!(source_summaries[0]["source_plan_count"], 2);
    assert_eq!(source_summaries[0]["total_route_token_count"], 2_465);
    assert_eq!(source_summaries[0]["payload_byte_count"], 1_800);
    assert_eq!(source_summaries[1]["phase"], "decode");
    assert_eq!(source_summaries[1]["source_plan_count"], 1);
    assert_eq!(source_summaries[1]["total_streamed_expert_count"], 8);
    assert_eq!(source_summaries[1]["total_source_shard_count"], 2);
    assert_eq!(source_summaries[1]["payload_byte_count"], 30);
    assert!(!report.to_string().contains("expert_ids"));
}

#[test]
fn should_attribute_a_multi_token_decode_route_to_decode() {
    // A speculative or multi-token decode forward routes several tokens at
    // once; inferring the phase from the route length would file its reads
    // under prefill. The engine-declared phase must win (issue #1715).
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        11,
        4,
        SessionStreamingPhase::Decode,
        8,
        32,
        2,
        500,
    ));
    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes),
        500
    );
    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes),
        0
    );
    let report = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    let source_summaries = report["expert_streaming_source_summaries"]
        .as_array()
        .expect("generation attribution should expose bounded expert source summaries");
    assert_eq!(source_summaries[0]["phase"], "decode");
    assert_eq!(source_summaries[0]["total_route_token_count"], 4);
}

#[test]
fn should_bound_expert_source_summaries_to_one_row_per_phase_and_layer() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        2,
        128,
        SessionStreamingPhase::Prefill,
        8,
        256,
        1,
        100,
    ));
    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        4,
        128,
        SessionStreamingPhase::Prefill,
        8,
        256,
        1,
        100,
    ));
    performance_attribution.record_expert_streaming_source_plan(&source_plan(
        2,
        128,
        SessionStreamingPhase::Prefill,
        8,
        256,
        1,
        100,
    ));

    let report = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    let source_summaries = report["expert_streaming_source_summaries"]
        .as_array()
        .expect("generation attribution should expose bounded expert source summaries");

    assert_eq!(source_summaries.len(), 2);
    assert_eq!(source_summaries[0]["layer_index"], 2);
    assert_eq!(source_summaries[0]["source_plan_count"], 2);
    assert_eq!(source_summaries[0]["payload_byte_count"], 200);
    assert_eq!(source_summaries[1]["layer_index"], 4);
    assert_eq!(source_summaries[1]["source_plan_count"], 1);
}
