//! Behavioral coverage for chunk-and-layer expert source attribution.

use super::*;

#[test]
fn should_aggregate_expert_source_plans_by_phase_and_layer_without_expert_ids() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_expert_streaming_source_plan(7, 2_048, 8, 256, 3, 900);
    performance_attribution.record_expert_streaming_source_plan(7, 417, 7, 256, 3, 900);
    performance_attribution.record_expert_streaming_source_plan(7, 1, 8, 8, 2, 30);

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
fn should_bound_expert_source_summaries_to_one_row_per_phase_and_layer() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_expert_streaming_source_plan(2, 128, 8, 256, 1, 100);
    performance_attribution.record_expert_streaming_source_plan(4, 128, 8, 256, 1, 100);
    performance_attribution.record_expert_streaming_source_plan(2, 128, 8, 256, 1, 100);

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
