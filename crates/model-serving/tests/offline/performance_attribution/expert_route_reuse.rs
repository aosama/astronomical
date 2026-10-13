use super::*;

#[test]
fn should_measure_previous_token_expert_route_reuse_without_serializing_expert_ids() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_previous_token_expert_route_reuse(3, 1, &[9, 2, 2, 5]);
    performance_attribution.record_previous_token_expert_route_reuse(3, 1, &[7, 5, 2, 2]);

    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRoutePredictedExpertCount),
        3
    );
    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRouteMatchedExpertCount),
        2
    );
    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRouteExaminedLayerCount),
        1
    );
    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::ExpertRouteCompletelyMatchedLayerCount),
        0
    );

    let performance_attribution_json = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    assert_eq!(
        performance_attribution_json["previous_token_expert_route_reuse_by_layer"][0]["layer_index"],
        3
    );
    assert_eq!(
        performance_attribution_json["previous_token_expert_route_reuse_by_layer"][0]["predicted_expert_count"],
        3
    );
    assert_eq!(
        performance_attribution_json["previous_token_expert_route_reuse_by_layer"][0]["matched_expert_count"],
        2
    );
    assert!(
        !performance_attribution_json
            .to_string()
            .contains("\"expert_ids\"")
    );
}

#[test]
fn should_measure_expert_route_reuse_independently_for_each_layer() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    performance_attribution.record_previous_token_expert_route_reuse(1, 1, &[2, 3]);
    performance_attribution.record_previous_token_expert_route_reuse(4, 1, &[8, 9]);
    performance_attribution.record_previous_token_expert_route_reuse(1, 1, &[2, 3]);
    performance_attribution.record_previous_token_expert_route_reuse(4, 1, &[9, 10]);

    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRoutePredictedExpertCount),
        4
    );
    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRouteMatchedExpertCount),
        3
    );
    assert_eq!(
        performance_attribution
            .counter_value(PerformanceCounter::ExpertRouteCompletelyMatchedLayerCount),
        1
    );
    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRouteExaminedLayerCount),
        2
    );
}

#[test]
fn should_skip_multi_token_expert_routes_and_disabled_attribution() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_previous_token_expert_route_reuse(2, 4, &[1, 2]);
    performance_attribution.record_previous_token_expert_route_reuse(2, 1, &[1, 2]);

    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::ExpertRouteExaminedLayerCount),
        0,
        "a multi-token route must not become the previous decode-token prediction"
    );

    let mut disabled_performance_attribution = PerformanceAttribution::disabled();
    disabled_performance_attribution.record_previous_token_expert_route_reuse(2, 1, &[1, 2]);
    disabled_performance_attribution.record_previous_token_expert_route_reuse(2, 1, &[1, 2]);
    assert_eq!(
        disabled_performance_attribution
            .counter_value(PerformanceCounter::ExpertRouteExaminedLayerCount),
        0
    );
}
