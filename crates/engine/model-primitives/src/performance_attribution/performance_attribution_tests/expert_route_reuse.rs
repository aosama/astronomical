use super::*;
use crate::performance_attribution::{
    PreviousTokenExpertRouteReuseMeasurement, sorted_unique_expert_ids,
};

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

#[test]
fn should_compare_one_layer_against_the_previous_token_from_sorted_expert_lists() {
    let partial_overlap =
        PreviousTokenExpertRouteReuseMeasurement::between_decode_tokens(&[2, 5, 9], &[2, 5, 7]);
    assert_eq!(partial_overlap.predicted_expert_count, 3);
    assert_eq!(partial_overlap.matched_expert_count, 2);
    assert_eq!(partial_overlap.completely_matched_layer_count, 0);
    assert_eq!(partial_overlap.examined_layer_count, 1);

    let identical_routes =
        PreviousTokenExpertRouteReuseMeasurement::between_decode_tokens(&[2, 5, 9], &[2, 5, 9]);
    assert_eq!(identical_routes.matched_expert_count, 3);
    assert_eq!(identical_routes.completely_matched_layer_count, 1);
}

#[test]
fn should_count_no_matches_between_disjoint_or_empty_expert_lists() {
    let disjoint_routes =
        PreviousTokenExpertRouteReuseMeasurement::between_decode_tokens(&[1, 2], &[3, 4]);
    assert_eq!(disjoint_routes.predicted_expert_count, 2);
    assert_eq!(disjoint_routes.matched_expert_count, 0);
    assert_eq!(disjoint_routes.completely_matched_layer_count, 0);

    let empty_routes = PreviousTokenExpertRouteReuseMeasurement::between_decode_tokens(&[], &[]);
    assert_eq!(empty_routes.predicted_expert_count, 0);
    assert_eq!(empty_routes.matched_expert_count, 0);
    assert_eq!(empty_routes.completely_matched_layer_count, 1);
    assert_eq!(empty_routes.examined_layer_count, 1);
}

#[test]
fn should_sort_and_deduplicate_the_selected_experts_before_comparison() {
    assert_eq!(sorted_unique_expert_ids(&[9, 2, 2, 5]), vec![2, 5, 9]);
    assert_eq!(sorted_unique_expert_ids(&[4]), vec![4]);
    assert_eq!(sorted_unique_expert_ids(&[]), Vec::<usize>::new());
}

#[test]
fn should_add_layer_observations_together_and_stop_at_the_largest_count() {
    let mut layer_total = PreviousTokenExpertRouteReuseMeasurement {
        predicted_expert_count: 3,
        matched_expert_count: 2,
        completely_matched_layer_count: 0,
        examined_layer_count: 1,
    };

    layer_total.accumulate(
        PreviousTokenExpertRouteReuseMeasurement::between_decode_tokens(&[1, 2], &[1, 2]),
    );

    assert_eq!(layer_total.predicted_expert_count, 5);
    assert_eq!(layer_total.matched_expert_count, 4);
    assert_eq!(layer_total.completely_matched_layer_count, 1);
    assert_eq!(layer_total.examined_layer_count, 2);

    let mut saturated_total = PreviousTokenExpertRouteReuseMeasurement {
        predicted_expert_count: u64::MAX,
        matched_expert_count: u64::MAX,
        completely_matched_layer_count: u64::MAX,
        examined_layer_count: u64::MAX,
    };
    saturated_total.accumulate(layer_total);
    assert_eq!(saturated_total.predicted_expert_count, u64::MAX);
    assert_eq!(saturated_total.examined_layer_count, u64::MAX);
}

#[test]
fn should_pair_each_route_reuse_field_with_its_report_counter() {
    let route_reuse = PreviousTokenExpertRouteReuseMeasurement {
        predicted_expert_count: 4,
        matched_expert_count: 3,
        completely_matched_layer_count: 1,
        examined_layer_count: 1,
    };

    assert_eq!(
        route_reuse.counter_increments(),
        [
            (PerformanceCounter::ExpertRoutePredictedExpertCount, 4),
            (PerformanceCounter::ExpertRouteMatchedExpertCount, 3),
            (
                PerformanceCounter::ExpertRouteCompletelyMatchedLayerCount,
                1
            ),
            (PerformanceCounter::ExpertRouteExaminedLayerCount, 1),
        ]
    );
}
