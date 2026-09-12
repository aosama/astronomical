use astronomical_model_serving::PerformanceAttribution;

#[test]
fn disabled_attribution_does_not_record_a_route_observation_chain() {
    let mut performance_attribution = PerformanceAttribution::disabled();
    let previous_route =
        performance_attribution.advance_route_observation_chain(vec![Some(vec![1, 2])]);
    assert!(
        previous_route.is_none(),
        "disabled attribution must not allocate a previous-route chain"
    );
}

#[test]
fn enabled_attribution_chains_the_previous_observed_route_within_one_request() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    let first_previous =
        performance_attribution.advance_route_observation_chain(vec![Some(vec![3, 7])]);
    assert!(
        first_previous.is_none(),
        "the first observed token of a request has no previous route"
    );
    let second_previous =
        performance_attribution.advance_route_observation_chain(vec![Some(vec![4, 8])]);
    assert_eq!(second_previous, Some(vec![Some(vec![3, 7])]));
}
