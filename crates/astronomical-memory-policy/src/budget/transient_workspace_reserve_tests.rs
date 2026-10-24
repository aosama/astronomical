use super::{
    TransientWorkspaceObservation, TransientWorkspaceReserve, TransientWorkspaceReserveSource,
};

#[test]
fn should_refuse_to_scale_a_smaller_observation_up() {
    let short_observation = TransientWorkspaceObservation::new(12_000, 4);

    assert_eq!(
        TransientWorkspaceReserve::scaled_down_to(short_observation, 5),
        None
    );
}

#[test]
fn should_reject_zero_token_spans_for_scaled_reserves() {
    assert_eq!(
        TransientWorkspaceReserve::scaled_down_to(TransientWorkspaceObservation::new(12_000, 0), 1,),
        None
    );
    assert_eq!(
        TransientWorkspaceReserve::scaled_down_to(TransientWorkspaceObservation::new(12_000, 4), 0,),
        None
    );
}

#[test]
fn should_record_provenance_for_exact_scaled_and_analytic_reserves() {
    let observed_forward = TransientWorkspaceObservation::new(10, 3);
    let exact_reserve = TransientWorkspaceReserve::exact_context(observed_forward);
    let scaled_reserve =
        TransientWorkspaceReserve::scaled_down_to(observed_forward, 1).expect("smaller span");
    let analytic_reserve = TransientWorkspaceReserve::analytic_floor(20, 1);

    assert_eq!(exact_reserve.reserve_bytes(), 10);
    assert_eq!(exact_reserve.forward_token_count(), 3);
    assert_eq!(
        exact_reserve.source(),
        TransientWorkspaceReserveSource::ExactContext
    );
    assert_eq!(scaled_reserve.reserve_bytes(), 4);
    assert_eq!(scaled_reserve.forward_token_count(), 1);
    assert_eq!(
        scaled_reserve.source(),
        TransientWorkspaceReserveSource::PhaseScaled
    );
    assert_eq!(
        analytic_reserve.source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan
    );
}

#[test]
fn should_keep_the_larger_smaller_span_observation_without_scaling_it() {
    let analytic_floor = TransientWorkspaceReserve::analytic_floor(100, 2_048);
    let smaller_observation = TransientWorkspaceObservation::new(150, 38);

    let combined_reserve =
        analytic_floor.max_with_smaller_unscaled_observation(smaller_observation);

    assert_eq!(combined_reserve.reserve_bytes(), 150);
    assert_eq!(combined_reserve.forward_token_count(), 2_048);
    assert_eq!(
        combined_reserve.source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan
    );
}

#[test]
fn should_record_zero_bytes_and_no_learned_source_for_request_admission() {
    let without_learned_reserve = TransientWorkspaceReserve::without_learned_reserve();

    assert_eq!(without_learned_reserve.reserve_bytes(), 0);
    assert_eq!(without_learned_reserve.forward_token_count(), 0);
    assert_eq!(
        without_learned_reserve.source(),
        TransientWorkspaceReserveSource::NoLearnedReserve
    );
}

#[test]
fn should_ignore_equal_or_larger_observation_spans_in_the_unscaled_combinator() {
    let analytic_floor = TransientWorkspaceReserve::analytic_floor(100, 38);

    assert_eq!(
        analytic_floor
            .max_with_smaller_unscaled_observation(TransientWorkspaceObservation::new(500, 38)),
        analytic_floor
    );
    assert_eq!(
        analytic_floor
            .max_with_smaller_unscaled_observation(TransientWorkspaceObservation::new(500, 0)),
        analytic_floor
    );
}
