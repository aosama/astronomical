//! Boundary, negative, and happy-path tests for measured RAM-budget
//! geometry composition.

use crate::budget::ram_geometry::{
    MeasuredExpertLayerPayload, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};

#[test]
fn should_compose_geometry_and_headroom_from_measured_layer_facts() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Two sparse layers plus a denser non-expert remainder; the startup
    // headroom must take the measured transient because it exceeds the
    // zero floor a bare expert-only geometry would suggest.
    let (model_geometry, required_headroom_bytes) =
        mlx_ram_budget_model_geometry_from_measured_layer_facts(
            &[
                MeasuredExpertLayerPayload::new(100, 3),
                MeasuredExpertLayerPayload::new(40, 4),
            ],
            200,
            2,
            30,
        )
        .expect("consistent layer facts should compose");

    assert_eq!(model_geometry.complete_expert_payload_bytes, 140);
    assert_eq!(model_geometry.largest_complete_expert_layer_bytes, 100);
    assert_eq!(
        model_geometry.model_core_payload_bytes, 60,
        "the non-expert remainder is the total minus every expert byte"
    );
    assert_eq!(required_headroom_bytes, 30);
}

#[test]
fn should_reserve_the_ceiling_of_the_largest_routed_page_when_payloads_do_not_divide_evenly() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // 100 bytes across 3 experts cannot split evenly, so the largest
    // top-2 page is at least ceil(100 * 2 / 3) = 67 bytes; a floor-rounded
    // 66 under-reserves the decode stream slot and the geometry-safe
    // minimum ceiling, letting admission accept a page that cannot fit.
    let (model_geometry, _) = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[MeasuredExpertLayerPayload::new(100, 3)],
        100,
        2,
        0,
    )
    .expect("one uneven layer should compose");

    assert_eq!(
        model_geometry.largest_routed_expert_page_bytes, 67,
        "the routed-page reserve is ceil(100 * 2 / 3): never below the true \
         top-K page, never more than one expert byte over it"
    );
}

#[test]
fn should_clamp_the_routed_page_to_one_complete_layer_when_top_k_exceeds_capacity() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Asking for more experts per token than a layer holds selects the
    // whole layer: the routed page equals the complete payload exactly.
    let (model_geometry, _) = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[MeasuredExpertLayerPayload::new(100, 3)],
        100,
        10,
        0,
    )
    .expect("one layer should compose");

    assert_eq!(model_geometry.largest_routed_expert_page_bytes, 100);
}

#[test]
fn should_reserve_no_routed_page_when_no_expert_is_routed_per_token() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let (model_geometry, _) = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[MeasuredExpertLayerPayload::new(100, 3)],
        100,
        0,
        0,
    )
    .expect("zero experts per token should compose");

    assert_eq!(model_geometry.largest_routed_expert_page_bytes, 0);
}

#[test]
fn should_compose_without_panicking_when_a_layer_reports_zero_expert_capacity() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A zero-capacity layer contradicts a nonzero payload, but the composer
    // must fail safe (a zero page reserve), never divide by zero.
    let (model_geometry, _) = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[MeasuredExpertLayerPayload::new(100, 0)],
        100,
        2,
        0,
    )
    .expect("a zero-capacity layer still composes today");

    assert_eq!(model_geometry.largest_routed_expert_page_bytes, 0);
}

#[test]
fn should_reject_an_empty_layer_fact_table_as_not_sparse() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let composition_error = mlx_ram_budget_model_geometry_from_measured_layer_facts(&[], 100, 2, 0)
        .expect_err("no sparse layers must fail closed");

    assert_eq!(
        composition_error,
        RamBudgetGeometryError::NotSparseMixtureOfExperts
    );
}

#[test]
fn should_reject_layer_facts_whose_expert_payloads_overflow() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let composition_error = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[
            MeasuredExpertLayerPayload::new(u64::MAX, 4),
            MeasuredExpertLayerPayload::new(1, 4),
        ],
        u64::MAX,
        2,
        0,
    )
    .expect_err("overflowing payload sums must fail closed");

    assert_eq!(
        composition_error,
        RamBudgetGeometryError::ExpertPayloadOverflow
    );
}

#[test]
fn should_floor_the_model_core_at_zero_when_the_total_undercounts_the_expert_payload() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A total smaller than the measured expert bytes is contradictory input;
    // today the composer clamps the core to zero rather than failing, which
    // this test pins so a future fail-closed tightening is a conscious
    // contract change.
    let (model_geometry, _) = mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &[MeasuredExpertLayerPayload::new(140, 4)],
        50,
        2,
        0,
    )
    .expect("contradictory totals still compose today");

    assert_eq!(model_geometry.model_core_payload_bytes, 0);
}

#[test]
fn should_expose_measured_layer_facts_through_accessors() {
    use crate::budget::ram_geometry::MeasuredExpertLayerPayload;
    let measured_layer = MeasuredExpertLayerPayload::new(408, 6);
    assert_eq!(measured_layer.complete_payload_bytes(), 408);
    assert_eq!(measured_layer.expert_capacity(), 6);
}
