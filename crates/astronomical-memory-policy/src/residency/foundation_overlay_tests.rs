//! Beside-source tests for the pure foundation-and-overlay planning stages.

use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPlanError, RetainedExpertPageClass,
};

use super::{
    FoundationOverlayLayerFacts, account_planned_layer_charges,
    select_complete_layers_within_ceiling, select_preserved_partial_layers,
};

fn layered_geometry(complete_layer_payload_bytes_per_layer: &[u64]) -> Vec<ExpertLayerGeometry> {
    complete_layer_payload_bytes_per_layer
        .iter()
        .enumerate()
        .map(
            |(layer_index, &complete_layer_payload_bytes)| ExpertLayerGeometry {
                layer_index,
                complete_layer_payload_bytes,
                expert_payload_bytes: 10,
                expert_capacity: 4,
                experts_per_token: 2,
            },
        )
        .collect()
}

fn elastic_layer_residency(layer_index: usize, payload_bytes: u64) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::ElasticRoutedExperts,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes,
        covered_weighted_demand: 5,
    }
}

fn complete_layer_residency(layer_index: usize, payload_bytes: u64) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes,
        covered_weighted_demand: 5,
    }
}

fn layer_facts<'a>(
    geometries: &'a [ExpertLayerGeometry],
    current_by_layer: &'a [Option<&'a CurrentExpertLayerResidency>],
    routed_floor_bytes: &'a [u64],
) -> FoundationOverlayLayerFacts<'a> {
    FoundationOverlayLayerFacts {
        geometries,
        current_by_layer,
        routed_floor_bytes,
    }
}

#[test]
fn should_promote_the_cheapest_complete_layers_first_until_the_ceiling_is_exhausted() {
    // Routed floors total 30, leaving a 50-byte headroom under the 80-byte
    // ceiling: layer 1's 30-byte incremental seats first, and the 40- and
    // 50-byte incrementals no longer fit afterwards.
    let geometries = layered_geometry(&[50, 40, 60]);
    let routed_floor_bytes = [10, 10, 10];
    let current_by_layer: [Option<&CurrentExpertLayerResidency>; 3] = [None, None, None];

    let selection = select_complete_layers_within_ceiling(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        80,
        30,
    )
    .expect("the selection should not overflow");

    assert_eq!(selection.complete_targets, vec![false, true, false]);
    assert_eq!(selection.foundation_and_floor_bytes, 60);
}

#[test]
fn should_select_no_complete_layer_when_no_incremental_fits_above_the_floors() {
    let geometries = layered_geometry(&[40, 40]);
    let routed_floor_bytes = [20, 20];
    let current_by_layer: [Option<&CurrentExpertLayerResidency>; 2] = [None, None];

    let selection = select_complete_layers_within_ceiling(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        40,
        40,
    )
    .expect("the selection should not overflow");

    assert_eq!(selection.complete_targets, vec![false, false]);
    assert_eq!(selection.foundation_and_floor_bytes, 40);
}

#[test]
fn should_seat_a_complete_layer_at_the_exact_incremental_boundary() {
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let current_by_layer: [Option<&CurrentExpertLayerResidency>; 1] = [None];

    let selection = select_complete_layers_within_ceiling(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        40,
        20,
    )
    .expect("the selection should not overflow");

    assert_eq!(selection.complete_targets, vec![true]);
    assert_eq!(selection.foundation_and_floor_bytes, 40);
}

#[test]
fn should_seat_no_complete_layer_when_the_ceiling_is_below_the_floor_foundation() {
    // The headroom saturates at zero instead of underflowing, so nothing
    // seats even though the raw comparison would go negative.
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let current_by_layer: [Option<&CurrentExpertLayerResidency>; 1] = [None];

    let selection = select_complete_layers_within_ceiling(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        10,
        20,
    )
    .expect("the selection should not overflow");

    assert_eq!(selection.complete_targets, vec![false]);
    assert_eq!(selection.foundation_and_floor_bytes, 20);
}

#[test]
fn should_prefer_an_already_seated_complete_layer_over_a_cheaper_unseated_one() {
    // With a 20-byte headroom, cheapest-first alone would seat layer 1's
    // 10-byte incremental and drop layer 0; the seated-complete preference
    // processes layer 0 first, so its 20-byte incremental takes the last of
    // the headroom and the cheaper unseated layer gets nothing.
    let geometries = layered_geometry(&[30, 20]);
    let routed_floor_bytes = [10, 10];
    let seated_complete_residency = complete_layer_residency(0, 30);
    let current_by_layer = [Some(&seated_complete_residency), None];

    let selection = select_complete_layers_within_ceiling(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        40,
        20,
    )
    .expect("the selection should not overflow");

    assert_eq!(selection.complete_targets, vec![true, false]);
    assert_eq!(selection.foundation_and_floor_bytes, 40);
}

#[test]
fn should_preserve_an_elastic_page_that_fits_the_overlay_budget_exactly() {
    let geometries = layered_geometry(&[40, 40]);
    let routed_floor_bytes = [20, 20];
    let elastic_residency = elastic_layer_residency(0, 30);
    let current_by_layer = [Some(&elastic_residency), None];

    let preserve_partial = select_preserved_partial_layers(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        50,
        40,
        &[false, false],
    );

    assert_eq!(preserve_partial, vec![true, false]);
}

#[test]
fn should_not_preserve_an_elastic_page_larger_than_the_overlay_budget() {
    let geometries = layered_geometry(&[40, 40]);
    let routed_floor_bytes = [20, 20];
    let oversized_elastic_residency = elastic_layer_residency(0, 31);
    let current_by_layer = [Some(&oversized_elastic_residency), None];

    let preserve_partial = select_preserved_partial_layers(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        50,
        40,
        &[false, false],
    );

    assert_eq!(preserve_partial, vec![false, false]);
}

#[test]
fn should_skip_elastic_pages_on_layers_with_complete_targets() {
    let geometries = layered_geometry(&[40, 40]);
    let routed_floor_bytes = [20, 20];
    let elastic_residency = elastic_layer_residency(0, 30);
    let current_by_layer = [Some(&elastic_residency), None];

    let preserve_partial = select_preserved_partial_layers(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        200,
        40,
        &[true, false],
    );

    assert_eq!(
        preserve_partial,
        vec![false, false],
        "a complete-target layer must never keep a partial page"
    );
}

#[test]
fn should_skip_seated_pages_that_are_not_elastic() {
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let seated_complete_residency = complete_layer_residency(0, 30);
    let current_by_layer = [Some(&seated_complete_residency)];

    let preserve_partial = select_preserved_partial_layers(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        200,
        20,
        &[false],
    );

    assert_eq!(preserve_partial, vec![false]);
}

#[test]
fn should_charge_only_the_missing_bytes_when_promoting_over_a_smaller_elastic_page() {
    let geometries = layered_geometry(&[40, 40]);
    let routed_floor_bytes = [20, 20];
    let elastic_residency = elastic_layer_residency(0, 20);
    let current_by_layer = [Some(&elastic_residency), None];

    let accounting = account_planned_layer_charges(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        &[true, false],
        &[false, false],
    )
    .expect("the charges should not overflow");

    assert_eq!(
        accounting.layer_targets,
        vec![
            ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead,
            ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
        ]
    );
    assert_eq!(accounting.preserved_bytes, 20);
    assert_eq!(accounting.new_retained_bytes, 40);
}

#[test]
fn should_preserve_the_full_elastic_page_and_charge_only_its_floor_overlap() {
    // The 30-byte elastic page overlaps its 20-byte charged floor, so only
    // that overlap reduces the new-retention budget; the 10 resident bytes
    // above the floor are already-resident overlay.
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let oversized_elastic_residency = elastic_layer_residency(0, 30);
    let current_by_layer = [Some(&oversized_elastic_residency)];

    let accounting = account_planned_layer_charges(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        &[false],
        &[true],
    )
    .expect("the charges should not overflow");

    assert_eq!(
        accounting.layer_targets,
        vec![ExpertLayerResidencyTarget::PreservePartial]
    );
    assert_eq!(accounting.preserved_bytes, 30);
    assert_eq!(accounting.new_retained_bytes, 0);
}

#[test]
fn should_count_the_reserved_floor_of_a_released_complete_layer_as_preserved() {
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let seated_complete_residency = complete_layer_residency(0, 40);
    let current_by_layer = [Some(&seated_complete_residency)];

    let accounting = account_planned_layer_charges(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        &[false],
        &[false],
    )
    .expect("the charges should not overflow");

    assert_eq!(
        accounting.layer_targets,
        vec![ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit]
    );
    assert_eq!(accounting.preserved_bytes, 20);
    assert_eq!(accounting.new_retained_bytes, 0);
}

#[test]
fn should_count_only_the_resident_remainder_when_a_released_page_is_smaller_than_its_floor() {
    let geometries = layered_geometry(&[40]);
    let routed_floor_bytes = [20];
    let undersized_complete_residency = complete_layer_residency(0, 10);
    let current_by_layer = [Some(&undersized_complete_residency)];

    let accounting = account_planned_layer_charges(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        &[false],
        &[false],
    )
    .expect("the charges should not overflow");

    assert_eq!(accounting.preserved_bytes, 10);
    assert_eq!(accounting.new_retained_bytes, 10);
}

#[test]
fn should_report_a_byte_count_overflow_when_layer_charges_overflow() {
    // Two admitted floors of u64::MAX and 1 cannot be summed, so the second
    // checked addition must stop planning with the overflow error.
    let geometries = layered_geometry(&[10, 10]);
    let routed_floor_bytes = [u64::MAX, 1];
    let current_by_layer: [Option<&CurrentExpertLayerResidency>; 2] = [None, None];

    let accounting_error = account_planned_layer_charges(
        &layer_facts(&geometries, &current_by_layer, &routed_floor_bytes),
        &[false, false],
        &[false, false],
    )
    .expect_err("a u64 overflow must stop planning before execution");

    assert_eq!(
        accounting_error,
        ExpertResidencyPlanError::ByteCountOverflow
    );
}
