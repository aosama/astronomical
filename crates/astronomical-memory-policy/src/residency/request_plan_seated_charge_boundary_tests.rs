//! Boundary contracts for charging already-seated streamed layers.
//!
//! A seated streamed layer is preserved without a fresh admission, so the
//! request contract must charge its bytes before funding anything new;
//! leaving them out let every chunk admit one more complete layer past the
//! ceiling. These tests pin that ordering at the exact ceiling edge.

use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, RequestExpertResidency,
    RetainedExpertPageClass,
};

use super::request_plan_boundary_tests::{four_layer_candidate, uniform_geometry};

#[test]
fn should_stop_admitting_streamed_layers_once_seated_streamed_layers_fill_the_ceiling() {
    // The 2026-10-25 cached-suffix flight: a seated streamed layer was
    // preserved without charging its bytes, so every chunk admitted one more
    // complete layer past the ceiling. One pin plus one seated streamed layer
    // fill an 80-byte ceiling exactly; the remaining streamed layers must
    // stream instead of seating on top of it.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0], 80));
    let seated_pin_and_streamed_layer = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];

    let stabilized = request_residency.stabilize_prefill_plan(
        four_layer_candidate(Vec::new(), 80),
        &seated_pin_and_streamed_layer,
        &geometries,
    );

    assert_eq!(
        stabilized.layer_targets[1],
        ExpertLayerResidencyTarget::PreserveComplete,
        "the seated streamed layer keeps its paid-for table"
    );
    assert_eq!(
        stabilized.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "a ceiling filled by seated layers must not admit another complete layer"
    );
    assert_eq!(
        stabilized.layer_targets[3],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
    assert_eq!(stabilized.maximum_new_retained_bytes, 0);
    assert_eq!(
        stabilized.expected_preserved_bytes + stabilized.maximum_new_retained_bytes,
        80,
        "preserved plus new retention must never exceed the retained ceiling"
    );
}

#[test]
fn should_fund_unseated_pins_only_after_charging_seated_streamed_layers() {
    // Edge: a seated streamed layer holds real bytes, so an unseated pin
    // competing for the same 40 bytes of headroom must lose its claim.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0, 2], 80));
    let seated_pin_and_streamed_layer = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];

    let stabilized = request_residency.stabilize_prefill_plan(
        four_layer_candidate(Vec::new(), 80),
        &seated_pin_and_streamed_layer,
        &geometries,
    );

    assert_eq!(stabilized.complete_layer_targets, vec![0]);
    assert_eq!(
        stabilized.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "an unseated pin the seated layers leave no room for must stream"
    );
}
