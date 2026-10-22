//! Boundary contracts for the deterministic residency planner.
//!
//! These tests pin the planner's fit edges: the routed floor can never exceed
//! a layer's complete payload (the invariant that keeps the overlay's
//! subtraction from underflowing), complete layers are admitted at the exact
//! leftover boundary, generation seating prefers complete layers before the
//! highest-coverage partials, and a zero budget streams everything instead of
//! erroring.

use crate::residency::ExpertLayerGeometry;
use crate::phase::MemoryPhase;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, ExpertResidencyPlanError,
    RetainedExpertPageClass, plan_expert_residency,
};

fn geometry(
    layer_index: usize,
    expert_payload_bytes: u64,
    expert_capacity: usize,
    experts_per_token: usize,
) -> ExpertLayerGeometry {
    ExpertLayerGeometry {
        layer_index,
        complete_layer_payload_bytes: expert_payload_bytes * expert_capacity as u64,
        expert_payload_bytes,
        expert_capacity,
        experts_per_token,
    }
}

#[test]
fn should_cap_the_routed_floor_at_the_complete_payload_for_adversarial_routing_width() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Routing width wider than the expert population: the per-layer floor is
    // min(experts_per_token, capacity) experts, so the floor equals the
    // complete payload and the overlay's complete-minus-floor subtraction
    // stays at zero instead of underflowing.
    let geometries = vec![geometry(0, 10, 4, 9), geometry(1, 10, 4, 4)];

    let plan = plan_expert_residency(MemoryPhase::Prefill, 80, &geometries, &[])
        .expect("adversarial routing width must plan, not underflow");

    assert_eq!(
        plan.complete_layer_targets,
        vec![0, 1],
        "a layer whose routed floor already equals its complete payload promotes at zero incremental cost"
    );
    assert!(!plan.is_low_budget_partial_mode);
}

#[test]
fn should_admit_a_complete_layer_at_the_exact_leftover_boundary_only() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Two uniform layers: floor 20 + complete 40 each. Floor sum 40.
    let geometries = vec![geometry(0, 10, 4, 2), geometry(1, 10, 4, 2)];

    let exact_fit = plan_expert_residency(MemoryPhase::Prefill, 60, &geometries, &[])
        .expect("a plan that exactly fits must succeed");
    assert_eq!(
        exact_fit.complete_layer_targets,
        vec![0],
        "20 leftover bytes admit layer 0 exactly; layer 1 must find nothing left"
    );

    let one_byte_short = plan_expert_residency(MemoryPhase::Prefill, 79, &geometries, &[])
        .expect("79 leaves 39 after the floor: layer 0 fits, layer 1 does not");
    assert_eq!(one_byte_short.complete_layer_targets, vec![0]);

    let both_fit = plan_expert_residency(MemoryPhase::Prefill, 80, &geometries, &[])
        .expect("the full ceiling must hold both complete layers");
    assert_eq!(both_fit.complete_layer_targets, vec![0, 1]);
}

#[test]
fn should_preserve_every_seated_generation_page_when_the_seating_ceiling_covers_it() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = vec![
        geometry(0, 10, 4, 2),
        geometry(1, 10, 4, 2),
        geometry(2, 10, 4, 2),
    ];
    let current_residencies = [
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0],
            payload_bytes: 10,
            covered_weighted_demand: 5,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1],
            payload_bytes: 20,
            covered_weighted_demand: 999,
        },
    ];

    // Input validation rejects any seating ceiling below the current
    // retained total, so a plan that reaches generation seating always fits
    // every current page. Preservation priority orders candidates (complete
    // layers first, then highest coverage-per-byte partials) but can never
    // release: the family must reclaim down to the seating ceiling before
    // planning, or validation fails the plan instead.
    let plan = plan_expert_residency(MemoryPhase::Decode, 70, &geometries, &current_residencies)
        .expect("a seating ceiling at the retained total must plan");

    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreserveComplete
    );
    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::PreservePartial
    );
    assert_eq!(
        plan.layer_targets[2],
        ExpertLayerResidencyTarget::PreservePartial
    );
    assert_eq!(plan.expected_preserved_bytes, 70);
}

#[test]
fn should_reject_generation_seating_when_the_leftover_ceiling_shrinks_below_current_retention() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = vec![geometry(0, 10, 4, 2)];
    let current_residency = CurrentExpertLayerResidency {
        layer_index: 0,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    };

    // Generation seating has no release path of its own: a leftover budget
    // smaller than current retention is an input defect, and the family's
    // request-pressure reclamation must run before this plan is composed.
    assert_eq!(
        plan_expert_residency(MemoryPhase::Decode, 39, &geometries, &[current_residency],)
            .unwrap_err(),
        ExpertResidencyPlanError::CurrentResidencyExceedsCeiling
    );
}

#[test]
fn should_stream_every_layer_when_the_budget_cannot_hold_even_one_routed_floor() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = vec![geometry(0, 10, 4, 2), geometry(1, 10, 4, 2)];

    let plan = plan_expert_residency(MemoryPhase::Decode, 5, &geometries, &[])
        .expect("a zero-capacity budget must plan into streaming mode, not error");

    assert!(plan.is_low_budget_partial_mode);
    assert_eq!(
        plan.layer_targets,
        vec![
            ExpertLayerResidencyTarget::StreamOperationLocal,
            ExpertLayerResidencyTarget::StreamOperationLocal,
        ]
    );
    assert_eq!(plan.maximum_new_retained_bytes, 0);
}

#[test]
fn should_reject_current_residency_one_byte_above_the_planning_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let geometries = vec![geometry(0, 10, 4, 2)];
    let current_residency = CurrentExpertLayerResidency {
        layer_index: 0,
        class: RetainedExpertPageClass::ElasticRoutedExperts,
        retained_expert_ids: vec![0],
        payload_bytes: 10,
        covered_weighted_demand: 0,
    };

    assert!(
        plan_expert_residency(
            MemoryPhase::Prefill,
            10,
            &geometries,
            &[current_residency.clone()],
        )
        .is_ok()
    );
    assert_eq!(
        plan_expert_residency(MemoryPhase::Prefill, 9, &geometries, &[current_residency])
            .unwrap_err(),
        ExpertResidencyPlanError::CurrentResidencyExceedsCeiling
    );
}
