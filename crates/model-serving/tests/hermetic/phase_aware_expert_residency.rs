use astronomical_model_serving::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlanError, RetainedExpertPageClass,
    plan_phase_aware_expert_residency,
};

fn uniform_geometry(layer_count: usize) -> Vec<ExpertLayerGeometry> {
    (0..layer_count)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect()
}

fn partial_residency(
    layer_index: usize,
    retained_expert_ids: &[usize],
    covered_weighted_demand: u64,
) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::ElasticRoutedExperts,
        retained_expert_ids: retained_expert_ids.to_vec(),
        payload_bytes: u64::try_from(retained_expert_ids.len()).unwrap_or(u64::MAX) * 10,
        covered_weighted_demand,
    }
}

fn complete_residency(layer_index: usize) -> CurrentExpertLayerResidency {
    CurrentExpertLayerResidency {
        layer_index,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    }
}

#[test]
fn should_target_every_layer_complete_when_the_composed_budget_fits_the_model() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Prefill,
        120,
        &uniform_geometry(3),
        &[],
    )
    .expect("complete model geometry should fit");

    assert_eq!(plan.complete_layer_targets, vec![0, 1, 2]);
    assert!(
        plan.layer_targets.iter().all(|target| {
            *target == ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        })
    );
    assert_eq!(plan.maximum_new_retained_bytes, 120);
}

#[test]
fn should_preserve_existing_complete_layers_before_selecting_new_targets() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::GenerationPreparation,
        80,
        &uniform_geometry(3),
        &[complete_residency(1)],
    )
    .expect("one complete layer plus routed floors should fit");

    assert_eq!(plan.complete_layer_targets, vec![1]);
    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::PreserveComplete
    );
    assert_eq!(
        plan.layer_targets[0],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
    );
    assert_eq!(plan.reserved_routed_overlay_bytes, 0);
}

#[test]
fn should_reserve_one_model_derived_routed_page_for_each_incomplete_layer() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Decode,
        80,
        &uniform_geometry(3),
        &[complete_residency(0)],
    )
    .expect("complete foundation and routed floors should fit");

    assert_eq!(plan.complete_layer_targets, vec![0]);
    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
    );
    assert_eq!(plan.reserved_routed_overlay_bytes, 0);
}

#[test]
fn should_select_additional_complete_layers_by_incremental_payload_then_layer_index() {
    let geometries = vec![
        ExpertLayerGeometry {
            layer_index: 0,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        },
        ExpertLayerGeometry {
            layer_index: 1,
            complete_layer_payload_bytes: 80,
            expert_payload_bytes: 20,
            expert_capacity: 4,
            experts_per_token: 2,
        },
    ];
    let plan =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 80, &geometries, &[])
            .expect("one incremental complete target should fit");

    assert_eq!(plan.complete_layer_targets, vec![0]);
}

#[test]
fn should_use_low_budget_partial_mode_when_routed_floors_do_not_fit() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Decode,
        10,
        &uniform_geometry(3),
        &[partial_residency(1, &[2], 4)],
    )
    .expect("one fitting page should survive low-budget mode");

    assert_eq!(
        plan.layer_targets,
        vec![
            ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
            ExpertLayerResidencyTarget::PreservePartial,
            ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
        ]
    );
}

#[test]
fn should_preserve_a_fitting_partial_page_without_exact_set_equality() {
    let current_page = partial_residency(1, &[0, 3], 8);
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::GenerationPreparation,
        60,
        &uniform_geometry(3),
        &[current_page],
    )
    .expect("routed floors should preserve an already useful partial page");

    assert_eq!(
        plan.layer_targets[1],
        ExpertLayerResidencyTarget::PreservePartial
    );
    assert_eq!(plan.expected_preserved_bytes, 20);
}

#[test]
fn should_not_plan_eager_io_for_an_empty_partial_layer_without_route_evidence() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Decode,
        60,
        &uniform_geometry(3),
        &[],
    )
    .expect("routed floors should fit exactly");

    assert!(
        plan.layer_targets.iter().all(|target| {
            *target == ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
        })
    );
    assert_eq!(plan.maximum_new_retained_bytes, 60);
}

#[test]
fn should_release_low_coverage_partial_pages_before_any_complete_layer() {
    let current_residencies = vec![
        partial_residency(0, &[0], 10),
        partial_residency(1, &[1], 1),
        complete_residency(2),
    ];
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Idle,
        60,
        &uniform_geometry(3),
        &current_residencies,
    )
    .expect("current ownership should fit its ceiling");

    assert_eq!(plan.deterministic_release_order, vec![1, 0, 2]);
}

#[test]
fn should_release_complete_layers_only_for_the_remaining_exact_deficit() {
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Prefill,
        60,
        &uniform_geometry(3),
        &[complete_residency(2)],
    )
    .expect("routed floors should consume the full ceiling");

    assert!(plan.complete_layer_targets.is_empty());
    assert_eq!(
        plan.layer_targets[2],
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    );
}

#[test]
fn should_fail_closed_for_invalid_geometry_residency_and_overflow() {
    let duplicate_residencies = vec![complete_residency(0), complete_residency(0)];
    assert!(matches!(
        plan_phase_aware_expert_residency(
            ExpertResidencyPhase::Prefill,
            80,
            &uniform_geometry(2),
            &duplicate_residencies,
        ),
        Err(PhaseAwareExpertResidencyPlanError::DuplicateOrUnorderedCurrentLayer { .. })
    ));

    let mut zero_geometry = uniform_geometry(1);
    zero_geometry[0].expert_capacity = 0;
    assert!(matches!(
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 0, &zero_geometry, &[],),
        Err(PhaseAwareExpertResidencyPlanError::ZeroGeometry { .. })
    ));

    let overflowing_geometry = [ExpertLayerGeometry {
        layer_index: 0,
        complete_layer_payload_bytes: u64::MAX,
        expert_payload_bytes: u64::MAX,
        expert_capacity: 2,
        experts_per_token: 1,
    }];
    assert_eq!(
        plan_phase_aware_expert_residency(
            ExpertResidencyPhase::Prefill,
            u64::MAX,
            &overflowing_geometry,
            &[],
        ),
        Err(PhaseAwareExpertResidencyPlanError::ByteCountOverflow)
    );
}

#[test]
fn should_keep_every_planned_owner_and_reservation_within_the_retained_budget() {
    let geometries = uniform_geometry(4);
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Decode,
        100,
        &geometries,
        &[partial_residency(3, &[0, 2], 12)],
    )
    .expect("mixed foundation and overlay should fit");
    let complete_target_bytes =
        u64::try_from(plan.complete_layer_targets.len()).unwrap_or(u64::MAX) * 40;

    assert!(
        complete_target_bytes + plan.reserved_routed_overlay_bytes
            <= plan.retained_expert_ceiling_bytes
    );
}
