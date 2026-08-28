use astronomical_model_serving::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPhase, ForwardRecoveryDecision, ForwardRecoveryRequirements,
    PhaseAwareExpertResidencyPlanError, RequestExpertLayerRole, RequestExpertResidency,
    RetainedExpertPageClass, plan_phase_aware_expert_residency,
    publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_enact_planned_expert_release,
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

#[test]
fn should_keep_complete_layers_during_prefill_even_when_the_plan_names_a_release() {
    let geometries = (0..3)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect::<Vec<_>>();
    let plan = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Prefill,
        60,
        &geometries,
        &[CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        }],
    )
    .expect("routed floors should consume the full ceiling");

    assert_eq!(
        plan.layer_targets[2],
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    );
    assert!(!should_enact_planned_expert_release(
        ExpertResidencyPhase::Prefill,
        plan.layer_targets[2],
    ));
    assert!(!should_enact_planned_expert_release(
        ExpertResidencyPhase::GenerationPreparation,
        plan.layer_targets[2],
    ));
    assert!(should_enact_planned_expert_release(
        ExpertResidencyPhase::Idle,
        plan.layer_targets[2],
    ));
}

#[test]
fn should_seat_complete_layers_on_mandatory_prefill_reads() {
    assert!(should_commit_mandatory_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
    ));
    assert!(should_commit_mandatory_complete_layer(
        1,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
    ));
    assert!(!should_commit_mandatory_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
    ));
    assert!(should_commit_mandatory_routed_page(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
        false,
    ));
    assert!(should_commit_mandatory_routed_page(
        1,
        true,
        Some(ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead),
        true,
    ));
    assert!(!should_commit_mandatory_routed_page(
        2_048,
        false,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
        true,
    ));
}

#[test]
fn should_refuse_the_same_prefill_chunk_retry_when_sparse_experts_are_paged() {
    let resident_retry = ForwardRecoveryRequirements {
        stable_active_memory_bytes: 900,
        active_memory_bytes_at_failure: 950,
        attempted_allocation_bytes: 100,
        observed_transient_high_water_bytes: 25,
        retained_expert_payload_bytes_before_reclamation: 200,
        retained_expert_payload_bytes_after_reclamation: 100,
        active_memory_ceiling_bytes: 1_000,
        has_already_retried_after_reclamation: false,
        sparse_experts_are_paged: false,
    };
    let paged_retry = ForwardRecoveryRequirements {
        sparse_experts_are_paged: true,
        ..resident_retry
    };

    assert!(matches!(
        resident_retry.decide(),
        ForwardRecoveryDecision::Retry { .. }
    ));
    assert!(matches!(
        paged_retry.decide(),
        ForwardRecoveryDecision::Reject { .. }
    ));
}

#[test]
fn should_keep_already_seated_complete_layers_when_leftover_budget_tightens() {
    let seated_complete_layer_payload_bytes = 80;
    let tighter_leftover_expert_budget_bytes = 70;
    let richer_leftover_expert_budget_bytes = 90;
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(
            tighter_leftover_expert_budget_bytes,
            seated_complete_layer_payload_bytes,
        ),
        seated_complete_layer_payload_bytes,
        "learned context reserve must not evict a complete layer this request already seated"
    );
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(
            richer_leftover_expert_budget_bytes,
            seated_complete_layer_payload_bytes,
        ),
        richer_leftover_expert_budget_bytes,
        "a richer leftover budget must still be able to seat additional complete layers"
    );
    assert_eq!(
        retained_complete_layer_ceiling_after_prefill_budget_refresh(0, 0),
        0
    );
}

#[test]
fn should_plan_prefill_with_the_floored_ceiling_when_seated_layers_exceed_leftover() {
    let geometries = uniform_geometry(3);
    let seated_complete_layers = [
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
    let leftover_expert_budget_bytes = 70;
    let seated_complete_layer_payload_bytes = 80;
    assert!(
        matches!(
            plan_phase_aware_expert_residency(
                ExpertResidencyPhase::Prefill,
                leftover_expert_budget_bytes,
                &geometries,
                &seated_complete_layers,
            ),
            Err(PhaseAwareExpertResidencyPlanError::CurrentResidencyExceedsCeiling),
        ),
        "planning against the tighter leftover number must not silently drop seated layers"
    );
    let floored_ceiling_bytes = retained_complete_layer_ceiling_after_prefill_budget_refresh(
        leftover_expert_budget_bytes,
        seated_complete_layer_payload_bytes,
    );
    plan_phase_aware_expert_residency(
        ExpertResidencyPhase::Prefill,
        floored_ceiling_bytes,
        &geometries,
        &seated_complete_layers,
    )
    .expect("the floored Prefill ceiling must accept already-seated complete layers");
}

#[test]
fn should_keep_the_opening_prefill_pin_set_when_later_leftover_wants_more_layers() {
    let geometries = uniform_geometry(3);
    let opening_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 100, &geometries, &[])
            .expect("two complete layers should fit the opening leftover");
    let (opened_residency, opened_plan) = publish_request_stable_residency_plan(
        ExpertResidencyPhase::Prefill,
        None,
        opening_candidate,
        &[],
        0,
        &geometries,
    );
    let opened_residency = opened_residency.expect("Prefill must open a request contract");
    assert_eq!(
        opened_residency.layer_role(0),
        Some(RequestExpertLayerRole::PinnedComplete)
    );
    assert_eq!(
        opened_residency.layer_role(2),
        Some(RequestExpertLayerRole::Streamed)
    );
    assert_eq!(
        opened_plan.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );

    let richer_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 120, &geometries, &[])
            .expect("the full model should fit a richer leftover");
    let (_continued_residency, continued_plan) = publish_request_stable_residency_plan(
        ExpertResidencyPhase::Prefill,
        Some(&opened_residency),
        richer_candidate,
        &[],
        0,
        &geometries,
    );

    assert_eq!(continued_plan.complete_layer_targets, vec![0, 1]);
    assert_eq!(
        continued_plan.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
}

#[test]
fn should_keep_opening_prefill_pins_when_later_leftover_is_tighter_without_capacity_failure() {
    let geometries = uniform_geometry(3);
    let opening_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 100, &geometries, &[])
            .expect("two complete layers should fit the opening leftover");
    let opened_residency = RequestExpertResidency::open_prefill(&opening_candidate);
    assert_eq!(opened_residency.pinned_complete_layer_indexes(), vec![0, 1]);

    let tighter_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 70, &geometries, &[])
            .expect("one complete layer should still fit a tighter leftover");
    let (continued_residency, continued_plan) = publish_request_stable_residency_plan(
        ExpertResidencyPhase::Prefill,
        Some(&opened_residency),
        tighter_candidate,
        &[],
        0,
        &geometries,
    );
    let continued_residency = continued_residency.expect("Prefill must keep the request contract");

    assert_eq!(
        continued_residency.pinned_complete_layer_indexes(),
        vec![0, 1],
        "learned leftover tightening is not a capacity failure and must not unpin seated layers"
    );
    assert_eq!(continued_plan.complete_layer_targets, vec![0, 1]);
    assert_eq!(
        continued_plan.layer_targets[1],
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
    );
}

#[test]
fn should_not_re_pin_a_layer_after_a_prefill_capacity_failure() {
    let geometries = uniform_geometry(3);
    let opening_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 100, &geometries, &[])
            .expect("two complete layers should fit");
    let opened_residency = RequestExpertResidency::open_prefill(&opening_candidate);
    let shrunk_residency = opened_residency.shrink_after_capacity_failure(40, &geometries);
    assert_eq!(
        shrunk_residency.layer_role(1),
        Some(RequestExpertLayerRole::Streamed)
    );

    let richer_candidate =
        plan_phase_aware_expert_residency(ExpertResidencyPhase::Prefill, 120, &geometries, &[])
            .expect("leftover after failure still wants every layer");
    let (_published_residency, published_plan) = publish_request_stable_residency_plan(
        ExpertResidencyPhase::Prefill,
        Some(&shrunk_residency),
        richer_candidate,
        &[],
        0,
        &geometries,
    );

    assert_eq!(published_plan.complete_layer_targets, vec![0]);
    assert_eq!(
        published_plan.layer_targets[1],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
}

#[test]
fn should_keep_prefill_routed_pages_when_generation_handoff_replans() {
    let geometries = uniform_geometry(3);
    let prefill_routed_pages = [
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1],
            payload_bytes: 20,
            covered_weighted_demand: 8,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 2],
            payload_bytes: 20,
            covered_weighted_demand: 6,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![1, 3],
            payload_bytes: 20,
            covered_weighted_demand: 4,
        },
    ];
    let generation_candidate = plan_phase_aware_expert_residency(
        ExpertResidencyPhase::GenerationPreparation,
        120,
        &geometries,
        &prefill_routed_pages,
    )
    .expect("generation leftover should keep every prefill routed page");
    let (generation_residency, generation_plan) = publish_request_stable_residency_plan(
        ExpertResidencyPhase::GenerationPreparation,
        None,
        generation_candidate,
        &prefill_routed_pages,
        0,
        &geometries,
    );

    assert!(generation_residency.is_none());
    assert!(generation_plan.complete_layer_targets.is_empty());
    assert!(
        generation_plan
            .layer_targets
            .iter()
            .all(|target| { *target == ExpertLayerResidencyTarget::PreservePartial })
    );
}
