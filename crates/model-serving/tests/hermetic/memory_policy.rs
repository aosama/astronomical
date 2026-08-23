use astronomical_model_serving::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyRequirements, ForwardRecoveryDecision, ForwardRecoveryRequirements,
    MemoryBoundary, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    SpeculativePrefillAdmission,
};

#[test]
fn should_identify_the_allocation_boundary_and_required_shortfall() {
    let observation = AllocationAdmissionObservation::new(900, 0, 125, 1_000);

    assert_eq!(
        observation.decide(),
        AllocationAdmissionDecision::Reject {
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: 25,
        }
    );
}

#[test]
fn should_plan_demotion_before_installing_a_lower_ceiling() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 25_000,
        requested_ceiling_bytes: 23_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 24_000,
        retained_paged_expert_payload_bytes: 20_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: true,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_demote_complete_residency_when_only_the_idle_snapshot_fits() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 30_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: true,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_keep_complete_residency_when_idle_memory_and_serving_headroom_fit() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 32_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_preserve_paged_reclamation_when_complete_residency_headroom_is_irrelevant() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 30_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 33_000,
        retained_paged_expert_payload_bytes: 2_000,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: u64::MAX,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 2_000,
        }
    );
}

#[test]
fn should_not_request_demotion_for_raised_or_unchanged_ceilings() {
    let requirements = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 30_000,
        requested_ceiling_bytes: 38_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: u64::MAX,
    };

    assert_eq!(
        requirements.decide(),
        MemoryCeilingChangeDecision::Raise {
            may_attempt_complete_residency: true,
        }
    );
    assert_eq!(
        MemoryCeilingChangeRequirements {
            requested_ceiling_bytes: requirements.current_ceiling_bytes,
            ..requirements
        }
        .decide(),
        MemoryCeilingChangeDecision::Unchanged
    );
}

#[test]
fn should_request_allocator_cleanup_without_confusing_cache_with_active_memory() {
    let observation = AllocationAdmissionObservation::new(800, 300, 100, 1_000);

    assert_eq!(
        observation.decide(),
        AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit
    );
}

#[test]
fn should_decide_complete_residency_from_one_replacement_aware_policy() {
    let requirements = CompleteResidencyRequirements {
        current_active_memory_bytes: 2_000,
        retained_paged_expert_payload_bytes: 600,
        complete_expert_payload_bytes: 1_000,
        required_headroom_bytes: 500,
        active_memory_ceiling_bytes: 3_000,
    };

    assert_eq!(
        requirements.decide(),
        CompleteResidencyDecision::Admit {
            projected_active_memory_bytes: 2_400,
            required_headroom_bytes: 500,
        }
    );
}

#[test]
fn should_authorize_only_one_retry_after_required_experts_were_reclaimed() {
    let requirements = ForwardRecoveryRequirements {
        stable_active_memory_bytes: 900,
        active_memory_bytes_at_failure: 950,
        attempted_allocation_bytes: 100,
        observed_transient_high_water_bytes: 25,
        retained_expert_payload_bytes_before_reclamation: 200,
        retained_expert_payload_bytes_after_reclamation: 100,
        active_memory_ceiling_bytes: 1_000,
        has_already_retried_after_reclamation: false,
    };

    assert_eq!(
        requirements.decide(),
        ForwardRecoveryDecision::Retry {
            fixed_forward_workspace_bytes: 150,
            required_reclamation_bytes: 50,
        }
    );
    assert!(matches!(
        ForwardRecoveryRequirements {
            has_already_retried_after_reclamation: true,
            ..requirements
        }
        .decide(),
        ForwardRecoveryDecision::Reject { .. }
    ));
}

#[test]
fn should_fail_speculative_prefill_closed_when_combined_owners_overflow() {
    assert_eq!(
        SpeculativePrefillAdmission::draft_scoring_reservation_bytes(usize::MAX, 1, 0, 0, 0,),
        None
    );
    assert_eq!(
        SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
            usize::MAX,
            1,
            usize::MAX,
        ),
        usize::MAX
    );
}
