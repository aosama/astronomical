use astronomical_model_serving::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyRequirements, ContextAdmissionRequirements, ExpertMemoryMode,
    ForwardRecoveryDecision, ForwardRecoveryRequirements, MemoryAdmissionDecision, MemoryBoundary,
    MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements, SpeculativePrefillAdmission,
    classify_expert_memory_mode, complete_residency_exceeds_ceiling_with_activation_headroom,
    persistent_context_restore_workspace_bytes, request_context_temporary_workspace_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
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
        sparse_experts_are_paged: false,
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
fn should_admit_decode_when_only_a_temporary_cache_restore_workspace_forced_demotion() {
    // A fitting resident model plus a large prompt-cache restore workspace can
    // exceed the ceiling even though decode itself still fits. Demote for the
    // restore, then admit once that workspace is gone.
    let restore_admission = ContextAdmissionRequirements {
        current_active_memory_bytes: 26_500_000_000,
        context_growth_bytes: 1_700_000_000,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: 8_100_000_000,
        retained_expert_payload_bytes: 21_600_000_000,
        active_memory_ceiling_bytes: 35_000_000_000,
        complete_experts_are_resident: true,
    }
    .decide();
    let decode_admission = ContextAdmissionRequirements {
        current_active_memory_bytes: 26_900_000_000,
        context_growth_bytes: 400_000_000,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: 0,
        retained_expert_payload_bytes: 21_600_000_000,
        active_memory_ceiling_bytes: 35_000_000_000,
        complete_experts_are_resident: true,
    }
    .decide();

    assert_eq!(
        restore_admission,
        MemoryAdmissionDecision::DemoteCompleteResidency {
            reassess_after_demotion: true,
        }
    );
    assert_eq!(decode_admission, MemoryAdmissionDecision::Admit);
}

#[test]
fn should_keep_complete_residency_when_cache_restore_overlap_excludes_output_budget() {
    // Live agent turn on resident sparse MoE at a 35 GB ceiling: 15,399 prompt tokens,
    // 65,535 max output, cache on, 21.59 GB experts already seated. Charging
    // restore overlap on prompt+output demoted; charging it on prompt tokens
    // only keeps complete residency.
    const CONTEXT_MEMORY_RESERVATION_BYTES_PER_TOKEN: usize = 20_480;
    const PROMPT_TOKEN_COUNT: usize = 15_399;
    const MAXIMUM_OUTPUT_TOKEN_COUNT: usize = 65_535;
    const CURRENT_ACTIVE_MEMORY_BYTES: usize = 26_488_988_932;
    const ACTIVE_MEMORY_CEILING_BYTES: usize = 35_000_000_000;
    const DIRECT_PUBLICATION_WORKSPACE_BYTES: usize = 3_407_872;
    const PREFILL_ACTIVATION_WORKSPACE_BYTES: usize = 4_831_838_208;
    const COMPLETE_LAYER_SCRATCH_BYTES: usize = 1_610_612_736;
    const RETAINED_EXPERT_PAYLOAD_BYTES: usize = 21_592_276_992;
    let total_context_token_count = PROMPT_TOKEN_COUNT + MAXIMUM_OUTPUT_TOKEN_COUNT;
    let context_growth_bytes = CONTEXT_MEMORY_RESERVATION_BYTES_PER_TOKEN
        .checked_mul(total_context_token_count)
        .expect("prompt plus output KV should fit usize");
    let restore_overlap_including_output_budget = persistent_context_restore_workspace_bytes(
        CONTEXT_MEMORY_RESERVATION_BYTES_PER_TOKEN,
        total_context_token_count,
    )
    .expect("output-inclusive restore overlap should fit usize");
    let restore_overlap_prompt_tokens_only = persistent_context_restore_workspace_bytes(
        CONTEXT_MEMORY_RESERVATION_BYTES_PER_TOKEN,
        PROMPT_TOKEN_COUNT,
    )
    .expect("prompt-only restore overlap should fit usize");
    let admission_with_output_budget_in_restore = ContextAdmissionRequirements {
        current_active_memory_bytes: CURRENT_ACTIVE_MEMORY_BYTES,
        context_growth_bytes,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: DIRECT_PUBLICATION_WORKSPACE_BYTES
            + restore_overlap_including_output_budget
            + PREFILL_ACTIVATION_WORKSPACE_BYTES
            + COMPLETE_LAYER_SCRATCH_BYTES,
        retained_expert_payload_bytes: RETAINED_EXPERT_PAYLOAD_BYTES,
        active_memory_ceiling_bytes: ACTIVE_MEMORY_CEILING_BYTES,
        complete_experts_are_resident: true,
    }
    .decide();
    let admission_with_prompt_only_restore = ContextAdmissionRequirements {
        current_active_memory_bytes: CURRENT_ACTIVE_MEMORY_BYTES,
        context_growth_bytes,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: DIRECT_PUBLICATION_WORKSPACE_BYTES
            + restore_overlap_prompt_tokens_only
            + PREFILL_ACTIVATION_WORKSPACE_BYTES
            + COMPLETE_LAYER_SCRATCH_BYTES,
        retained_expert_payload_bytes: RETAINED_EXPERT_PAYLOAD_BYTES,
        active_memory_ceiling_bytes: ACTIVE_MEMORY_CEILING_BYTES,
        complete_experts_are_resident: true,
    }
    .decide();

    assert_eq!(
        admission_with_output_budget_in_restore,
        MemoryAdmissionDecision::DemoteCompleteResidency {
            reassess_after_demotion: true,
        }
    );
    assert_eq!(
        admission_with_prompt_only_restore,
        MemoryAdmissionDecision::Admit
    );
}

#[test]
fn should_take_the_larger_exclusive_request_phase_instead_of_summing_them() {
    // Live 33 GB turn: restore 0.32 GB and serving KV 1.67 GB do not coexist.
    const CURRENT_ACTIVE_MEMORY_BYTES: usize = 26_488_988_934;
    const CONTEXT_GROWTH_BYTES: usize = 1_665_556_480;
    const RESTORE_OVERLAP_WORKSPACE_BYTES: usize = 323_399_680;
    const PUBLICATION_WORKSPACE_BYTES: usize = 3_407_872;
    let stacked_exclusive_peaks_bytes = CURRENT_ACTIVE_MEMORY_BYTES
        + CONTEXT_GROWTH_BYTES
        + RESTORE_OVERLAP_WORKSPACE_BYTES
        + PUBLICATION_WORKSPACE_BYTES
        + 4_831_838_208
        + 1_610_612_736;
    let concurrent_peak_bytes = seated_complete_expert_request_peak_active_memory_bytes(
        CURRENT_ACTIVE_MEMORY_BYTES,
        CONTEXT_GROWTH_BYTES,
        RESTORE_OVERLAP_WORKSPACE_BYTES,
        PUBLICATION_WORKSPACE_BYTES,
    )
    .expect("the seated peak should fit usize");

    assert_eq!(
        concurrent_peak_bytes,
        CURRENT_ACTIVE_MEMORY_BYTES + CONTEXT_GROWTH_BYTES + PUBLICATION_WORKSPACE_BYTES
    );
    assert!(concurrent_peak_bytes <= 33_000_000_000);
    assert!(stacked_exclusive_peaks_bytes > 33_000_000_000);
}

#[test]
fn should_admit_seated_complete_experts_at_a_33_gb_ceiling_without_stacked_layer_weight_headroom() {
    const CURRENT_ACTIVE_MEMORY_BYTES: usize = 26_488_988_934;
    const CONTEXT_GROWTH_BYTES: usize = 1_665_556_480;
    const RESTORE_OVERLAP_WORKSPACE_BYTES: usize = 323_399_680;
    const PUBLICATION_WORKSPACE_BYTES: usize = 3_407_872;
    const RETAINED_EXPERT_PAYLOAD_BYTES: usize = 21_592_276_992;
    const ACTIVE_MEMORY_CEILING_BYTES: usize = 33_000_000_000;
    let temporary_workspace_bytes = seated_complete_expert_request_temporary_workspace_bytes(
        CONTEXT_GROWTH_BYTES,
        RESTORE_OVERLAP_WORKSPACE_BYTES,
        PUBLICATION_WORKSPACE_BYTES,
    )
    .expect("the seated temporary workspace should fit usize");

    assert_eq!(temporary_workspace_bytes, PUBLICATION_WORKSPACE_BYTES);
    assert_eq!(
        ContextAdmissionRequirements {
            current_active_memory_bytes: CURRENT_ACTIVE_MEMORY_BYTES,
            context_growth_bytes: CONTEXT_GROWTH_BYTES,
            expert_page_reservation_bytes: 0,
            temporary_workspace_bytes,
            retained_expert_payload_bytes: RETAINED_EXPERT_PAYLOAD_BYTES,
            active_memory_ceiling_bytes: ACTIVE_MEMORY_CEILING_BYTES,
            complete_experts_are_resident: true,
        }
        .decide(),
        MemoryAdmissionDecision::Admit
    );
}

#[test]
fn should_admit_complete_residency_without_prefill_three_layer_weight_headroom() {
    const PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES: u64 = 28_193_105_290;
    const ACTIVE_MEMORY_CEILING_BYTES: u64 = 33_000_000_000;
    const GATE_UP_FUSION_TRANSIENT_PAYLOAD_BYTES: u64 = 1_073_741_824;
    const PREFILL_THREE_LAYER_WEIGHT_HEADROOM_BYTES: u64 = 4_831_838_208;

    assert!(
        !complete_residency_exceeds_ceiling_with_activation_headroom(
            PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES,
            ACTIVE_MEMORY_CEILING_BYTES,
            GATE_UP_FUSION_TRANSIENT_PAYLOAD_BYTES,
        )
    );
    assert!(complete_residency_exceeds_ceiling_with_activation_headroom(
        PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES,
        ACTIVE_MEMORY_CEILING_BYTES,
        PREFILL_THREE_LAYER_WEIGHT_HEADROOM_BYTES,
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

#[test]
fn should_classify_complete_owner_as_resident_and_empty_pager_cache_as_paged() {
    assert_eq!(
        classify_expert_memory_mode(true, true, 0),
        ExpertMemoryMode::Resident
    );
    assert_eq!(
        classify_expert_memory_mode(false, false, 0),
        ExpertMemoryMode::Resident
    );
    assert_eq!(
        classify_expert_memory_mode(false, true, 0),
        ExpertMemoryMode::Paged
    );
    assert_eq!(
        classify_expert_memory_mode(false, true, 15_854_469_120),
        ExpertMemoryMode::Hybrid
    );
}

#[test]
fn should_ignore_layer_weight_workspace_when_complete_experts_are_already_seated() {
    let seated_temporary_workspace_bytes = request_context_temporary_workspace_bytes(
        true,
        1_665_556_480,
        323_399_680,
        3_407_872,
        4_831_838_208,
        1_610_612_736,
    )
    .expect("seated workspace should fit usize");
    let paged_temporary_workspace_bytes = request_context_temporary_workspace_bytes(
        false,
        1_665_556_480,
        323_399_680,
        3_407_872,
        4_831_838_208,
        1_610_612_736,
    )
    .expect("paged workspace should fit usize");

    assert_eq!(seated_temporary_workspace_bytes, 3_407_872);
    assert_eq!(
        paged_temporary_workspace_bytes,
        3_407_872 + 323_399_680 + 4_831_838_208 + 1_610_612_736
    );
}
