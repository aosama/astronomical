use astronomical_model_serving::{
    ExpertMemoryAdmissionError, ExpertReclamationPlan,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
    should_retry_fixed_forward_after_expert_reclamation,
};

#[test]
fn should_project_complete_residency_as_replacement() {
    assert_eq!(
        projected_active_memory_after_complete_expert_replacement(30, 20, 25),
        Ok(35)
    );
    assert_eq!(
        projected_active_memory_after_complete_expert_replacement(9, 10, 1),
        Err(ExpertMemoryAdmissionError::RetainedExpertPayloadExceedsActiveMemory)
    );
    assert_eq!(
        projected_active_memory_after_complete_expert_replacement(u64::MAX, 0, 1),
        Err(ExpertMemoryAdmissionError::CompleteResidencyProjectionOverflow)
    );
}

#[test]
fn should_reserve_activation_headroom_before_complete_residency() {
    assert_eq!(
        required_complete_residency_activation_headroom_bytes(900_000_000, 0),
        900_000_000
    );
    assert_eq!(
        required_complete_residency_activation_headroom_bytes(900_000_000, 5_000_000_000),
        5_000_000_000
    );
    assert!(complete_residency_exceeds_ceiling_with_activation_headroom(
        38_000_000_000,
        39_000_000_000,
        3_600_000_000,
    ));
}

#[test]
fn should_reclaim_only_the_expert_bytes_needed_by_a_fixed_forward() {
    assert_eq!(
        expert_reclamation_bytes_to_fit_fixed_forward(
            30_000_000_000,
            25_000_000_000,
            32_000_000_000,
            4_000_000_000,
        ),
        2_000_000_000
    );
    assert_eq!(
        expert_reclamation_bytes_to_fit_fixed_forward(
            20_000_000_000,
            10_000_000_000,
            32_000_000_000,
            4_000_000_000,
        ),
        0
    );
}

#[test]
fn should_include_active_transients_in_failed_forward_workspace() {
    assert_eq!(
        fixed_forward_workspace_after_allocation_failure(
            36_000_000_000,
            38_000_000_000,
            800_000_000,
            1_000_000_000,
        ),
        2_800_000_000
    );
}

#[test]
fn should_retry_once_only_after_the_reclamation_target_was_released() {
    assert!(should_retry_fixed_forward_after_expert_reclamation(
        false, 1_000, 700, 300,
    ));
    assert!(!should_retry_fixed_forward_after_expert_reclamation(
        true, 1_000, 700, 300,
    ));
    assert!(!should_retry_fixed_forward_after_expert_reclamation(
        false, 1_000, 800, 300,
    ));
}

#[test]
fn should_choose_the_largest_memory_boundary_deficit() {
    let plan = ExpertReclamationPlan::for_projected_memory(110, 130, 125, 100, 120, 15);
    assert_eq!(plan.required_reclamation_bytes(), 10);
    assert_eq!(plan.reclamation_target_bytes(), 10);
    assert_eq!(plan.unresolved_shortfall_bytes(), 0);
    assert!(plan.can_satisfy_every_memory_boundary());
}

#[test]
fn should_report_unresolved_reclamation_shortfall() {
    let plan = ExpertReclamationPlan::for_projected_memory(140, 150, 145, 100, 120, 20);
    assert_eq!(plan.required_reclamation_bytes(), 40);
    assert_eq!(plan.reclamation_target_bytes(), 20);
    assert_eq!(plan.unresolved_shortfall_bytes(), 20);
    assert!(!plan.can_satisfy_every_memory_boundary());
}
