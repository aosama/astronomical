use crate::reclamation::{
    ExpertMemoryAdmissionError, complete_residency_exceeds_ceiling_with_activation_headroom,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
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
