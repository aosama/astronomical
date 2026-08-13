use astronomical_model_serving::SpeculativePrefillAdmission;

#[test]
fn should_keep_resident_target_experts_when_the_drafter_payload_fits_the_remaining_capacity() {
    assert!(
        SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
            23_000_000_000,
            3_000_000_000,
            32_320_000_000,
        )
    );
}

#[test]
fn should_release_resident_target_experts_when_the_drafter_payload_exceeds_capacity() {
    assert!(
        !SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
            30_000_000_000,
            3_000_000_000,
            32_320_000_000,
        )
    );
}

#[test]
fn should_release_resident_target_experts_when_the_combined_projection_overflows() {
    assert!(
        !SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
            usize::MAX,
            1,
            usize::MAX,
        )
    );
}

#[test]
fn should_reserve_draft_decoder_state_vision_expert_page_boundary_and_publication_workspace() {
    assert_eq!(
        SpeculativePrefillAdmission::draft_scoring_reservation_bytes(900, 100, 75, 15, 10),
        Some(1_100),
    );
}

#[test]
fn should_not_reclaim_target_experts_when_draft_scoring_reservation_fits() {
    assert_eq!(
        SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(900, 100, 1_000),
        0,
    );
}

#[test]
fn should_reclaim_only_the_target_expert_bytes_needed_for_draft_scoring() {
    assert_eq!(
        SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(900, 125, 1_000),
        25,
    );
}

#[test]
fn should_treat_projection_overflow_as_requiring_all_available_target_expert_reclamation() {
    assert_eq!(
        SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
            usize::MAX - 4,
            8,
            1_000,
        ),
        usize::MAX,
    );
}

#[test]
fn should_reject_draft_loading_when_reclamation_still_leaves_insufficient_capacity() {
    assert!(
        !SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
            30_000_000_000,
            3_000_000_000,
            32_320_000_000,
        )
    );
}
