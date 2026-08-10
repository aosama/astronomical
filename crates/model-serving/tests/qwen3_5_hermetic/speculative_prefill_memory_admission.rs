#[path = "speculative_prefill_memory_admission_policy.rs"]
mod speculative_prefill_memory_admission;

use speculative_prefill_memory_admission::{
    speculative_prefill_draft_scoring_reclamation_target_bytes,
    speculative_prefill_draft_scoring_reservation_bytes,
};

#[test]
fn should_reserve_draft_decoder_state_vision_expert_page_boundary_and_publication_workspace() {
    assert_eq!(
        speculative_prefill_draft_scoring_reservation_bytes(900, 100, 75, 15, 10),
        Some(1_100),
    );
}

#[test]
fn should_not_reclaim_target_experts_when_draft_scoring_reservation_fits() {
    assert_eq!(
        speculative_prefill_draft_scoring_reclamation_target_bytes(900, 100, 1_000),
        0,
    );
}

#[test]
fn should_reclaim_only_the_target_expert_bytes_needed_for_draft_scoring() {
    assert_eq!(
        speculative_prefill_draft_scoring_reclamation_target_bytes(900, 125, 1_000),
        25,
    );
}

#[test]
fn should_treat_projection_overflow_as_requiring_all_available_target_expert_reclamation() {
    assert_eq!(
        speculative_prefill_draft_scoring_reclamation_target_bytes(usize::MAX - 4, 8, 1_000,),
        usize::MAX - 1_000,
    );
}
