/// Returns the target expert-retention payload that must be reclaimed before
/// drafter scoring can allocate its remaining decoder state and one expert page.
#[must_use]
pub(crate) const fn speculative_prefill_draft_scoring_reclamation_target_bytes(
    current_active_memory_bytes: usize,
    draft_scoring_reservation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> usize {
    current_active_memory_bytes
        .saturating_add(draft_scoring_reservation_bytes)
        .saturating_sub(allowed_active_memory_bytes)
}

/// Combines the independently owned drafter allocations required before its
/// scoring graph can run.
#[must_use]
pub(crate) fn speculative_prefill_draft_scoring_reservation_bytes(
    draft_decoder_state_growth_bytes: usize,
    draft_vision_payload_bytes: usize,
    draft_maximum_expert_page_reservation_bytes: usize,
    draft_boundary_checkpoint_workspace_bytes: usize,
    draft_direct_publication_workspace_bytes: usize,
) -> Option<usize> {
    draft_decoder_state_growth_bytes
        .checked_add(draft_vision_payload_bytes)?
        .checked_add(draft_maximum_expert_page_reservation_bytes)?
        .checked_add(draft_boundary_checkpoint_workspace_bytes)?
        .checked_add(draft_direct_publication_workspace_bytes)
}
