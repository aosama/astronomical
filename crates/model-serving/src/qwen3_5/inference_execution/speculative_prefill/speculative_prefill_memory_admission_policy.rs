//! Pure memory projections for request-scoped drafter loading and scoring.

/// Returns target expert retention that must yield before a draft allocation.
/// Arithmetic overflow requests maximal reclamation so admission fails closed.
#[must_use]
pub(crate) const fn speculative_prefill_required_target_expert_reclamation_bytes(
    current_active_memory_bytes: usize,
    draft_allocation_reservation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> usize {
    match current_active_memory_bytes.checked_add(draft_allocation_reservation_bytes) {
        Some(projected_active_memory_bytes) => {
            projected_active_memory_bytes.saturating_sub(allowed_active_memory_bytes)
        }
        None => usize::MAX,
    }
}

/// Returns whether a request-scoped drafter fits beside current target memory.
#[must_use]
pub(crate) const fn speculative_prefill_draft_load_fits_with_target_active_memory(
    current_target_active_memory_bytes: usize,
    draft_artifact_payload_bytes: usize,
    stable_active_memory_ceiling_bytes: usize,
) -> bool {
    match current_target_active_memory_bytes.checked_add(draft_artifact_payload_bytes) {
        Some(combined_active_memory_bytes) => {
            combined_active_memory_bytes <= stable_active_memory_ceiling_bytes
        }
        None => false,
    }
}

/// Combines the independently owned drafter allocations required before its
/// scoring graph can run.
///
/// Decoder growth and visual payload are long-lived for scoring. The expert page,
/// boundary checkpoint, and direct-publication workspace are transient but may
/// overlap at a cache boundary, so admission must reserve all five categories.
#[must_use]
pub(crate) fn speculative_prefill_draft_scoring_reservation_bytes(
    draft_decoder_state_growth_bytes: usize,
    draft_vision_payload_bytes: usize,
    draft_maximum_expert_page_reservation_bytes: usize,
    draft_boundary_checkpoint_workspace_bytes: usize,
    draft_direct_publication_workspace_bytes: usize,
) -> Option<usize> {
    // Overflow means admission cannot prove the request fits and must fail
    // closed. `Option` keeps the pure formula independent of engine error types.
    draft_decoder_state_growth_bytes
        .checked_add(draft_vision_payload_bytes)?
        .checked_add(draft_maximum_expert_page_reservation_bytes)?
        .checked_add(draft_boundary_checkpoint_workspace_bytes)?
        .checked_add(draft_direct_publication_workspace_bytes)
}
