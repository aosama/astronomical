//! Memory policy for request-scoped speculative-prefill draft ownership.

/// Namespace owner for speculative-prefill memory decisions.
pub struct SpeculativePrefillAdmission;

impl SpeculativePrefillAdmission {
    /// Returns target expert retention that must yield before a draft allocation.
    #[must_use]
    pub const fn required_target_expert_reclamation_bytes(
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

    /// Returns whether a request-scoped draft model fits beside the target model.
    #[must_use]
    pub const fn draft_load_fits_with_target_active_memory(
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

    /// Combines every independently owned allocation needed for draft scoring.
    #[must_use]
    pub fn draft_scoring_reservation_bytes(
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
}
