//! Replacement-aware complete expert residency policy.

use super::{
    ExpertMemoryAdmissionError, MemoryBoundary,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    projected_active_memory_after_complete_expert_replacement,
};

/// Exact ownership and headroom required by complete expert residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompleteResidencyRequirements {
    /// Live active bytes while paged expert ownership still exists.
    pub current_active_memory_bytes: u64,
    /// Paged payload replaced, rather than coexisting, with complete experts.
    pub retained_paged_expert_payload_bytes: u64,
    /// Exact complete expert payload derived from validated artifact geometry.
    pub complete_expert_payload_bytes: u64,
    /// Context, activation, and additional fixed headroom required after promotion.
    pub required_headroom_bytes: u64,
    /// Stable MLX active-memory ceiling.
    pub active_memory_ceiling_bytes: u64,
}

impl CompleteResidencyRequirements {
    #[must_use]
    pub fn decide(self) -> CompleteResidencyDecision {
        let projected_active_memory_bytes =
            match projected_active_memory_after_complete_expert_replacement(
                self.current_active_memory_bytes,
                self.retained_paged_expert_payload_bytes,
                self.complete_expert_payload_bytes,
            ) {
                Ok(projected_active_memory_bytes) => projected_active_memory_bytes,
                Err(error) => return CompleteResidencyDecision::RejectInvalidObservation { error },
            };
        if complete_residency_exceeds_ceiling_with_activation_headroom(
            projected_active_memory_bytes,
            self.active_memory_ceiling_bytes,
            self.required_headroom_bytes,
        ) {
            let projected_with_headroom_bytes = projected_active_memory_bytes
                .checked_add(self.required_headroom_bytes)
                .unwrap_or(u64::MAX);
            return CompleteResidencyDecision::DoesNotFit {
                boundary: MemoryBoundary::CompleteResidency,
                shortfall_bytes: projected_with_headroom_bytes
                    .saturating_sub(self.active_memory_ceiling_bytes),
                projected_active_memory_bytes,
                required_headroom_bytes: self.required_headroom_bytes,
            };
        }
        CompleteResidencyDecision::Admit {
            projected_active_memory_bytes,
            required_headroom_bytes: self.required_headroom_bytes,
        }
    }
}

/// Typed complete-residency policy result; execution performs no fit arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompleteResidencyDecision {
    Admit {
        projected_active_memory_bytes: u64,
        required_headroom_bytes: u64,
    },
    DoesNotFit {
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
        projected_active_memory_bytes: u64,
        required_headroom_bytes: u64,
    },
    RejectInvalidObservation {
        error: ExpertMemoryAdmissionError,
    },
}
