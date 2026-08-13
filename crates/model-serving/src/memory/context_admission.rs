//! Pure context-growth requirements and admission projections.

use super::{MemoryAdmissionDecision, MemoryBoundary};

/// Exact context and workspace categories supplied by execution owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextAdmissionRequirements {
    /// Live MLX active bytes sampled before admitting context growth.
    pub current_active_memory_bytes: usize,
    /// Exact persistent decoder-state growth required by the operation.
    pub context_growth_bytes: usize,
    /// Largest expert page that may coexist with the context operation.
    pub expert_page_reservation_bytes: usize,
    /// Explicit temporary owner, such as prompt-cache reconstruction workspace.
    pub temporary_workspace_bytes: usize,
    /// Elastic paged-expert payload available for reclamation.
    pub retained_expert_payload_bytes: usize,
    /// Stable MLX active-memory ceiling resolved for the worker.
    pub active_memory_ceiling_bytes: usize,
    /// Whether the current expert owner is indivisible and must demote first.
    pub complete_experts_are_resident: bool,
}

impl ContextAdmissionRequirements {
    /// Returns the complete active-memory projection or `None` on overflow.
    #[must_use]
    pub fn projected_active_memory_bytes(self) -> Option<usize> {
        self.current_active_memory_bytes
            .checked_add(self.context_growth_bytes)?
            .checked_add(self.expert_page_reservation_bytes)?
            .checked_add(self.temporary_workspace_bytes)
    }

    /// Decides admission without mutating expert ownership.
    #[must_use]
    pub fn decide(self) -> MemoryAdmissionDecision {
        let Some(projected_active_memory_bytes) = self.projected_active_memory_bytes() else {
            return MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: u64::MAX,
            };
        };
        if projected_active_memory_bytes <= self.active_memory_ceiling_bytes {
            return MemoryAdmissionDecision::Admit;
        }
        if self.complete_experts_are_resident {
            return MemoryAdmissionDecision::DemoteCompleteResidency {
                reassess_after_demotion: true,
            };
        }
        let required_bytes =
            projected_active_memory_bytes.saturating_sub(self.active_memory_ceiling_bytes);
        if required_bytes <= self.retained_expert_payload_bytes {
            MemoryAdmissionDecision::Reclaim {
                required_bytes: u64::try_from(required_bytes).unwrap_or(u64::MAX),
            }
        } else {
            MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: u64::try_from(
                    required_bytes.saturating_sub(self.retained_expert_payload_bytes),
                )
                .unwrap_or(u64::MAX),
            }
        }
    }
}

/// Context-cache reconstruction temporarily owns both loaded and concatenated state.
#[must_use]
pub fn persistent_context_restore_workspace_bytes(
    context_memory_reservation_bytes_per_token: usize,
    restored_context_token_count: usize,
) -> Option<usize> {
    context_memory_reservation_bytes_per_token.checked_mul(restored_context_token_count)
}

/// Combines independently owned persistent growth categories.
#[must_use]
pub fn combined_persistent_growth_bytes(
    target_persistent_state_growth_bytes: usize,
    additional_persistent_state_growth_bytes: usize,
) -> Option<usize> {
    target_persistent_state_growth_bytes.checked_add(additional_persistent_state_growth_bytes)
}

/// Smallest safe idle ceiling after reclaiming elastic expert payload.
#[must_use]
pub const fn safe_minimum_active_memory_ceiling_bytes(
    current_idle_active_memory_bytes: u64,
    evictable_retained_expert_payload_bytes: u64,
    maximum_expert_page_reserve_bytes: u64,
) -> u64 {
    current_idle_active_memory_bytes
        .saturating_sub(evictable_retained_expert_payload_bytes)
        .saturating_add(maximum_expert_page_reserve_bytes)
}
