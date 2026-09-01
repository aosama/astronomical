//! Policy for changing the live MLX active-memory ceiling.
//!
//! Question answered: may the worker raise or lower the ceiling right now, and
//! what ownership must demote or reclaim first? Families supply the current
//! owner facts; this module returns a `MemoryCeilingChangeDecision` that the
//! family enacts against the MLX native limit. Raising and lowering are not
//! symmetric: raising lets MLX accept capacity before Rust publishes a larger
//! budget, lowering reclaims before MLX enforces the smaller limit.

use super::MemoryBoundary;

/// Ownership evidence used before changing a process-wide MLX ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryCeilingChangeRequirements {
    /// Ceiling currently installed in the MLX runtime and memory policy owners.
    pub current_ceiling_bytes: u64,
    /// User-requested replacement ceiling.
    pub requested_ceiling_bytes: u64,
    /// Smallest ceiling that preserves non-evictable model and page ownership.
    pub minimum_safe_ceiling_bytes: u64,
    /// Live active bytes sampled before planning the transition.
    pub current_active_memory_bytes: u64,
    /// Elastic paged-expert bytes that a lower ceiling may reclaim.
    pub retained_paged_expert_payload_bytes: u64,
    /// Whether expert ownership must first transition from complete to paged.
    pub complete_experts_are_resident: bool,
    /// Request workspace that must remain available beside complete experts.
    pub complete_residency_required_headroom_bytes: u64,
}

impl MemoryCeilingChangeRequirements {
    #[must_use]
    pub const fn decide(self) -> MemoryCeilingChangeDecision {
        if self.requested_ceiling_bytes < self.minimum_safe_ceiling_bytes {
            return MemoryCeilingChangeDecision::Reject {
                boundary: MemoryBoundary::LiveCeilingMinimum,
                shortfall_bytes: self
                    .minimum_safe_ceiling_bytes
                    .saturating_sub(self.requested_ceiling_bytes),
            };
        }
        if self.requested_ceiling_bytes == self.current_ceiling_bytes {
            return MemoryCeilingChangeDecision::Unchanged;
        }
        if self.requested_ceiling_bytes > self.current_ceiling_bytes {
            return MemoryCeilingChangeDecision::Raise {
                may_attempt_complete_residency: true,
            };
        }
        let required_reclamation_bytes = self
            .current_active_memory_bytes
            .saturating_sub(self.requested_ceiling_bytes);
        let complete_residency_projection_bytes = self
            .current_active_memory_bytes
            .saturating_add(self.complete_residency_required_headroom_bytes);
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: self.complete_experts_are_resident
                && complete_residency_projection_bytes > self.requested_ceiling_bytes,
            retained_paged_expert_reclamation_bytes: if self.complete_experts_are_resident {
                0
            } else if required_reclamation_bytes < self.retained_paged_expert_payload_bytes {
                required_reclamation_bytes
            } else {
                self.retained_paged_expert_payload_bytes
            },
        }
    }
}

/// Typed sequencing advice for a live ceiling change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryCeilingChangeDecision {
    Unchanged,
    Raise {
        may_attempt_complete_residency: bool,
    },
    Lower {
        must_demote_complete_residency: bool,
        retained_paged_expert_reclamation_bytes: u64,
    },
    Reject {
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
    },
}
