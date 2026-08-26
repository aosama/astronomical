//! Policy for one unchanged forward retry after typed allocation pressure.

use super::{
    MemoryBoundary, expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    should_retry_fixed_forward_after_expert_reclamation,
};

/// Stateless policy boundary used before and after the executor reclaims experts.
pub struct ForwardRecoveryPolicy;

impl ForwardRecoveryPolicy {
    #[must_use]
    pub const fn fixed_workspace_bytes(
        stable_active_memory_bytes: usize,
        active_memory_bytes_at_failure: usize,
        attempted_allocation_bytes: usize,
        observed_transient_high_water_bytes: usize,
    ) -> usize {
        fixed_forward_workspace_after_allocation_failure(
            stable_active_memory_bytes,
            active_memory_bytes_at_failure,
            attempted_allocation_bytes,
            observed_transient_high_water_bytes,
        )
    }

    #[must_use]
    pub const fn required_reclamation_bytes(
        stable_active_memory_bytes: usize,
        retained_expert_payload_bytes: usize,
        active_memory_ceiling_bytes: usize,
        fixed_forward_workspace_bytes: usize,
    ) -> usize {
        expert_reclamation_bytes_to_fit_fixed_forward(
            stable_active_memory_bytes,
            retained_expert_payload_bytes,
            active_memory_ceiling_bytes,
            fixed_forward_workspace_bytes,
        )
    }

    #[must_use]
    pub fn retry_is_authorized(
        has_already_retried_after_reclamation: bool,
        retained_expert_payload_bytes_before_reclamation: u64,
        retained_expert_payload_bytes_after_reclamation: u64,
        required_reclamation_bytes: usize,
        sparse_experts_are_paged: bool,
    ) -> bool {
        // A paged retry re-reads the layer just freed to satisfy the ceiling,
        // promotes it again, and hits the same boundary after a long SSD stall.
        // Shrink the chunk instead of repeating that loop.
        !sparse_experts_are_paged
            && should_retry_fixed_forward_after_expert_reclamation(
                has_already_retried_after_reclamation,
                retained_expert_payload_bytes_before_reclamation,
                retained_expert_payload_bytes_after_reclamation,
                required_reclamation_bytes,
            )
    }
}

/// Evidence available after checkpoint restoration and expert reclamation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForwardRecoveryRequirements {
    /// Stable active bytes after restoring the request checkpoint.
    pub stable_active_memory_bytes: usize,
    /// Active bytes observed at the failed lazy-allocation boundary.
    pub active_memory_bytes_at_failure: usize,
    /// Exact allocation size rejected by MLX, or a conservative substitute.
    pub attempted_allocation_bytes: usize,
    /// Reusable transient lower bound learned from completed forwards.
    pub observed_transient_high_water_bytes: usize,
    /// Elastic expert payload before the executor performs reclamation.
    pub retained_expert_payload_bytes_before_reclamation: usize,
    /// Elastic expert payload after the executor performs reclamation.
    pub retained_expert_payload_bytes_after_reclamation: usize,
    /// Ceiling against which the unchanged retry must fit.
    pub active_memory_ceiling_bytes: usize,
    /// Prevents a capacity failure from entering an unbounded retry loop.
    pub has_already_retried_after_reclamation: bool,
    /// Paged experts must not retry the same chunk after reclaiming a layer.
    pub sparse_experts_are_paged: bool,
}

impl ForwardRecoveryRequirements {
    #[must_use]
    pub fn decide(self) -> ForwardRecoveryDecision {
        let fixed_forward_workspace_bytes = ForwardRecoveryPolicy::fixed_workspace_bytes(
            self.stable_active_memory_bytes,
            self.active_memory_bytes_at_failure,
            self.attempted_allocation_bytes,
            self.observed_transient_high_water_bytes,
        );
        let required_reclamation_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
            self.stable_active_memory_bytes,
            self.retained_expert_payload_bytes_before_reclamation,
            self.active_memory_ceiling_bytes,
            fixed_forward_workspace_bytes,
        );
        let should_retry = ForwardRecoveryPolicy::retry_is_authorized(
            self.has_already_retried_after_reclamation,
            u64::try_from(self.retained_expert_payload_bytes_before_reclamation)
                .unwrap_or(u64::MAX),
            u64::try_from(self.retained_expert_payload_bytes_after_reclamation).unwrap_or(u64::MAX),
            required_reclamation_bytes,
            self.sparse_experts_are_paged,
        );
        if should_retry {
            ForwardRecoveryDecision::Retry {
                fixed_forward_workspace_bytes,
                required_reclamation_bytes,
            }
        } else {
            ForwardRecoveryDecision::Reject {
                boundary: MemoryBoundary::AllocationProjection,
                fixed_forward_workspace_bytes,
                required_reclamation_bytes,
            }
        }
    }
}

/// Retry authorization and its complete calculation evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForwardRecoveryDecision {
    Retry {
        fixed_forward_workspace_bytes: usize,
        required_reclamation_bytes: usize,
    },
    Reject {
        boundary: MemoryBoundary,
        fixed_forward_workspace_bytes: usize,
        required_reclamation_bytes: usize,
    },
}
