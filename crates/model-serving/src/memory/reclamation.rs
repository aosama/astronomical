//! Pure checked arithmetic for expert-memory ownership transitions.
//!
//! This module deliberately knows nothing about MLX, model identities, tensor
//! layouts, or hardware. Runtime code supplies measured byte counts; these
//! formulas decide only whether those counts are internally consistent and how
//! much elastic expert retention must yield.

use thiserror::Error;

/// Invalid byte accounting that must fail closed before changing ownership.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ExpertMemoryAdmissionError {
    #[error("retained expert payload exceeds current active memory")]
    RetainedExpertPayloadExceedsActiveMemory,
    #[error("complete expert residency projection overflowed")]
    CompleteResidencyProjectionOverflow,
}

/// Projects active memory after replacing paged experts with complete experts.
///
/// `current_active_memory_bytes` already includes
/// `retained_paged_expert_payload_bytes`. Complete residency replaces that owner;
/// it does not coexist with it. Therefore:
///
/// `projected = current_active - retained_paged + complete`
pub fn projected_active_memory_after_complete_expert_replacement(
    current_active_memory_bytes: u64,
    retained_paged_expert_payload_bytes: u64,
    complete_expert_payload_bytes: u64,
) -> Result<u64, ExpertMemoryAdmissionError> {
    current_active_memory_bytes
        .checked_sub(retained_paged_expert_payload_bytes)
        .ok_or(ExpertMemoryAdmissionError::RetainedExpertPayloadExceedsActiveMemory)?
        .checked_add(complete_expert_payload_bytes)
        .ok_or(ExpertMemoryAdmissionError::CompleteResidencyProjectionOverflow)
}

/// Activation headroom required before complete expert residency may promote.
///
/// Complete residency only budgets static expert and non-expert payload. Serving
/// still needs temporary activations, key-value growth, and workspace memory.
/// Prefer the observed transient high-water from completed forwards; otherwise
/// reserve one complete layer. A tenth of the whole expert payload withheld
/// several gigabytes and forced SSD streaming when the user had already given
/// enough RAM to seat the model.
#[must_use]
pub const fn required_complete_residency_activation_headroom_bytes(
    startup_activation_floor_bytes: u64,
    observed_transient_high_water_bytes: u64,
) -> u64 {
    if observed_transient_high_water_bytes > startup_activation_floor_bytes {
        observed_transient_high_water_bytes
    } else {
        startup_activation_floor_bytes
    }
}

/// Returns true when static residency plus activation headroom exceeds the ceiling.
#[must_use]
pub const fn complete_residency_exceeds_ceiling_with_activation_headroom(
    projected_resident_active_memory_bytes: u64,
    stable_memory_ceiling_bytes: u64,
    required_activation_headroom_bytes: u64,
) -> bool {
    match projected_resident_active_memory_bytes.checked_add(required_activation_headroom_bytes) {
        Some(projected_with_headroom_bytes) => {
            projected_with_headroom_bytes > stable_memory_ceiling_bytes
        }
        // Overflow means the projection already cannot fit any finite ceiling.
        None => true,
    }
}

/// Expert bytes that must yield so a **fixed** forward size can fit the ceiling.
///
/// Chunk size is an input, not a free variable. Non-expert memory (model core,
/// restored context, and other owners) is treated as fixed for this decision.
/// Retained experts are the elastic category:
///
/// `non_elastic = current_active - retained_experts`
/// `max_experts_keep = ceiling - non_elastic - fixed_forward_workspace`
/// `reclaim = retained_experts - max(0, max_experts_keep)`
///
/// If even zero experts cannot leave room for the fixed workspace, the full
/// retained payload is required for reclamation and the caller must reject when
/// that still cannot satisfy the forward.
#[must_use]
pub const fn expert_reclamation_bytes_to_fit_fixed_forward(
    current_active_memory_bytes: usize,
    retained_expert_payload_bytes: usize,
    memory_ceiling_bytes: usize,
    fixed_forward_workspace_bytes: usize,
) -> usize {
    let non_elastic_active_memory_bytes =
        current_active_memory_bytes.saturating_sub(retained_expert_payload_bytes);
    let maximum_expert_payload_bytes_after_forward = memory_ceiling_bytes
        .saturating_sub(non_elastic_active_memory_bytes)
        .saturating_sub(fixed_forward_workspace_bytes);
    retained_expert_payload_bytes.saturating_sub(maximum_expert_payload_bytes_after_forward)
}

/// Reconstructs the workspace required when one forward allocation fails.
///
/// The failed allocation is additional to the transient arrays that were already
/// active at the failure boundary. Taking only the larger value underestimates
/// the retry. The observed high-water remains a reusable lower bound.
#[must_use]
pub const fn fixed_forward_workspace_after_allocation_failure(
    stable_active_memory_bytes: usize,
    active_memory_bytes_at_failure: usize,
    attempted_allocation_bytes: usize,
    observed_transient_high_water_bytes: usize,
) -> usize {
    let active_transient_memory_bytes =
        active_memory_bytes_at_failure.saturating_sub(stable_active_memory_bytes);
    let failed_forward_workspace_bytes =
        active_transient_memory_bytes.saturating_add(attempted_allocation_bytes);
    if failed_forward_workspace_bytes > observed_transient_high_water_bytes {
        failed_forward_workspace_bytes
    } else {
        observed_transient_high_water_bytes
    }
}

/// Returns whether one unchanged fixed forward should retry after expert eviction.
///
/// Native cache residency is authoritative at this ownership boundary. An MLX
/// active-memory sample can remain unchanged until an immutable execution
/// snapshot releases the evicted page array, even though cache policy has made
/// enough capacity available for the restored request to retry.
#[must_use]
pub fn should_retry_fixed_forward_after_expert_reclamation(
    has_already_retried_after_reclamation: bool,
    retained_expert_payload_bytes_before_reclamation: u64,
    retained_expert_payload_bytes_after_reclamation: u64,
    expert_reclamation_target_bytes: usize,
) -> bool {
    let released_expert_payload_bytes = retained_expert_payload_bytes_before_reclamation
        .saturating_sub(retained_expert_payload_bytes_after_reclamation);
    !has_already_retried_after_reclamation
        && expert_reclamation_target_bytes > 0
        && released_expert_payload_bytes
            >= u64::try_from(expert_reclamation_target_bytes).unwrap_or(u64::MAX)
}

/// Exact expert-retention reclamation required by one request operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertReclamationPlan {
    required_reclamation_bytes: usize,
    reclamation_target_bytes: usize,
    unresolved_shortfall_bytes: usize,
}

impl ExpertReclamationPlan {
    /// Computes the largest deficit across every mandatory memory boundary.
    ///
    /// Reclaiming one retained expert byte lowers stable, peak, and recovery
    /// projections by one byte. The smallest reclamation that satisfies all
    /// boundaries is therefore the maximum of their individual deficits.
    #[must_use]
    pub const fn for_projected_memory(
        stable_projected_bytes: usize,
        peak_projected_bytes: usize,
        recovery_projected_bytes: usize,
        stable_memory_ceiling_bytes: usize,
        transient_memory_ceiling_bytes: usize,
        retained_expert_payload_bytes: usize,
    ) -> Self {
        let stable_deficit_bytes =
            stable_projected_bytes.saturating_sub(stable_memory_ceiling_bytes);
        let peak_deficit_bytes =
            peak_projected_bytes.saturating_sub(transient_memory_ceiling_bytes);
        let recovery_deficit_bytes =
            recovery_projected_bytes.saturating_sub(transient_memory_ceiling_bytes);
        let required_reclamation_bytes = maximum_three(
            stable_deficit_bytes,
            peak_deficit_bytes,
            recovery_deficit_bytes,
        );
        let reclamation_target_bytes =
            minimum(required_reclamation_bytes, retained_expert_payload_bytes);
        let unresolved_shortfall_bytes =
            required_reclamation_bytes.saturating_sub(reclamation_target_bytes);
        Self {
            required_reclamation_bytes,
            reclamation_target_bytes,
            unresolved_shortfall_bytes,
        }
    }

    #[must_use]
    pub const fn required_reclamation_bytes(self) -> usize {
        self.required_reclamation_bytes
    }

    /// Never exceeds the currently retained expert payload.
    #[must_use]
    pub const fn reclamation_target_bytes(self) -> usize {
        self.reclamation_target_bytes
    }

    #[must_use]
    pub const fn unresolved_shortfall_bytes(self) -> usize {
        self.unresolved_shortfall_bytes
    }

    #[must_use]
    pub const fn can_satisfy_every_memory_boundary(self) -> bool {
        self.unresolved_shortfall_bytes == 0
    }
}

const fn minimum(left: usize, right: usize) -> usize {
    if left < right { left } else { right }
}

const fn maximum_three(first: usize, second: usize, third: usize) -> usize {
    let first_two_maximum = if first > second { first } else { second };
    if first_two_maximum > third {
        first_two_maximum
    } else {
        third
    }
}
