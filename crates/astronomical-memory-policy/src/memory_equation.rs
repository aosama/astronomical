//! One checked owner for stable, transient-peak, and recovery memory projections.

use crate::budget::TransientWorkspaceReserve;
use crate::reclamation::ExpertReclamationPlan;

/// Composed memory ownership for one request admission or forward.
///
/// The peak allowance is one percent of the active-memory ceiling, matching
/// the per-forward policy. Request admission can still ask for a strict-ceiling
/// verdict because its temporary workspace is promised before forward execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryEquation {
    current_active_memory_bytes: usize,
    persistent_growth_bytes: usize,
    routed_expert_page_reservation_bytes: usize,
    explicit_temporary_workspace_bytes: usize,
    transient_workspace_reserve: TransientWorkspaceReserve,
    stable_projected_bytes: usize,
    peak_projected_bytes: usize,
    recovery_projected_bytes: usize,
    active_memory_ceiling_bytes: usize,
    allowed_active_memory_bytes: usize,
}

impl MemoryEquation {
    /// Builds the checked projection, returning `None` if any projected sum overflows.
    #[must_use]
    pub fn new(
        current_active_memory_bytes: usize,
        persistent_growth_bytes: usize,
        routed_expert_page_reservation_bytes: usize,
        explicit_temporary_workspace_bytes: usize,
        transient_workspace_reserve: TransientWorkspaceReserve,
        active_memory_ceiling_bytes: usize,
    ) -> Option<Self> {
        let stable_projected_bytes = current_active_memory_bytes
            .checked_add(persistent_growth_bytes)?
            .checked_add(routed_expert_page_reservation_bytes)?;
        let forward_transient_workspace_bytes = explicit_temporary_workspace_bytes
            .checked_add(transient_workspace_reserve.reserve_bytes())?;
        let peak_projected_bytes =
            stable_projected_bytes.checked_add(forward_transient_workspace_bytes)?;
        let recovery_projected_bytes =
            peak_projected_bytes.checked_add(forward_transient_workspace_bytes)?;
        let transient_allowance_bytes = active_memory_ceiling_bytes / 100;
        let allowed_active_memory_bytes =
            active_memory_ceiling_bytes.saturating_add(transient_allowance_bytes);
        Some(Self {
            current_active_memory_bytes,
            persistent_growth_bytes,
            routed_expert_page_reservation_bytes,
            explicit_temporary_workspace_bytes,
            transient_workspace_reserve,
            stable_projected_bytes,
            peak_projected_bytes,
            recovery_projected_bytes,
            active_memory_ceiling_bytes,
            allowed_active_memory_bytes,
        })
    }

    /// Composes request admission's explicit workspace under its existing strict ceiling.
    ///
    /// Request admission reserves workspace before a particular forward exists,
    /// so it has no learned reserve: the explicit temporary workspace is the
    /// entire transient claim, and `without_learned_reserve` states that
    /// provenance instead of mislabeling a zero-byte global maximum. The
    /// composed numbers are identical to the zero-byte reserve this path has
    /// always used, so the strict-ceiling verdict is unchanged.
    #[must_use]
    pub fn for_request_admission(
        current_active_memory_bytes: usize,
        persistent_growth_bytes: usize,
        routed_expert_page_reservation_bytes: usize,
        temporary_workspace_bytes: usize,
        active_memory_ceiling_bytes: usize,
    ) -> Option<Self> {
        Self::new(
            current_active_memory_bytes,
            persistent_growth_bytes,
            routed_expert_page_reservation_bytes,
            temporary_workspace_bytes,
            TransientWorkspaceReserve::without_learned_reserve(),
            active_memory_ceiling_bytes,
        )
    }

    #[must_use]
    pub const fn current_active_memory_bytes(&self) -> usize {
        self.current_active_memory_bytes
    }

    #[must_use]
    pub const fn persistent_growth_bytes(&self) -> usize {
        self.persistent_growth_bytes
    }

    #[must_use]
    pub const fn routed_expert_page_reservation_bytes(&self) -> usize {
        self.routed_expert_page_reservation_bytes
    }

    #[must_use]
    pub const fn explicit_temporary_workspace_bytes(&self) -> usize {
        self.explicit_temporary_workspace_bytes
    }

    #[must_use]
    pub const fn transient_workspace_reserve(&self) -> TransientWorkspaceReserve {
        self.transient_workspace_reserve
    }

    #[must_use]
    pub const fn stable_projected_bytes(&self) -> usize {
        self.stable_projected_bytes
    }

    #[must_use]
    pub const fn peak_projected_bytes(&self) -> usize {
        self.peak_projected_bytes
    }

    #[must_use]
    pub const fn recovery_projected_bytes(&self) -> usize {
        self.recovery_projected_bytes
    }

    #[must_use]
    pub const fn active_memory_ceiling_bytes(&self) -> usize {
        self.active_memory_ceiling_bytes
    }

    /// Returns the ceiling plus its approved one-percent forward-peak allowance.
    #[must_use]
    pub const fn allowed_active_memory_bytes(&self) -> usize {
        self.allowed_active_memory_bytes
    }

    #[must_use]
    pub const fn stable_shortfall_bytes(&self) -> usize {
        self.stable_projected_bytes
            .saturating_sub(self.active_memory_ceiling_bytes)
    }

    #[must_use]
    pub const fn peak_shortfall_bytes(&self) -> usize {
        self.peak_projected_bytes
            .saturating_sub(self.allowed_active_memory_bytes)
    }

    #[must_use]
    pub const fn request_memory_shortfall_bytes(&self) -> usize {
        self.peak_projected_bytes
            .saturating_sub(self.active_memory_ceiling_bytes)
    }

    #[must_use]
    pub const fn recovery_reserve_shortfall_bytes(&self) -> usize {
        self.recovery_projected_bytes
            .saturating_sub(self.allowed_active_memory_bytes)
    }

    #[must_use]
    pub const fn fits_stable_limit(&self) -> bool {
        self.stable_shortfall_bytes() == 0
    }

    #[must_use]
    pub const fn fits_stable_and_peak_limits(&self) -> bool {
        self.fits_stable_limit() && self.peak_shortfall_bytes() == 0
    }

    /// Checks request admission using its pre-existing strict active-memory ceiling.
    #[must_use]
    pub const fn fits_request_memory_limit(&self) -> bool {
        self.request_memory_shortfall_bytes() == 0
    }

    #[must_use]
    pub const fn has_full_recovery_reserve(&self) -> bool {
        self.recovery_reserve_shortfall_bytes() == 0
    }

    #[must_use]
    pub const fn forward_reserve_bytes(&self) -> usize {
        self.peak_projected_bytes
            .saturating_sub(self.current_active_memory_bytes)
    }

    #[must_use]
    pub const fn operation_reclamation_required_bytes(&self) -> usize {
        let stable_deficit_bytes = self.stable_shortfall_bytes();
        let peak_deficit_bytes = self.peak_shortfall_bytes();
        if stable_deficit_bytes > peak_deficit_bytes {
            stable_deficit_bytes
        } else {
            peak_deficit_bytes
        }
    }

    /// Plans reclamation against stable and expected peak limits, excluding recovery-only reserve.
    #[must_use]
    pub const fn expert_retention_reclamation_plan(
        &self,
        retained_expert_payload_bytes: usize,
    ) -> ExpertReclamationPlan {
        ExpertReclamationPlan::for_projected_memory(
            self.stable_projected_bytes,
            self.peak_projected_bytes,
            self.peak_projected_bytes,
            self.active_memory_ceiling_bytes,
            self.allowed_active_memory_bytes,
            retained_expert_payload_bytes,
        )
    }
}

#[cfg(test)]
#[path = "memory_equation_tests.rs"]
mod memory_equation_tests;
