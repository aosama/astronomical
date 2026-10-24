//! Immutable evidence for one adaptive growth admission.

use crate::budget::TransientWorkspaceReserve;
use crate::memory_equation::MemoryEquation;
use crate::reclamation::ExpertReclamationPlan;

pub use super::transient_workspace_reserve::TransientWorkspaceReserveSource;

/// Checked projection evidence for one adaptive growth operation.
///
/// The equation owns every byte total and boundary; this wrapper retains the
/// established per-forward projection name used by serving diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRamGrowthProjection {
    memory_equation: MemoryEquation,
}

impl AdaptiveRamGrowthProjection {
    pub(crate) const fn from_memory_equation(memory_equation: MemoryEquation) -> Self {
        Self { memory_equation }
    }

    #[must_use]
    pub const fn memory_equation(&self) -> &MemoryEquation {
        &self.memory_equation
    }

    #[must_use]
    pub const fn transient_workspace_reserve(&self) -> TransientWorkspaceReserve {
        self.memory_equation.transient_workspace_reserve()
    }

    #[must_use]
    pub const fn current_active_memory_bytes(&self) -> usize {
        self.memory_equation.current_active_memory_bytes()
    }

    #[must_use]
    pub const fn exact_persistent_growth_bytes(&self) -> usize {
        self.memory_equation.persistent_growth_bytes()
    }

    #[must_use]
    pub const fn routed_expert_page_reservation_bytes(&self) -> usize {
        self.memory_equation.routed_expert_page_reservation_bytes()
    }

    #[must_use]
    pub const fn exact_temporary_workspace_bytes(&self) -> usize {
        self.memory_equation.explicit_temporary_workspace_bytes()
    }

    #[must_use]
    pub const fn observed_transient_high_water_bytes(&self) -> usize {
        self.transient_workspace_reserve().reserve_bytes()
    }

    #[must_use]
    pub const fn transient_reserve_source(&self) -> TransientWorkspaceReserveSource {
        self.transient_workspace_reserve().source()
    }

    #[must_use]
    pub const fn stable_projected_bytes(&self) -> usize {
        self.memory_equation.stable_projected_bytes()
    }

    #[must_use]
    pub const fn peak_projected_bytes(&self) -> usize {
        self.memory_equation.peak_projected_bytes()
    }

    #[must_use]
    pub const fn recovery_projected_bytes(&self) -> usize {
        self.memory_equation.recovery_projected_bytes()
    }

    #[must_use]
    pub const fn forward_reserve_bytes(&self) -> usize {
        self.memory_equation.forward_reserve_bytes()
    }

    #[must_use]
    pub const fn active_memory_ceiling_bytes(&self) -> usize {
        self.memory_equation.active_memory_ceiling_bytes()
    }

    #[must_use]
    pub const fn allowed_active_memory_bytes(&self) -> usize {
        self.memory_equation.allowed_active_memory_bytes()
    }

    #[must_use]
    pub const fn operation_reclamation_required_bytes(&self) -> usize {
        self.memory_equation.operation_reclamation_required_bytes()
    }

    #[must_use]
    pub const fn recovery_reserve_shortfall_bytes(&self) -> usize {
        self.memory_equation.recovery_reserve_shortfall_bytes()
    }

    #[must_use]
    pub const fn expert_retention_reclamation_plan(
        &self,
        retained_expert_payload_bytes: usize,
    ) -> ExpertReclamationPlan {
        self.memory_equation
            .expert_retention_reclamation_plan(retained_expert_payload_bytes)
    }

    #[must_use]
    pub const fn fits_stable_and_peak_limits(&self) -> bool {
        self.memory_equation.fits_stable_and_peak_limits()
    }

    #[must_use]
    pub const fn fits_stable_limit(&self) -> bool {
        self.memory_equation.fits_stable_limit()
    }

    #[must_use]
    pub const fn has_full_recovery_reserve(&self) -> bool {
        self.memory_equation.has_full_recovery_reserve()
    }
}
