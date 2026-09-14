//! The checked projection evidence one adaptive growth admission produces.
//!
//! [`AdaptiveRamGrowthProjection`] is the immutable outcome of projecting one
//! concrete forward against the stable ceiling, the expected peak, and the
//! diagnostic recovery window. Keeping it beside the transient-reserve
//! vocabulary separates *what admission decided from* (the projection and the
//! source of its reserve) from *how evidence is learned* (the guard).

use crate::memory::ExpertReclamationPlan;

/// Which evidence level supplied a forward admission's transient reserve.
///
/// Recorded on the projection and in admission-decision logs so a demotion can
/// be attributed to the reserve that caused it (issue #623).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRamGrowthTransientReserveSource {
    /// The same phase, token count, position bucket, and ownership mode
    /// completed a forward before.
    ExactContext,
    /// Same-phase observations rescaled to this forward's token count.
    PhaseScaled,
    /// The phase had no evidence; the largest window ever observed in any
    /// phase was reserved.
    GlobalMaximum,
}

/// Rescales an observed transient window to a different forward token count.
///
/// Activation workspace grows with the number of tokens in the forward, so the
/// estimate is proportional with ceiling division (conservative). Returns
/// `None` when the observation carries a zero token count, which cannot bound
/// a proportional estimate; such observations still participate through the
/// global-maximum fallback.
#[must_use]
pub(crate) fn scale_transient_to_token_count(
    observed_high_water_bytes: usize,
    observed_forward_token_count: usize,
    target_forward_token_count: usize,
) -> Option<usize> {
    if observed_forward_token_count == 0 || target_forward_token_count == 0 {
        return None;
    }
    let observed_bytes_u128 =
        u128::from(u64::try_from(observed_high_water_bytes).unwrap_or(u64::MAX));
    let target_tokens_u128 =
        u128::from(u64::try_from(target_forward_token_count).unwrap_or(u64::MAX));
    let observed_tokens_u128 =
        u128::from(u64::try_from(observed_forward_token_count).unwrap_or(u64::MAX));
    let scaled_bytes = (observed_bytes_u128 * target_tokens_u128 + observed_tokens_u128 - 1)
        / observed_tokens_u128;
    usize::try_from(scaled_bytes).ok()
}

/// Checked projection evidence for one adaptive growth operation.
///
/// Fields are crate-visible because the guard is the single constructor; all
/// external readers go through the accessor methods below.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRamGrowthProjection {
    /// MLX active bytes sampled immediately before this admission decision.
    pub(crate) current_active_memory_bytes: usize,
    /// Exact key/value, recurrent, and caller-declared persistent growth.
    pub(crate) exact_persistent_growth_bytes: usize,
    /// Maximum bounded expert page that may coexist with this forward.
    pub(crate) routed_expert_page_reservation_bytes: usize,
    /// Known one-operation workspace not represented by learned history.
    pub(crate) exact_temporary_workspace_bytes: usize,
    /// Conservative reusable transient evidence resolved for this context.
    pub(crate) observed_transient_high_water_bytes: usize,
    /// Which evidence level supplied `observed_transient_high_water_bytes`.
    pub(crate) transient_reserve_source: AdaptiveRamGrowthTransientReserveSource,
    pub(crate) stable_projected_bytes: usize,
    pub(crate) peak_projected_bytes: usize,
    pub(crate) recovery_projected_bytes: usize,
    pub(crate) active_memory_ceiling_bytes: usize,
    pub(crate) allowed_active_memory_bytes: usize,
}

impl AdaptiveRamGrowthProjection {
    #[must_use]
    pub const fn current_active_memory_bytes(&self) -> usize {
        self.current_active_memory_bytes
    }

    #[must_use]
    pub const fn exact_persistent_growth_bytes(&self) -> usize {
        self.exact_persistent_growth_bytes
    }

    #[must_use]
    pub const fn routed_expert_page_reservation_bytes(&self) -> usize {
        self.routed_expert_page_reservation_bytes
    }

    #[must_use]
    pub const fn exact_temporary_workspace_bytes(&self) -> usize {
        self.exact_temporary_workspace_bytes
    }

    #[must_use]
    pub const fn observed_transient_high_water_bytes(&self) -> usize {
        self.observed_transient_high_water_bytes
    }

    #[must_use]
    pub const fn transient_reserve_source(&self) -> AdaptiveRamGrowthTransientReserveSource {
        self.transient_reserve_source
    }

    /// Returns stable bytes after persistent state growth and before temporary work.
    #[must_use]
    pub const fn stable_projected_bytes(&self) -> usize {
        self.stable_projected_bytes
    }

    /// Returns stable bytes plus the exact-context transient high-water window.
    #[must_use]
    pub const fn peak_projected_bytes(&self) -> usize {
        self.peak_projected_bytes
    }

    /// Returns peak bytes plus one equal diagnostic recovery window.
    #[must_use]
    pub const fn recovery_projected_bytes(&self) -> usize {
        self.recovery_projected_bytes
    }

    /// Returns the complete non-expert reserve that expert residency must leave
    /// available for this admitted forward through its expected peak.
    ///
    /// Recovery-only shortfall is diagnostic and handled by typed allocation-
    /// failure rollback, exact reclamation, and retry. Passing the expected-peak
    /// difference into the residency planner keeps both policy owners on the same
    /// initial-admission equation.
    #[must_use]
    pub const fn forward_reserve_bytes(&self) -> usize {
        self.peak_projected_bytes
            .saturating_sub(self.current_active_memory_bytes)
    }

    #[must_use]
    pub const fn active_memory_ceiling_bytes(&self) -> usize {
        self.active_memory_ceiling_bytes
    }

    /// Returns the configured ceiling plus its approved one-percent transient allowance.
    #[must_use]
    pub const fn allowed_active_memory_bytes(&self) -> usize {
        self.allowed_active_memory_bytes
    }

    /// Returns the exact retained-expert reclamation needed by stable and peak work.
    #[must_use]
    pub const fn operation_reclamation_required_bytes(&self) -> usize {
        let stable_deficit_bytes = self
            .stable_projected_bytes
            .saturating_sub(self.active_memory_ceiling_bytes);
        let peak_deficit_bytes = self
            .peak_projected_bytes
            .saturating_sub(self.allowed_active_memory_bytes);
        if stable_deficit_bytes > peak_deficit_bytes {
            stable_deficit_bytes
        } else {
            peak_deficit_bytes
        }
    }

    /// Returns the diagnostic recovery-reserve shortfall against the transient ceiling.
    #[must_use]
    pub const fn recovery_reserve_shortfall_bytes(&self) -> usize {
        self.recovery_projected_bytes
            .saturating_sub(self.allowed_active_memory_bytes)
    }

    /// Plans preemptive reclamation for stable and expected-peak deficits.
    ///
    /// Recovery remains diagnostic. A recovery-only shortfall is handled by the
    /// typed allocation-failure checkpoint, exact reclamation, and retry path.
    #[must_use]
    pub const fn expert_retention_reclamation_plan(
        &self,
        retained_expert_payload_bytes: usize,
    ) -> ExpertReclamationPlan {
        // Pass peak twice on purpose. `ExpertReclamationPlan` is a pure
        // three-boundary formula also used by stricter callers; replacing recovery
        // with peak here excludes recovery-only deficits from preemptive eviction
        // while preserving one shared checked-arithmetic implementation.
        ExpertReclamationPlan::for_projected_memory(
            self.stable_projected_bytes,
            self.peak_projected_bytes,
            self.peak_projected_bytes,
            self.active_memory_ceiling_bytes,
            self.allowed_active_memory_bytes,
            retained_expert_payload_bytes,
        )
    }

    #[must_use]
    pub const fn fits_stable_and_peak_limits(&self) -> bool {
        self.operation_reclamation_required_bytes() == 0
    }

    #[must_use]
    pub const fn has_full_recovery_reserve(&self) -> bool {
        self.recovery_reserve_shortfall_bytes() == 0
    }
}
