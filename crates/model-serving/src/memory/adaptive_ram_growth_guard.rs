//! Forward-specific admission and transient-memory learning.
//!
//! The guard answers a narrower question than `MlxRamBudget`: can one concrete
//! forward grow from the *current* MLX state without crossing stable or expected
//! peak limits? It uses exact persistent growth supplied by decoder-state owners,
//! one routed expert-page reservation, explicit temporary workspace, and the
//! largest transient high-water observed from completed forwards.
//!
//! Three projections are retained for diagnostics:
//!
//! - `stable`: current active + persistent growth + routed page;
//! - `peak`: stable + explicit workspace + learned transient high-water;
//! - `recovery`: peak + one equal transient window.
//!
//! Stable and peak are admission boundaries. Recovery is deliberately diagnostic.
//! Preemptively evicting experts for a recovery-only shortfall caused avoidable
//! SSD paging on requests whose actual expected peak fitted. If a real allocation
//! still fails, the caller restores the request checkpoint, reclaims the exact
//! expert deficit, and retries the unchanged forward once.

use std::collections::BTreeMap;

use thiserror::Error;

use super::ExpertRetentionReclamationPlan;

/// Forward-pass class whose temporary allocation history informs admission.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdaptiveRamGrowthPhase {
    Prefill,
    Decode,
}

/// Exact recurrent execution shape whose temporary allocation evidence may recur.
///
/// Observations are deliberately not transferable between chunk sizes, prompt
/// positions, visual requests, MTP histories, or sparse-expert residency modes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdaptiveRamGrowthContext {
    adaptive_ram_growth_phase: AdaptiveRamGrowthPhase,
    forward_token_count: usize,
    prompt_position_context_bucket: u64,
    has_visual_embeddings: bool,
    has_mtp_prompt_history: bool,
    sparse_experts_are_paged: bool,
}

impl AdaptiveRamGrowthContext {
    /// Builds a prefill context using the prompt sizer's position bucket.
    #[must_use]
    pub const fn prefill(
        forward_token_count: usize,
        prompt_position_context_bucket: u64,
        has_visual_embeddings: bool,
        has_mtp_prompt_history: bool,
        sparse_experts_are_paged: bool,
    ) -> Self {
        Self {
            adaptive_ram_growth_phase: AdaptiveRamGrowthPhase::Prefill,
            forward_token_count,
            prompt_position_context_bucket,
            has_visual_embeddings,
            has_mtp_prompt_history,
            sparse_experts_are_paged,
        }
    }

    /// Builds a decode context. Decode has no prompt-chunk position bucket.
    #[must_use]
    pub const fn decode(
        forward_token_count: usize,
        has_mtp_prompt_history: bool,
        sparse_experts_are_paged: bool,
    ) -> Self {
        Self {
            adaptive_ram_growth_phase: AdaptiveRamGrowthPhase::Decode,
            forward_token_count,
            prompt_position_context_bucket: 0,
            has_visual_embeddings: false,
            has_mtp_prompt_history,
            sparse_experts_are_paged,
        }
    }

    #[must_use]
    pub const fn adaptive_ram_growth_phase(self) -> AdaptiveRamGrowthPhase {
        self.adaptive_ram_growth_phase
    }

    /// Returns the exact number of tokens forwarded by this operation.
    #[must_use]
    pub const fn forward_token_count(self) -> usize {
        self.forward_token_count
    }

    /// Replaces only the observed sparse-expert residency dimension after a forward.
    #[must_use]
    pub const fn with_sparse_experts_are_paged(mut self, sparse_experts_are_paged: bool) -> Self {
        self.sparse_experts_are_paged = sparse_experts_are_paged;
        self
    }

    #[must_use]
    pub const fn sparse_experts_are_paged(self) -> bool {
        self.sparse_experts_are_paged
    }
}

/// Protects a machine-derived MLX active-memory limit with exact workload evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRamGrowthGuard {
    /// Stable ceiling C. A one-percent allowance P is derived for transient peak.
    active_memory_limit_bytes: usize,
    /// High-water transient bytes keyed by execution shape and ownership mode.
    observed_transient_high_water_bytes_by_context: BTreeMap<AdaptiveRamGrowthContext, usize>,
}

/// Checked projection evidence for one adaptive growth operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRamGrowthProjection {
    /// MLX active bytes sampled immediately before this admission decision.
    current_active_memory_bytes: usize,
    /// Exact key/value, recurrent, and caller-declared persistent growth.
    exact_persistent_growth_bytes: usize,
    /// Maximum bounded expert page that may coexist with this forward.
    routed_expert_page_reservation_bytes: usize,
    /// Known one-operation workspace not represented by learned history.
    exact_temporary_workspace_bytes: usize,
    /// Conservative reusable transient evidence from prior completed forwards.
    observed_transient_high_water_bytes: usize,
    stable_projected_bytes: usize,
    peak_projected_bytes: usize,
    recovery_projected_bytes: usize,
    active_memory_limit_bytes: usize,
    allowed_active_memory_bytes: usize,
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
    pub const fn active_memory_limit_bytes(&self) -> usize {
        self.active_memory_limit_bytes
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
            .saturating_sub(self.active_memory_limit_bytes);
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
    ) -> ExpertRetentionReclamationPlan {
        // Pass peak twice on purpose. `ExpertRetentionReclamationPlan` is a pure
        // three-boundary formula also used by stricter callers; replacing recovery
        // with peak here excludes recovery-only deficits from preemptive eviction
        // while preserving one shared checked-arithmetic implementation.
        ExpertRetentionReclamationPlan::for_projected_memory(
            self.stable_projected_bytes,
            self.peak_projected_bytes,
            self.peak_projected_bytes,
            self.active_memory_limit_bytes,
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

impl AdaptiveRamGrowthGuard {
    /// Creates a guard for one machine-derived MLX active-memory limit.
    pub fn new(active_memory_limit_bytes: usize) -> Result<Self, AdaptiveRamGrowthGuardError> {
        if active_memory_limit_bytes == 0 {
            return Err(AdaptiveRamGrowthGuardError::InvalidActiveMemoryLimit);
        }
        Ok(Self {
            active_memory_limit_bytes,
            observed_transient_high_water_bytes_by_context: BTreeMap::new(),
        })
    }

    /// Replaces the active limit while retaining all exact-context measurements.
    pub fn update_active_memory_limit_bytes(
        &mut self,
        active_memory_limit_bytes: usize,
    ) -> Result<(), AdaptiveRamGrowthGuardError> {
        if active_memory_limit_bytes == 0 {
            return Err(AdaptiveRamGrowthGuardError::InvalidActiveMemoryLimit);
        }
        self.active_memory_limit_bytes = active_memory_limit_bytes;
        Ok(())
    }

    /// Returns whether this phase has at least one retained exact-context observation.
    #[must_use]
    pub fn has_completed_growth_observation(
        &self,
        adaptive_ram_growth_phase: AdaptiveRamGrowthPhase,
    ) -> bool {
        self.observed_transient_high_water_bytes_by_context
            .keys()
            .any(|adaptive_ram_growth_context| {
                adaptive_ram_growth_context.adaptive_ram_growth_phase() == adaptive_ram_growth_phase
            })
    }

    /// Returns the phase maximum for phase-specific telemetry.
    #[must_use]
    pub fn observed_transient_high_water_bytes(
        &self,
        adaptive_ram_growth_phase: AdaptiveRamGrowthPhase,
    ) -> usize {
        self.observed_transient_high_water_bytes_by_context
            .iter()
            .filter_map(
                |(adaptive_ram_growth_context, observed_transient_high_water_bytes)| {
                    (adaptive_ram_growth_context.adaptive_ram_growth_phase()
                        == adaptive_ram_growth_phase)
                        .then_some(*observed_transient_high_water_bytes)
                },
            )
            .max()
            .unwrap_or(0)
    }

    /// Returns the largest transient window observed across all completed
    /// forward phases. Admission uses this conservative value because a future
    /// request may begin in a different phase than the request that supplied
    /// the only reusable observation. For example, a short prompt may not be
    /// retained as prefill evidence, while its decode forwards still establish
    /// the transient workspace required by the same model and memory owner.
    #[must_use]
    pub fn admission_transient_high_water_bytes(&self) -> usize {
        self.observed_transient_high_water_bytes_by_context
            .values()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Builds a checked C-stable and P-peak projection from exact-context evidence.
    pub fn project_growth_for_context(
        &self,
        _adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        current_active_memory_bytes: usize,
        exact_persistent_growth_bytes: usize,
        routed_expert_page_reservation_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<AdaptiveRamGrowthProjection, AdaptiveRamGrowthGuardError> {
        // Use the global maximum transient observation rather than an
        // exact-context or phase-only lookup. Exact contexts include token count
        // and position bucket, while a short prompt may not retain prefill
        // evidence at all. A completed decode can still establish the transient
        // workspace needed by the next request's prefill, so admission must
        // reserve the largest reusable window across every phase.
        let observed_transient_high_water_bytes = self.admission_transient_high_water_bytes();
        let stable_projected_bytes = current_active_memory_bytes
            .checked_add(exact_persistent_growth_bytes)
            .and_then(|projected_bytes| {
                projected_bytes.checked_add(routed_expert_page_reservation_bytes)
            })
            .ok_or(AdaptiveRamGrowthGuardError::MemoryProjectionOverflow)?;
        // Explicit workspace and learned transient history are additive. Taking
        // only their maximum would under-reserve when a new operation introduces
        // known workspace on top of ordinary activation behavior.
        let predicted_transient_bytes = exact_temporary_workspace_bytes
            .checked_add(observed_transient_high_water_bytes)
            .ok_or(AdaptiveRamGrowthGuardError::MemoryProjectionOverflow)?;
        let peak_projected_bytes = stable_projected_bytes
            .checked_add(predicted_transient_bytes)
            .ok_or(AdaptiveRamGrowthGuardError::MemoryProjectionOverflow)?;
        let recovery_projected_bytes = peak_projected_bytes
            .checked_add(predicted_transient_bytes)
            .ok_or(AdaptiveRamGrowthGuardError::MemoryProjectionOverflow)?;
        // P is the repository's approved temporary allowance. Stable ownership
        // must fit C; a short-lived peak may use C + 1 percent.
        let transient_allowance_bytes = self.active_memory_limit_bytes / 100;
        let allowed_active_memory_bytes = self
            .active_memory_limit_bytes
            .checked_add(transient_allowance_bytes)
            .unwrap_or(usize::MAX);
        Ok(AdaptiveRamGrowthProjection {
            current_active_memory_bytes,
            exact_persistent_growth_bytes,
            routed_expert_page_reservation_bytes,
            exact_temporary_workspace_bytes,
            observed_transient_high_water_bytes,
            stable_projected_bytes,
            peak_projected_bytes,
            recovery_projected_bytes,
            active_memory_limit_bytes: self.active_memory_limit_bytes,
            allowed_active_memory_bytes,
        })
    }

    /// Retains only recurring-context transient evidence after a successful forward.
    pub fn record_completed_growth_for_context(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        should_retain_observation: bool,
        active_memory_bytes_before_growth: usize,
        active_memory_bytes_after_growth: usize,
        peak_memory_bytes_during_growth: usize,
        exact_temporary_workspace_bytes: usize,
    ) {
        if !should_retain_observation {
            return;
        }
        // The post-forward active sample already includes every newly retained
        // expert page. Using it as the stable baseline excludes expert growth
        // from the transient window. Subtracting expert growth again would erase
        // real activation headroom and let the next forward overfill retention.
        let stable_active_memory_bytes =
            active_memory_bytes_before_growth.max(active_memory_bytes_after_growth);
        let observed_transient_growth_bytes = peak_memory_bytes_during_growth
            .saturating_sub(stable_active_memory_bytes)
            .saturating_sub(exact_temporary_workspace_bytes);
        // Store the residual only. Explicit workspace is supplied again by the
        // next operation; retaining it in learned history would double-count it.
        self.observed_transient_high_water_bytes_by_context
            .entry(adaptive_ram_growth_context)
            .and_modify(|existing_observed_transient_high_water_bytes| {
                *existing_observed_transient_high_water_bytes =
                    (*existing_observed_transient_high_water_bytes)
                        .max(observed_transient_growth_bytes);
            })
            .or_insert(observed_transient_growth_bytes);
    }
}

/// Typed rejection from adaptive RAM growth admission.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AdaptiveRamGrowthGuardError {
    #[error("adaptive RAM growth requires a positive active-memory limit")]
    InvalidActiveMemoryLimit,
    #[error("adaptive RAM growth memory projection overflowed")]
    MemoryProjectionOverflow,
}
