//! Forward-specific admission and transient-memory learning.
//!
//! The guard answers a narrower question than `MlxRamBudget`: can one concrete
//! forward grow from the *current* MLX state without crossing stable or expected
//! peak limits? It uses exact persistent growth supplied by decoder-state owners,
//! one routed expert-page reservation, explicit temporary workspace, and a
//! transient reserve resolved from completed forwards at the highest specificity
//! available: the exact execution context first, then larger same-phase
//! observations scaled down to the forward's token count, then the caller's
//! analytic floor when every same-phase observation was smaller, then the
//! global maximum for a phase with no evidence at all (issue #623). Evidence is
//! never scaled UP by a token ratio (measured 2026-10-24).
//!
//! Three projections are retained for diagnostics:
//!
//! - `stable`: current active + persistent growth + routed page;
//! - `peak`: stable + explicit workspace + the resolved transient reserve;
//! - `recovery`: peak + one equal transient window.
//!
//! Stable and peak are admission boundaries. Recovery is deliberately diagnostic.
//! Preemptively evicting experts for a recovery-only shortfall caused avoidable
//! SSD paging on requests whose actual expected peak fitted. If a real allocation
//! still fails, the caller restores the request checkpoint, reclaims the exact
//! expert deficit, and retries the unchanged forward once.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::budget::adaptive_growth_projection::{
    AdaptiveRamGrowthProjection, AdaptiveRamGrowthTransientReserveSource,
    scale_transient_to_token_count,
};
use crate::phase::MemoryPhase;

/// Separates complete-residency forwards from forwards that use expert paging.
///
/// Hybrid and Paged ownership share the paged profile because both execute
/// pager-backed operations whose transient memory evidence is not interchangeable
/// with complete-residency forwards.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdaptiveRamGrowthExecutionProfile {
    Resident,
    Paged,
}

/// Exact recurrent execution shape whose temporary allocation evidence may recur.
///
/// Each context keys its own exact-shape evidence. Admission reuses evidence
/// from other same-phase contexts only conservatively: a larger forward's
/// observation is scaled down, a smaller forward's observation is used
/// unscaled, and no observation is ever scaled up to a larger forward.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdaptiveRamGrowthContext {
    memory_phase: MemoryPhase,
    forward_token_count: usize,
    prompt_position_context_bucket: u64,
    has_visual_embeddings: bool,
    execution_profile: AdaptiveRamGrowthExecutionProfile,
}

impl AdaptiveRamGrowthContext {
    /// Builds a prefill context using the prompt sizer's position bucket.
    #[must_use]
    pub const fn prefill(
        forward_token_count: usize,
        prompt_position_context_bucket: u64,
        has_visual_embeddings: bool,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> Self {
        Self {
            memory_phase: MemoryPhase::Prefill,
            forward_token_count,
            prompt_position_context_bucket,
            has_visual_embeddings,
            execution_profile,
        }
    }

    /// Builds a decode context. Decode has no prompt-chunk position bucket.
    #[must_use]
    pub const fn decode(
        forward_token_count: usize,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> Self {
        Self {
            memory_phase: MemoryPhase::Decode,
            forward_token_count,
            prompt_position_context_bucket: 0,
            has_visual_embeddings: false,
            execution_profile,
        }
    }

    #[must_use]
    pub const fn memory_phase(self) -> MemoryPhase {
        self.memory_phase
    }

    /// Returns the exact number of tokens forwarded by this operation.
    #[must_use]
    pub const fn forward_token_count(self) -> usize {
        self.forward_token_count
    }

    /// Replaces only the execution profile after a forward.
    #[must_use]
    pub const fn with_execution_profile(
        mut self,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> Self {
        self.execution_profile = execution_profile;
        self
    }

    #[must_use]
    pub const fn execution_profile(self) -> AdaptiveRamGrowthExecutionProfile {
        self.execution_profile
    }
}

/// Protects a machine-derived MLX active-memory limit with exact workload evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRamGrowthGuard {
    /// Stable ceiling C. A one-percent allowance P is derived for transient peak.
    active_memory_ceiling_bytes: usize,
    /// High-water transient bytes keyed by execution shape and ownership mode.
    observed_transient_high_water_bytes_by_context: BTreeMap<AdaptiveRamGrowthContext, usize>,
}

impl AdaptiveRamGrowthGuard {
    /// Creates a guard for one machine-derived MLX active-memory limit.
    pub fn new(active_memory_ceiling_bytes: usize) -> Result<Self, AdaptiveRamGrowthGuardError> {
        if active_memory_ceiling_bytes == 0 {
            return Err(AdaptiveRamGrowthGuardError::InvalidActiveMemoryCeiling);
        }
        Ok(Self {
            active_memory_ceiling_bytes,
            observed_transient_high_water_bytes_by_context: BTreeMap::new(),
        })
    }

    /// Replaces the active limit while retaining all exact-context measurements.
    pub fn update_active_memory_ceiling_bytes(
        &mut self,
        active_memory_ceiling_bytes: usize,
    ) -> Result<(), AdaptiveRamGrowthGuardError> {
        if active_memory_ceiling_bytes == 0 {
            return Err(AdaptiveRamGrowthGuardError::InvalidActiveMemoryCeiling);
        }
        self.active_memory_ceiling_bytes = active_memory_ceiling_bytes;
        Ok(())
    }

    /// Returns whether this phase has at least one retained exact-context observation.
    #[must_use]
    pub fn has_completed_growth_observation(&self, memory_phase: MemoryPhase) -> bool {
        self.observed_transient_high_water_bytes_by_context
            .keys()
            .any(|adaptive_ram_growth_context| {
                adaptive_ram_growth_context.memory_phase() == memory_phase
            })
    }

    /// Returns the phase maximum for phase-specific telemetry.
    #[must_use]
    pub fn observed_transient_high_water_bytes(&self, memory_phase: MemoryPhase) -> usize {
        self.observed_transient_high_water_bytes_by_context
            .iter()
            .filter_map(
                |(adaptive_ram_growth_context, observed_transient_high_water_bytes)| {
                    (adaptive_ram_growth_context.memory_phase() == memory_phase)
                        .then_some(*observed_transient_high_water_bytes)
                },
            )
            .max()
            .unwrap_or(0)
    }

    /// Returns transient evidence for this exact execution context.
    #[must_use]
    pub fn observed_transient_high_water_bytes_for_context(
        &self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    ) -> usize {
        self.observed_transient_high_water_bytes_by_context
            .get(&adaptive_ram_growth_context)
            .copied()
            .unwrap_or(0)
    }

    /// Returns the largest transient window observed across all completed
    /// forward phases. Callers without a concrete forward context use this
    /// conservative value; forward admission resolves a shape-fitted reserve
    /// through [`Self::admission_transient_reserve_for_context`] instead.
    #[must_use]
    pub fn admission_transient_high_water_bytes(&self) -> usize {
        self.observed_transient_high_water_bytes_by_context
            .values()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Resolves the transient reserve for one concrete forward context.
    ///
    /// The ladder widens one dimension class at a time, and each level has a
    /// distinct evidence basis:
    ///
    /// 1. **Exact context** — the same phase, token count, position bucket,
    ///    and ownership mode completed a forward before; its high-water is the
    ///    tightest known bound for this operation.
    /// 2. **Phase-scaled** — no exact observation, but the phase completed a
    ///    forward at least as large as this one. Activation workspace scales
    ///    with the forward's token count, so each larger observation is scaled
    ///    DOWN to this forward's token count (ceiling division) and the
    ///    largest scaled value wins. An 8,192-token chunk's 8 GB window
    ///    therefore reserves about 2 GB for a 2,048-token chunk instead of
    ///    borrowing the absolute 8 GB. Smaller observations join unscaled as a
    ///    lower bound, because a forward never needs less workspace than a
    ///    smaller forward of the same phase.
    /// 3. **Beyond the observed token span** — every same-phase observation
    ///    forwarded fewer tokens than this one. Scaling a small forward up by
    ///    the token ratio multiplies its fixed per-forward overhead: a
    ///    one-token prefill's 11 MB window became a 23 GB reserve for a
    ///    2,048-token chunk and demoted a fully resident model to SSD
    ///    streaming with more than 5 GB of the ceiling unused (measured
    ///    2026-10-24). The caller's analytic floor for an unobserved scope
    ///    governs instead — the same activation workspace request admission
    ///    already promised for this operation — unless a smaller observation
    ///    is larger still. This mirrors the rule `MlxRamBudget` follows for
    ///    its own activation promise (issue #690).
    /// 4. **Global maximum** — the phase has no evidence at all. The first
    ///    forward of a new phase reserves the largest window ever observed, or
    ///    the analytic floor when larger; its completion records that phase's
    ///    own evidence. Observations that carry a zero token count only
    ///    participate here, because they cannot bound a per-token estimate.
    ///
    /// Under-projection remains recoverable by design: a typed allocation
    /// failure restores the request checkpoint, reclaims the exact deficit,
    /// and retries the unchanged forward, which also records the missing
    /// shape's evidence. Over-reservation has no such self-correction — it
    /// demotes expert ownership outright — so specificity wins over blanket
    /// conservatism here.
    #[must_use]
    pub fn admission_transient_reserve_for_context(
        &self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        unobserved_scope_transient_floor_bytes: usize,
    ) -> (usize, AdaptiveRamGrowthTransientReserveSource) {
        if let Some(&exact_context_high_water_bytes) = self
            .observed_transient_high_water_bytes_by_context
            .get(&adaptive_ram_growth_context)
        {
            return (
                exact_context_high_water_bytes,
                AdaptiveRamGrowthTransientReserveSource::ExactContext,
            );
        }
        let forward_token_count = adaptive_ram_growth_context.forward_token_count();
        let mut largest_scaled_down_covering_reserve_bytes: Option<usize> = None;
        let mut largest_unscaled_smaller_observation_bytes: Option<usize> = None;
        for (observed_context, &observed_high_water_bytes) in self
            .observed_transient_high_water_bytes_by_context
            .iter()
            .filter(|(observed_context, _)| {
                observed_context.memory_phase() == adaptive_ram_growth_context.memory_phase()
                    && observed_context.forward_token_count() > 0
            })
        {
            if observed_context.forward_token_count() < forward_token_count {
                largest_unscaled_smaller_observation_bytes = Some(
                    largest_unscaled_smaller_observation_bytes
                        .map_or(observed_high_water_bytes, |largest_bytes| {
                            largest_bytes.max(observed_high_water_bytes)
                        }),
                );
            } else if let Some(scaled_down_reserve_bytes) = scale_transient_to_token_count(
                observed_high_water_bytes,
                observed_context.forward_token_count(),
                forward_token_count,
            ) {
                largest_scaled_down_covering_reserve_bytes = Some(
                    largest_scaled_down_covering_reserve_bytes
                        .map_or(scaled_down_reserve_bytes, |largest_bytes| {
                            largest_bytes.max(scaled_down_reserve_bytes)
                        }),
                );
            }
        }
        match (
            largest_scaled_down_covering_reserve_bytes,
            largest_unscaled_smaller_observation_bytes,
        ) {
            (
                Some(scaled_down_covering_reserve_bytes),
                optional_unscaled_smaller_observation_bytes,
            ) => (
                scaled_down_covering_reserve_bytes
                    .max(optional_unscaled_smaller_observation_bytes.unwrap_or(0)),
                AdaptiveRamGrowthTransientReserveSource::PhaseScaled,
            ),
            (None, Some(unscaled_smaller_observation_bytes)) => (
                unscaled_smaller_observation_bytes.max(unobserved_scope_transient_floor_bytes),
                AdaptiveRamGrowthTransientReserveSource::BeyondObservedTokenSpan,
            ),
            (None, None) => (
                self.admission_transient_high_water_bytes()
                    .max(unobserved_scope_transient_floor_bytes),
                AdaptiveRamGrowthTransientReserveSource::GlobalMaximum,
            ),
        }
    }

    /// Returns the retained-payload ceiling that keeps the adaptive growth
    /// guard's peak projection inside its limits: the active ceiling plus its
    /// transient allowance, minus current active memory, the learned transient
    /// reserve, and one routed-page reservation for the next forward, expressed
    /// relative to current retained ownership.
    ///
    /// Decode warming allocates persistent tables that the next forward's
    /// admission would otherwise count against the peak limit. Capping warming
    /// at this headroom — and reclaiming tables when the headroom is negative —
    /// keeps the hot-expert cache from taking memory the adaptive growth guard
    /// must hold for transients and KV growth.
    #[must_use]
    pub fn hot_expert_retention_ceiling_bytes(
        &self,
        memory_phase: MemoryPhase,
        current_active_memory_bytes: usize,
        current_retained_payload_bytes: u64,
        routed_expert_page_reservation_bytes: usize,
    ) -> u64 {
        let transient_allowance_bytes = self.active_memory_ceiling_bytes / 100;
        let allowed_active_memory_bytes =
            self.active_memory_ceiling_bytes + transient_allowance_bytes;
        // Warming runs inside one phase, so the reserve it must respect is that
        // phase's own learned workspace. Capping decode warming at the largest
        // workspace ever seen in any phase let one huge prefill suppress decode
        // warming forever (issue #512): the prefill transient is spent by the
        // time decode runs, and prefill admission still reserves its own
        // shape-specific transient through `project_growth_for_context`, so a later
        // large prefill reclaims warm tables through the existing pressure path
        // instead of needing warming to have predicted it. Before this phase has
        // any observation the all-phase maximum keeps the first warming steps
        // conservative.
        let phase_transient_reserve_bytes = if self.has_completed_growth_observation(memory_phase) {
            self.observed_transient_high_water_bytes(memory_phase)
        } else {
            self.admission_transient_high_water_bytes()
        };
        let signed_headroom_bytes = i128::try_from(allowed_active_memory_bytes)
            .unwrap_or(i128::MAX)
            .saturating_sub(i128::try_from(current_active_memory_bytes).unwrap_or(i128::MAX))
            .saturating_sub(i128::try_from(phase_transient_reserve_bytes).unwrap_or(i128::MAX))
            .saturating_sub(
                i128::try_from(routed_expert_page_reservation_bytes).unwrap_or(i128::MAX),
            );
        let signed_ceiling_bytes =
            i128::from(current_retained_payload_bytes) + signed_headroom_bytes;
        signed_ceiling_bytes.clamp(0, i128::from(u64::MAX)) as u64
    }

    /// Builds a checked C-stable and P-peak projection from exact-context evidence.
    ///
    /// `unobserved_scope_transient_floor_bytes` is the caller's analytic
    /// activation reserve for this forward, used only when no same-phase
    /// observation covers the forward's token count (see
    /// [`Self::admission_transient_reserve_for_context`]). Callers pass the
    /// same per-operation workspace their request admission promised, so this
    /// guard can never demote a forward that request admission already sized.
    pub fn project_growth_for_context(
        &self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        current_active_memory_bytes: usize,
        exact_persistent_growth_bytes: usize,
        routed_expert_page_reservation_bytes: usize,
        exact_temporary_workspace_bytes: usize,
        unobserved_scope_transient_floor_bytes: usize,
    ) -> Result<AdaptiveRamGrowthProjection, AdaptiveRamGrowthGuardError> {
        // Reserve from the most specific evidence available and widen only when
        // it is missing (issue #623): the exact context first, then larger
        // same-phase observations scaled down to this forward, then the
        // analytic floor beyond the observed token span, then the global
        // maximum for a phase with no evidence at all. Charging one shape's
        // absolute transient window to every other shape demoted fully
        // resident expert owners whose own chunk needed a fraction of it.
        let (observed_transient_high_water_bytes, transient_reserve_source) = self
            .admission_transient_reserve_for_context(
                adaptive_ram_growth_context,
                unobserved_scope_transient_floor_bytes,
            );
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
        let transient_allowance_bytes = self.active_memory_ceiling_bytes / 100;
        let allowed_active_memory_bytes = self
            .active_memory_ceiling_bytes
            .checked_add(transient_allowance_bytes)
            .unwrap_or(usize::MAX);
        Ok(AdaptiveRamGrowthProjection {
            current_active_memory_bytes,
            exact_persistent_growth_bytes,
            routed_expert_page_reservation_bytes,
            exact_temporary_workspace_bytes,
            observed_transient_high_water_bytes,
            transient_reserve_source,
            stable_projected_bytes,
            peak_projected_bytes,
            recovery_projected_bytes,
            active_memory_ceiling_bytes: self.active_memory_ceiling_bytes,
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
        self.record_completed_growth_excluding_expert_page_stream(
            adaptive_ram_growth_context,
            should_retain_observation,
            active_memory_bytes_before_growth,
            active_memory_bytes_after_growth,
            peak_memory_bytes_during_growth,
            exact_temporary_workspace_bytes,
            0,
            0,
        );
    }

    /// Retains recurring-context transient evidence after a forward that
    /// promoted mandatory expert pages.
    ///
    /// `retained_expert_payload_growth_bytes` is the resident-payload delta the
    /// forward created and `promoted_expert_page_stream_bytes` is everything
    /// that forward streamed from storage. Pages that stayed resident are
    /// already excluded because the post-forward active sample (`max` below)
    /// includes the retention delta; subtracting them again would erase real
    /// activation headroom. Pages that were streamed and evicted appear only
    /// in the peak, so they must leave the residual — otherwise every future
    /// activation reserve derived from this evidence inflates by the streamed
    /// byte size and shrinks the expert-retention budget (issue #691).
    #[allow(clippy::too_many_arguments)]
    pub fn record_completed_growth_excluding_expert_page_stream(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        should_retain_observation: bool,
        active_memory_bytes_before_growth: usize,
        active_memory_bytes_after_growth: usize,
        peak_memory_bytes_during_growth: usize,
        exact_temporary_workspace_bytes: usize,
        retained_expert_payload_growth_bytes: u64,
        promoted_expert_page_stream_bytes: u64,
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
        // Streamed page bytes inside the peak but outside that stable baseline:
        // promotions beyond the resident delta were read in and evicted, so they
        // are expert-page spike ownership, never activation (issue #691).
        let expert_page_bytes_outside_stable_baseline = usize::try_from(
            promoted_expert_page_stream_bytes.saturating_sub(retained_expert_payload_growth_bytes),
        )
        .unwrap_or(usize::MAX);
        let observed_transient_growth_bytes = peak_memory_bytes_during_growth
            .saturating_sub(stable_active_memory_bytes)
            .saturating_sub(exact_temporary_workspace_bytes)
            .saturating_sub(expert_page_bytes_outside_stable_baseline);
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
    InvalidActiveMemoryCeiling,
    #[error("adaptive RAM growth memory projection overflowed")]
    MemoryProjectionOverflow,
}
