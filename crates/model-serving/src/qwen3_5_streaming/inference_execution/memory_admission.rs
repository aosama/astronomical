//! Request and forward memory admission for Qwen3.5 execution.
//!
//! This module joins three ownership views without letting any one of them guess:
//!
//! - decoder state projects exact persistent growth for the requested token count;
//! - adaptive RAM growth supplies learned transient and peak evidence;
//! - expert residency exposes the elastic bytes that may be reclaimed.
//!
//! Initial request admission reclaims retained pages only; complete-residency
//! admission belongs to the resident engine. Per-forward admission preserves the
//! chosen chunk/token count and reclaims only enough retained pages to satisfy
//! stable and expected-peak limits. Recovery-only headroom is diagnostic: an
//! actual typed MLX allocation failure owns checkpoint rollback, exact
//! reclamation, and one unchanged retry.
//!
//! On successful admission this module constrains retained expert ownership to
//! the strict-ceiling capacity left by the concrete forward reserve. That handoff
//! prevents mandatory reads from consuming bytes already proven necessary for
//! decoder growth, one operation-local expert page, and transient work.

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_streaming;
use crate::qwen3_5_streaming::model::adaptive_ram_growth_logging::{
    log_adaptive_ram_growth_admission_decision, log_adaptive_ram_growth_pressure,
};
use crate::qwen3_5_streaming::model::memory_admission;
use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, MemoryPhase, PagedForwardAdmissionDecision,
    PerformanceAttribution, PerformanceCounter, PerformanceOperation,
    combined_persistent_growth_bytes,
};

use super::{Qwen3_5EngineState, qwen3_5_runtime_error};

/// Bound on reclaim passes for one forward. Each pass snapshots MLX after a
/// synchronize, so a handful is enough to drain leftover experts; spinning
/// further would only hide a projection that cannot recede.
const MAXIMUM_PAGED_EXPERT_RECLAMATION_PASSES: u32 = 8;

pub(in crate::qwen3_5_streaming) enum AdaptiveRamGrowthMemoryAdmissionError {
    /// A typed paged-forward decision that the streaming caller could not enact.
    ForwardAdmission {
        decision: PagedForwardAdmissionDecision,
        reason: String,
    },
    /// A runtime or internal engine failure that must retain its original typed cause.
    Engine(InferenceEngineError),
}

impl From<InferenceEngineError> for AdaptiveRamGrowthMemoryAdmissionError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for InferenceEngineError {
    fn from(admission_error: AdaptiveRamGrowthMemoryAdmissionError) -> Self {
        match admission_error {
            AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission { reason, .. } => {
                memory_admission::invalid_request_error(reason)
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                inference_engine_error
            }
        }
    }
}

/// The exact pre-forward baselines post-forward learning compares against.
///
/// `usize::MAX` / `u64::MAX` sentinels mark a disabled adaptive guard; callers
/// may still sample telemetry but perform no learning from them.
pub(in crate::qwen3_5_streaming) struct AdaptiveRamGrowthAdmissionBaseline {
    pub(in crate::qwen3_5_streaming) active_memory_bytes: usize,
    pub(in crate::qwen3_5_streaming) retained_expert_payload_bytes: u64,
    /// Cumulative logical expert-streaming payload bytes at admission commit.
    /// The delta to this request's value after the forward is that forward's
    /// mandatory expert-page stream (issue #691).
    pub(in crate::qwen3_5_streaming) streamed_expert_page_bytes: u64,
    /// Evidence level that composed the admitted projection, exported so
    /// performance attribution can attribute reserve inflation to its source.
    pub(in crate::qwen3_5_streaming) transient_reserve_source:
        Option<crate::TransientWorkspaceReserveSource>,
    /// Allowed active memory the admitted forward's projected peak leaves
    /// unclaimed. Per-chunk staging spends this before borrowing warm expert
    /// tables, so free ceiling is used instead of evicting seated layers.
    pub(in crate::qwen3_5_streaming) unreserved_headroom_bytes: u64,
}

impl AdaptiveRamGrowthAdmissionBaseline {
    pub(in crate::qwen3_5_streaming) const fn disabled_guard_sentinel() -> Self {
        Self {
            active_memory_bytes: usize::MAX,
            retained_expert_payload_bytes: u64::MAX,
            streamed_expert_page_bytes: 0,
            transient_reserve_source: None,
            // A disabled guard projects nothing, so no headroom is proven.
            unreserved_headroom_bytes: 0,
        }
    }
}

impl Qwen3_5EngineState {
    /// Attributes adaptive admission, including any retained-expert reclamation.
    pub(in crate::qwen3_5_streaming) fn measure_adaptive_ram_growth_memory_admission(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        performance_attribution: &mut PerformanceAttribution,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<AdaptiveRamGrowthAdmissionBaseline, AdaptiveRamGrowthMemoryAdmissionError> {
        if !self.adaptive_ram_growth_guard_enabled {
            return Ok(AdaptiveRamGrowthAdmissionBaseline::disabled_guard_sentinel());
        }
        let performance_attribution_measures_admission = performance_attribution.is_enabled();
        let admitted_baseline = performance_attribution.measure_operation(
            PerformanceOperation::AdaptiveRamGrowthMemoryAdmission,
            |performance_attribution| {
                self.begin_adaptive_ram_growth(
                    adaptive_ram_growth_context,
                    request_decoder_state,
                    additional_persistent_state_growth_bytes,
                    exact_temporary_workspace_bytes,
                    performance_attribution_measures_admission,
                    performance_attribution,
                )
            },
        )?;
        // Performance attribution records which reserve source produced this
        // admission decision, so reserve inflation is attributable from
        // performance-attribution.jsonl alone (issue #691).
        if let Some(transient_reserve_source) = admitted_baseline.transient_reserve_source {
            // NoLearnedReserve marks request-scoped equations that carry no
            // learned transient evidence; a forward admission cannot produce it,
            // and no source-attribution counter exists for it.
            let admission_reserve_source_counter = match transient_reserve_source {
                crate::TransientWorkspaceReserveSource::ExactContext => {
                    Some(PerformanceCounter::AdmissionReserveExactContextSourceCount)
                }
                crate::TransientWorkspaceReserveSource::PhaseScaled => {
                    Some(PerformanceCounter::AdmissionReservePhaseScaledSourceCount)
                }
                crate::TransientWorkspaceReserveSource::BeyondObservedTokenSpan => {
                    Some(PerformanceCounter::AdmissionReserveBeyondObservedTokenSpanSourceCount)
                }
                crate::TransientWorkspaceReserveSource::GlobalMaximum => {
                    Some(PerformanceCounter::AdmissionReserveGlobalMaximumSourceCount)
                }
                crate::TransientWorkspaceReserveSource::NoLearnedReserve => None,
            };
            if let Some(admission_reserve_source_counter) = admission_reserve_source_counter {
                performance_attribution.record_counter(admission_reserve_source_counter, 1);
            }
        }
        Ok(admitted_baseline)
    }

    /// Admits one forward pass and starts an operation-local MLX peak sample.
    #[expect(clippy::too_many_lines)]
    fn begin_adaptive_ram_growth(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
        should_log_memory_decision: bool,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<AdaptiveRamGrowthAdmissionBaseline, AdaptiveRamGrowthMemoryAdmissionError> {
        // Capture one internally consistent ownership snapshot. The model borrow
        // ends with this block so later demotion/reclamation may borrow it mutably.
        let (
            target_persistent_state_growth_bytes,
            routed_expert_page_reservation_bytes,
            unobserved_scope_transient_floor_bytes,
            mut memory_snapshot_before_growth,
            mut retained_expert_payload_bytes_before_growth,
        ) = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let target_persistent_state_growth_bytes = request_decoder_state
                .projected_persistent_state_growth_bytes(
                    model.decoder_cache_layout(),
                    adaptive_ram_growth_context.forward_token_count(),
                )
                .map_err(qwen3_5_runtime_error)?;
            let routed_expert_page_reservation_bytes = if model.expert_pager.is_some() {
                // Reserve the largest model-derived routed page, not merely the
                // route expected from this token. Router output is lazy and is not
                // known on the host at initial admission.
                model
                    .expert_page_reservation_bytes_for_forward(
                        adaptive_ram_growth_context.forward_token_count(),
                    )
                    .map_err(InferenceEngineError::from)?
                    .try_into()
                    .map_err(|_| {
                        memory_admission::invalid_request_error(
                            "routed expert page reservation exceeds the platform range",
                        )
                    })?
            } else {
                0
            };
            let memory_snapshot_before_growth = model
                .runtime()
                .memory_snapshot()
                .map_err(qwen3_5_runtime_error)?;
            let retained_expert_payload_bytes_before_growth = model
                .expert_weight_memory_cache_statistics()
                .resident_payload_byte_count;
            let unobserved_scope_transient_floor_bytes = model
                .mlx_ram_budget()
                .adaptive_growth_unobserved_scope_transient_floor_bytes(
                    adaptive_ram_growth_context,
                );
            (
                target_persistent_state_growth_bytes,
                routed_expert_page_reservation_bytes,
                unobserved_scope_transient_floor_bytes,
                memory_snapshot_before_growth,
                retained_expert_payload_bytes_before_growth,
            )
        };
        let exact_context_growth_bytes = combined_persistent_growth_bytes(
            target_persistent_state_growth_bytes,
            additional_persistent_state_growth_bytes,
        )
        .ok_or_else(|| {
            memory_admission::invalid_request_error(
                "target and additional persistent growth overflowed",
            )
        })?;
        // The first projection describes ownership exactly as sampled. It may be
        // replaced below after retained-page eviction.
        let mut first_forward_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_context_growth_bytes,
                routed_expert_page_reservation_bytes,
                exact_temporary_workspace_bytes,
                unobserved_scope_transient_floor_bytes,
            )
            .map_err(|adaptive_ram_growth_projection_error| {
                tracing::warn!(
                    action = "reject",
                    current_active_memory_bytes =
                        memory_snapshot_before_growth.active_memory_bytes(),
                    exact_context_growth_bytes,
                    error = %adaptive_ram_growth_projection_error,
                    "stopped Qwen3.5 forward after adaptive RAM growth projection failed"
                );
                memory_admission::invalid_request_error(format!(
                    "adaptive RAM growth rejected: {adaptive_ram_growth_projection_error}"
                ))
            })?;

        let initial_forward_admission_decision = PagedForwardAdmissionDecision::for_paged_forward(
            &first_forward_projection,
            usize::try_from(retained_expert_payload_bytes_before_growth).unwrap_or(usize::MAX),
            None,
        );
        if should_log_memory_decision
            && matches!(
                adaptive_ram_growth_context.memory_phase(),
                MemoryPhase::Prefill
            )
            && initial_forward_admission_decision.is_admitted()
            && !first_forward_projection.has_full_recovery_reserve()
        {
            // This is an intentional admission, not a warning hidden by policy:
            // expected work fits, while only the optional second recovery window
            // does not. Emit enough evidence to distinguish those conditions.
            log_adaptive_ram_growth_admission_decision(
                adaptive_ram_growth_context,
                &first_forward_projection,
                "admit_with_recovery_constraint",
            );
        }
        if !initial_forward_admission_decision.is_admitted() {
            if should_log_memory_decision {
                log_adaptive_ram_growth_admission_decision(
                    adaptive_ram_growth_context,
                    &first_forward_projection,
                    initial_forward_admission_decision.paged_log_label(),
                );
            }

            // Every retained page is an elastic byte, so each typed target
            // releases exactly the contribution the equation proves necessary.
            let expert_weight_memory_cache_statistics_before_reclamation = {
                let model = self.model.as_ref().ok_or_else(|| {
                    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                })?;
                model.expert_weight_memory_cache_statistics()
            };
            let mut previous_pass_released_pages = None;
            let mut paged_expert_reclamation_admitted = false;
            for _ in 0..MAXIMUM_PAGED_EXPERT_RECLAMATION_PASSES {
                let model = self.model.as_ref().ok_or_else(|| {
                    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                })?;
                let expert_weight_memory_cache_statistics =
                    model.expert_weight_memory_cache_statistics();
                let retained_expert_payload_bytes = usize::try_from(
                    expert_weight_memory_cache_statistics.resident_payload_byte_count,
                )
                .unwrap_or(usize::MAX);
                let expert_reclamation_plan = first_forward_projection
                    .expert_retention_reclamation_plan(retained_expert_payload_bytes);
                let forward_admission_decision = PagedForwardAdmissionDecision::for_paged_forward(
                    &first_forward_projection,
                    retained_expert_payload_bytes,
                    previous_pass_released_pages,
                );
                match forward_admission_decision {
                    PagedForwardAdmissionDecision::Admit => {
                        if previous_pass_released_pages.is_some() {
                            log_adaptive_ram_growth_pressure(
                                &first_forward_projection,
                                expert_weight_memory_cache_statistics_before_reclamation,
                                expert_weight_memory_cache_statistics,
                                memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                                expert_reclamation_plan.reclamation_target_bytes(),
                                "admit",
                            );
                        }
                        paged_expert_reclamation_admitted = true;
                        break;
                    }
                    PagedForwardAdmissionDecision::Reject { .. } => {
                        log_adaptive_ram_growth_pressure(
                            &first_forward_projection,
                            expert_weight_memory_cache_statistics_before_reclamation,
                            expert_weight_memory_cache_statistics,
                            memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                            expert_reclamation_plan.reclamation_target_bytes(),
                            "reject",
                        );
                        let rejection_reason = if !expert_reclamation_plan
                            .can_satisfy_every_memory_boundary()
                        {
                            format!(
                                "adaptive RAM growth exceeds reclaimable expert capacity by {} bytes",
                                expert_reclamation_plan.unresolved_shortfall_bytes(),
                            )
                        } else if matches!(previous_pass_released_pages, Some(false)) {
                            format!(
                                "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, and recovery projection of {} bytes do not fit C={} bytes and P={} bytes while retained expert paging is unavailable",
                                first_forward_projection.stable_projected_bytes(),
                                first_forward_projection.peak_projected_bytes(),
                                first_forward_projection.recovery_projected_bytes(),
                                first_forward_projection.active_memory_ceiling_bytes(),
                                first_forward_projection.allowed_active_memory_bytes(),
                            )
                        } else {
                            format!(
                                "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, or recovery projection of {} bytes remains above C={} bytes or P={} bytes after retained-expert reclamation",
                                first_forward_projection.stable_projected_bytes(),
                                first_forward_projection.peak_projected_bytes(),
                                first_forward_projection.recovery_projected_bytes(),
                                first_forward_projection.active_memory_ceiling_bytes(),
                                first_forward_projection.allowed_active_memory_bytes(),
                            )
                        };
                        return Err(AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission {
                            decision: forward_admission_decision,
                            reason: rejection_reason,
                        });
                    }
                    PagedForwardAdmissionDecision::ReclaimRetainedExperts { target_bytes } => {
                        let Some(memory_snapshot_after_reclamation) =
                            qwen3_5_streaming::reclaim_retained_experts_for_request_memory_pressure(
                                model,
                                target_bytes,
                            )?
                        else {
                            previous_pass_released_pages = Some(false);
                            continue;
                        };
                        let expert_weight_memory_cache_statistics_after_reclamation =
                            model.expert_weight_memory_cache_statistics();
                        log_adaptive_ram_growth_pressure(
                            &first_forward_projection,
                            expert_weight_memory_cache_statistics_before_reclamation,
                            expert_weight_memory_cache_statistics_after_reclamation,
                            memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                            target_bytes,
                            "reclaim_experts",
                        );
                        first_forward_projection = self
                            .adaptive_ram_growth_guard
                            .project_growth_for_context(
                                adaptive_ram_growth_context,
                                memory_snapshot_after_reclamation.active_memory_bytes(),
                                exact_context_growth_bytes,
                                routed_expert_page_reservation_bytes,
                                exact_temporary_workspace_bytes,
                                unobserved_scope_transient_floor_bytes,
                            )
                            .map_err(|adaptive_ram_growth_projection_error| {
                                tracing::warn!(
                                    action = "reject",
                                    error = %adaptive_ram_growth_projection_error,
                                    "stopped Qwen3.5 forward after post-reclamation adaptive RAM growth projection failed"
                                );
                                memory_admission::invalid_request_error(format!(
                                    "adaptive RAM growth rejected: {adaptive_ram_growth_projection_error}"
                                ))
                            })?;
                        previous_pass_released_pages = Some(true);
                        memory_snapshot_before_growth = memory_snapshot_after_reclamation;
                        retained_expert_payload_bytes_before_growth =
                            expert_weight_memory_cache_statistics_after_reclamation
                                .resident_payload_byte_count;
                    }
                }
            }
            if !paged_expert_reclamation_admitted {
                let retained_expert_payload_bytes =
                    usize::try_from(retained_expert_payload_bytes_before_growth)
                        .unwrap_or(usize::MAX);
                let exhausted_decision = PagedForwardAdmissionDecision::for_paged_forward(
                    &first_forward_projection,
                    retained_expert_payload_bytes,
                    Some(false),
                );
                let rejection_decision = match exhausted_decision {
                    PagedForwardAdmissionDecision::Reject { .. } => exhausted_decision,
                    PagedForwardAdmissionDecision::Admit
                    | PagedForwardAdmissionDecision::ReclaimRetainedExperts { .. } => {
                        PagedForwardAdmissionDecision::Reject {
                            unresolved_shortfall_bytes: first_forward_projection
                                .operation_reclamation_required_bytes(),
                        }
                    }
                };
                return Err(AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission {
                    reason: format!(
                        "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, or recovery projection of {} bytes remains above C={} bytes or P={} bytes after retained-expert reclamation",
                        first_forward_projection.stable_projected_bytes(),
                        first_forward_projection.peak_projected_bytes(),
                        first_forward_projection.recovery_projected_bytes(),
                        first_forward_projection.active_memory_ceiling_bytes(),
                        first_forward_projection.allowed_active_memory_bytes(),
                    ),
                    decision: rejection_decision,
                });
            }
        }
        // MLX's peak counter is process-global. Reset it only after admission so
        // the next sample measures this one forward pass rather than model loading
        // or an earlier prefill chunk.
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let admitted_forward_reserve_bytes =
            u64::try_from(first_forward_projection.forward_reserve_bytes()).unwrap_or(u64::MAX);
        let current_active_memory_bytes =
            u64::try_from(memory_snapshot_before_growth.active_memory_bytes()).unwrap_or(u64::MAX);
        let current_retained_expert_payload_bytes = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        if model.limit_expert_retention_for_admitted_forward(
            current_active_memory_bytes,
            current_retained_expert_payload_bytes,
            admitted_forward_reserve_bytes,
        ) {
            model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .map_err(qwen3_5_runtime_error)?;
            memory_snapshot_before_growth = model
                .runtime()
                .memory_snapshot()
                .map_err(qwen3_5_runtime_error)?;
            retained_expert_payload_bytes_before_growth = model
                .expert_weight_memory_cache_statistics()
                .resident_payload_byte_count;
        }
        model
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        // Return the exact pre-forward baselines used by post-forward learning:
        // active memory, retained payload, and the cumulative expert-streaming
        // counter whose later delta is this forward's promoted page bytes
        // (issue #691). `usize::MAX` is reserved by the disabled guard.
        Ok(AdaptiveRamGrowthAdmissionBaseline {
            active_memory_bytes: memory_snapshot_before_growth.active_memory_bytes(),
            retained_expert_payload_bytes: retained_expert_payload_bytes_before_growth,
            streamed_expert_page_bytes: performance_attribution
                .expert_streaming_payload_byte_count(),
            transient_reserve_source: Some(first_forward_projection.transient_reserve_source()),
            unreserved_headroom_bytes: u64::try_from(
                first_forward_projection
                    .allowed_active_memory_bytes()
                    .saturating_sub(first_forward_projection.peak_projected_bytes()),
            )
            .unwrap_or(u64::MAX),
        })
    }
}
