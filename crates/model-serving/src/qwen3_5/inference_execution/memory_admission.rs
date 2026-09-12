//! Request and forward memory admission for Qwen3.5 execution.
//!
//! This module joins three ownership views without letting any one of them guess:
//!
//! - decoder state projects exact persistent growth for the requested token count;
//! - adaptive RAM growth supplies learned transient and peak evidence;
//! - expert residency exposes the elastic bytes that may be reclaimed.
//!
//! Initial request admission may demote an indivisible complete-resident expert
//! model. Per-forward admission then preserves the chosen chunk/token count and
//! reclaims only enough retained paged experts to satisfy stable and expected-peak
//! limits. Recovery-only headroom is diagnostic: an actual typed MLX allocation
//! failure owns checkpoint rollback, exact reclamation, and one unchanged retry.
//!
//! On successful admission this module constrains retained expert ownership to
//! the strict-ceiling capacity left by the concrete forward reserve. That handoff
//! prevents mandatory reads from consuming bytes already proven necessary for
//! decoder growth, one operation-local expert page, and transient work.

use crate::qwen3_5::decoder::RequestDecoderStateStack;
use crate::qwen3_5::model::adaptive_ram_growth_logging::{
    log_adaptive_ram_growth_admission_decision, log_adaptive_ram_growth_pressure,
};
use crate::qwen3_5::model::memory_admission::invalid_request_error;
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;
use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, MemoryPhase, PagedExpertReclamationStep,
    PerformanceAttribution, PerformanceOperation, combined_persistent_growth_bytes,
    next_paged_expert_reclamation_step,
};

use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

/// Bound on reclaim passes for one forward. Each pass snapshots MLX after a
/// synchronize, so a handful is enough to drain leftover experts; spinning
/// further would only hide a projection that cannot recede.
const MAXIMUM_PAGED_EXPERT_RECLAMATION_PASSES: u32 = 8;

pub(in crate::qwen3_5) enum AdaptiveRamGrowthMemoryAdmissionError {
    /// The request is valid, but this fixed operation cannot fit after all legal reclamation.
    InsufficientCapacity { reason: String },
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
            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason } => {
                invalid_request_error(reason)
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                inference_engine_error
            }
        }
    }
}

impl Qwen3_5EngineState {
    /// Attributes adaptive admission, including any retained-expert reclamation.
    pub(in crate::qwen3_5) fn measure_adaptive_ram_growth_memory_admission(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        performance_attribution: &mut PerformanceAttribution,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<(usize, u64), AdaptiveRamGrowthMemoryAdmissionError> {
        if !self.adaptive_ram_growth_guard_enabled {
            return Ok((usize::MAX, u64::MAX));
        }
        let should_log_memory_decision = performance_attribution.is_enabled();
        performance_attribution.measure_operation(
            PerformanceOperation::AdaptiveRamGrowthMemoryAdmission,
            |_performance_attribution| {
                self.begin_adaptive_ram_growth(
                    adaptive_ram_growth_context,
                    request_decoder_state,
                    additional_persistent_state_growth_bytes,
                    exact_temporary_workspace_bytes,
                    should_log_memory_decision,
                )
            },
        )
    }

    /// Admits one forward pass and starts an operation-local MLX peak sample.
    fn begin_adaptive_ram_growth(
        &mut self,
        mut adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
        should_log_memory_decision: bool,
    ) -> Result<(usize, u64), AdaptiveRamGrowthMemoryAdmissionError> {
        // Capture one internally consistent ownership snapshot. The model borrow
        // ends with this block so later demotion/reclamation may borrow it mutably.
        let (
            target_persistent_state_growth_bytes,
            mut routed_expert_page_reservation_bytes,
            mut memory_snapshot_before_growth,
            mut retained_expert_payload_bytes_before_growth,
        ) = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let target_persistent_state_growth_bytes = request_decoder_state
                .projected_persistent_state_growth_bytes(
                    model.decoder_cache_layout(),
                    adaptive_ram_growth_context.forward_token_count(),
                )
                .map_err(qwen3_5_runtime_error)?;
            let routed_expert_page_reservation_bytes = if model.sparse_experts_are_paged() {
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
                        invalid_request_error(
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
            (
                target_persistent_state_growth_bytes,
                routed_expert_page_reservation_bytes,
                memory_snapshot_before_growth,
                retained_expert_payload_bytes_before_growth,
            )
        };
        let exact_context_growth_bytes = combined_persistent_growth_bytes(
            target_persistent_state_growth_bytes,
            additional_persistent_state_growth_bytes,
        )
        .ok_or_else(|| {
            invalid_request_error("target and additional persistent growth overflowed")
        })?;
        // The first projection describes ownership exactly as sampled. It may be
        // replaced below after complete-resident demotion or paged-byte eviction.
        let mut first_forward_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_context_growth_bytes,
                routed_expert_page_reservation_bytes,
                exact_temporary_workspace_bytes,
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
                invalid_request_error(format!(
                    "adaptive RAM growth rejected: {adaptive_ram_growth_projection_error}"
                ))
            })?;

        if should_log_memory_decision
            && matches!(
                adaptive_ram_growth_context.memory_phase(),
                MemoryPhase::Prefill
            )
            && first_forward_projection.fits_stable_and_peak_limits()
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
        if !first_forward_projection.fits_stable_and_peak_limits() {
            if should_log_memory_decision {
                log_adaptive_ram_growth_admission_decision(
                    adaptive_ram_growth_context,
                    &first_forward_projection,
                    "demote_resident_experts_or_reclaim_paged_experts",
                );
            }
            // Chunk size is fixed. Complete resident experts are the elastic
            // owner: demote them at the configured forward size so page-level
            // reclamation can free enough memory for that same chunk.
            if let Some(resident_demotion) = self.demote_resident_experts_for_adaptive_growth(
                adaptive_ram_growth_context,
                exact_context_growth_bytes,
                exact_temporary_workspace_bytes,
            )? {
                // Demotion changes active memory, routed-page need, and the
                // residency dimension of learned context. Replace every dependent
                // value together; mixing pre/post-demotion evidence is invalid.
                adaptive_ram_growth_context = resident_demotion.adaptive_ram_growth_context;
                memory_snapshot_before_growth = resident_demotion.memory_snapshot;
                routed_expert_page_reservation_bytes =
                    resident_demotion.routed_expert_page_reservation_bytes;
                first_forward_projection = resident_demotion.projection;
                retained_expert_payload_bytes_before_growth = self
                    .model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                    .expert_weight_memory_cache_statistics()
                    .resident_payload_byte_count;
            }

            if !first_forward_projection.fits_stable_and_peak_limits() {
                // From here onward experts must be paged. Every retained page is an
                // elastic byte, so the pure plan's one-byte-in/one-byte-out proof is
                // valid for stable and expected-peak deficits.
                let expert_weight_memory_cache_statistics_before_reclamation = {
                    let model = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    if model.resident_expert_weights.is_some() {
                        return Err(
                            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                                reason:
                                    "resident expert ownership remained indivisible after demotion"
                                        .to_owned(),
                            },
                        );
                    }
                    model.expert_weight_memory_cache_statistics()
                };
                let mut previous_pass_released_pages = None;
                let mut paged_expert_reclamation_admitted = false;
                for _ in 0..MAXIMUM_PAGED_EXPERT_RECLAMATION_PASSES {
                    let model = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    let expert_weight_memory_cache_statistics =
                        model.expert_weight_memory_cache_statistics();
                    let retained_expert_payload_bytes = usize::try_from(
                        expert_weight_memory_cache_statistics.resident_payload_byte_count,
                    )
                    .unwrap_or(usize::MAX);
                    let expert_reclamation_plan = first_forward_projection
                        .expert_retention_reclamation_plan(retained_expert_payload_bytes);
                    match next_paged_expert_reclamation_step(
                        first_forward_projection.fits_stable_and_peak_limits(),
                        expert_reclamation_plan,
                        previous_pass_released_pages,
                    ) {
                        PagedExpertReclamationStep::Admit => {
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
                        PagedExpertReclamationStep::Reject => {
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
                            return Err(
                                AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                                    reason: rejection_reason,
                                },
                            );
                        }
                        PagedExpertReclamationStep::Reclaim { target_bytes } => {
                            let Some(memory_snapshot_after_reclamation) =
                                reclaim_retained_experts_for_request_memory_pressure(
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
                                )
                                .map_err(|adaptive_ram_growth_projection_error| {
                                    tracing::warn!(
                                        action = "reject",
                                        error = %adaptive_ram_growth_projection_error,
                                        "stopped Qwen3.5 forward after post-reclamation adaptive RAM growth projection failed"
                                    );
                                    invalid_request_error(format!(
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
                    return Err(
                        AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                            reason: format!(
                                "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, or recovery projection of {} bytes remains above C={} bytes or P={} bytes after retained-expert reclamation",
                                first_forward_projection.stable_projected_bytes(),
                                first_forward_projection.peak_projected_bytes(),
                                first_forward_projection.recovery_projected_bytes(),
                                first_forward_projection.active_memory_ceiling_bytes(),
                                first_forward_projection.allowed_active_memory_bytes(),
                            ),
                        },
                    );
                }
            }
        }
        // MLX's peak counter is process-global. Reset it only after admission so
        // the next sample measures this one forward pass rather than model loading
        // or an earlier prefill chunk.
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
        // Return the exact pre-forward baseline and retained payload used by
        // post-forward learning. `usize::MAX` is reserved by the disabled guard.
        Ok((
            memory_snapshot_before_growth.active_memory_bytes(),
            retained_expert_payload_bytes_before_growth,
        ))
    }
}
