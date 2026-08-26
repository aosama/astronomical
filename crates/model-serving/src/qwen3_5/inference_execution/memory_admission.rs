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
    AdaptiveRamGrowthContext, AdaptiveRamGrowthPhase, InferenceEngineError, MlxRamBudgetPhase,
    PerformanceAttribution, PerformanceAttributionOutcome, PerformanceCounter,
    PerformanceOperation, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes,
};
use astronomical_ipc_protocol::RequestId;

use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

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
    pub(super) fn admit_initial_generation_context_or_record_rejection(
        &mut self,
        request_id: RequestId,
        configured_maximum_output_tokens: u16,
        total_context_tokens: usize,
        can_use_persistent_prompt_cache: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, InferenceEngineError> {
        // Return reclaimed expert bytes so request diagnostics can explain work
        // performed specifically to reserve direct cache-publication workspace.
        match self.validate_initial_generation_context_memory_admission(
            total_context_tokens,
            can_use_persistent_prompt_cache,
            performance_attribution,
        ) {
            Ok(reclaimed_expert_payload_bytes) => Ok(reclaimed_expert_payload_bytes),
            Err(context_admission_error) => {
                self.record_generation_performance_attribution(
                    std::mem::replace(performance_attribution, PerformanceAttribution::disabled()),
                    PerformanceAttributionOutcome::Rejected,
                    request_id,
                    configured_maximum_output_tokens,
                    None,
                    Some("generation context admission rejected"),
                );
                Err(context_admission_error)
            }
        }
    }

    pub(super) fn validate_initial_generation_context_memory_admission(
        &mut self,
        total_context_tokens: usize,
        can_use_persistent_prompt_cache: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, InferenceEngineError> {
        let direct_publication_workspace_bytes = if can_use_persistent_prompt_cache {
            self.persistent_prompt_cache_model_contract
                .as_ref()
                .map_or(0, |model_contract| {
                    model_contract.direct_publication_workspace_bytes()
                })
        } else {
            0
        };
        let additional_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        // Cache restore temporarily owns source tensors beside live decoder
        // state. Charge that overlap plus learned prefill activations here so a
        // seated model is demoted before restore, not killed mid-restore.
        let restore_overlap_workspace_bytes = if can_use_persistent_prompt_cache {
            persistent_context_restore_workspace_bytes(
                self.context_memory_reservation_bytes_per_token,
                total_context_tokens,
            )
            .ok_or_else(|| invalid_request_error("prompt-cache restore workspace overflowed"))?
        } else {
            0
        };
        let (prefill_activation_workspace_bytes, complete_layer_scratch_bytes) = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let ram_budget = model.mlx_ram_budget();
            let prefill_activation_workspace_bytes =
                usize::try_from(ram_budget.activation_headroom_bytes(MlxRamBudgetPhase::Prefill))
                    .map_err(|_| {
                    invalid_request_error("prefill activation workspace exceeds the platform range")
                })?;
            let complete_layer_scratch_bytes = usize::try_from(
                ram_budget
                    .model_geometry()
                    .largest_complete_expert_layer_bytes,
            )
            .map_err(|_| {
                invalid_request_error(
                    "complete-layer scratch reservation exceeds the platform range",
                )
            })?;
            (
                prefill_activation_workspace_bytes,
                complete_layer_scratch_bytes,
            )
        };
        let temporary_workspace_reservation_bytes = direct_publication_workspace_bytes
            .checked_add(restore_overlap_workspace_bytes)
            .and_then(|workspace_bytes| {
                workspace_bytes.checked_add(prefill_activation_workspace_bytes)
            })
            .and_then(|workspace_bytes| workspace_bytes.checked_add(complete_layer_scratch_bytes))
            .ok_or_else(|| {
                invalid_request_error("generation context workspace reservation overflowed")
            })?;
        let target_expert_payload_bytes_reclaimed_during_context_admission = self
            .validate_context_memory_admission_with_resident_expert_demotion(
                total_context_tokens,
                temporary_workspace_reservation_bytes,
                additional_maximum_expert_page_reservation_bytes,
                performance_attribution,
            )?;
        if self.speculative_prefill.enabled {
            performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                target_expert_payload_bytes_reclaimed_during_context_admission,
            );
        }
        Ok(if can_use_persistent_prompt_cache {
            target_expert_payload_bytes_reclaimed_during_context_admission
        } else {
            0
        })
    }

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
                adaptive_ram_growth_context.adaptive_ram_growth_phase(),
                AdaptiveRamGrowthPhase::Prefill
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
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
                if model.resident_expert_weights.is_some() {
                    return Err(
                        AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                            reason: "resident expert ownership remained indivisible after demotion"
                                .to_owned(),
                        },
                    );
                }
                let expert_weight_memory_cache_statistics_before_reclamation =
                    model.expert_weight_memory_cache_statistics();
                let retained_expert_payload_bytes = usize::try_from(
                    expert_weight_memory_cache_statistics_before_reclamation
                        .resident_payload_byte_count,
                )
                .unwrap_or(usize::MAX);
                let expert_reclamation_plan = first_forward_projection
                    .expert_retention_reclamation_plan(retained_expert_payload_bytes);
                // Whole-layer eviction may release more than requested, but the pure
                // plan first proves whether the *available byte category* is large
                // enough. An unresolved shortfall cannot be fixed by cache policy.
                if !expert_reclamation_plan.can_satisfy_every_memory_boundary() {
                    log_adaptive_ram_growth_pressure(
                        &first_forward_projection,
                        expert_weight_memory_cache_statistics_before_reclamation,
                        expert_weight_memory_cache_statistics_before_reclamation,
                        memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                        expert_reclamation_plan.reclamation_target_bytes(),
                        "reject",
                    );
                    return Err(
                        AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                            reason: format!(
                                "adaptive RAM growth exceeds reclaimable expert capacity by {} bytes",
                                expert_reclamation_plan.unresolved_shortfall_bytes(),
                            ),
                        },
                    );
                }
                let expert_reclamation_target_bytes =
                    expert_reclamation_plan.reclamation_target_bytes();
                let Some(memory_snapshot_after_reclamation) =
                    reclaim_retained_experts_for_request_memory_pressure(
                        model,
                        expert_reclamation_target_bytes,
                    )?
                else {
                    // A nonzero reclamation target with no paged cache owner means
                    // there is no legal elastic category left. Reject instead of
                    // shrinking the fixed chunk behind the user's configuration.
                    log_adaptive_ram_growth_pressure(
                        &first_forward_projection,
                        expert_weight_memory_cache_statistics_before_reclamation,
                        expert_weight_memory_cache_statistics_before_reclamation,
                        memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                        expert_reclamation_target_bytes,
                        "reject",
                    );
                    return Err(
                        AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                            reason: format!(
                                "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, and recovery projection of {} bytes do not fit C={} bytes and P={} bytes while retained expert paging is unavailable",
                                first_forward_projection.stable_projected_bytes(),
                                first_forward_projection.peak_projected_bytes(),
                                first_forward_projection.recovery_projected_bytes(),
                                first_forward_projection.active_memory_limit_bytes(),
                                first_forward_projection.allowed_active_memory_bytes(),
                            ),
                        },
                    );
                };
                let expert_weight_memory_cache_statistics_after_reclamation =
                    model.expert_weight_memory_cache_statistics();
                log_adaptive_ram_growth_pressure(
                    &first_forward_projection,
                    expert_weight_memory_cache_statistics_before_reclamation,
                    expert_weight_memory_cache_statistics_after_reclamation,
                    memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                    expert_reclamation_target_bytes,
                    "reclaim_experts",
                );
                let projection_after_expert_reclamation = self
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
                if !projection_after_expert_reclamation.fits_stable_and_peak_limits() {
                    // Always verify with a fresh MLX snapshot. The cache's accounting
                    // proves ownership changed, but allocator visibility and unrelated
                    // active arrays still determine whether this operation now fits.
                    log_adaptive_ram_growth_pressure(
                        &projection_after_expert_reclamation,
                        expert_weight_memory_cache_statistics_before_reclamation,
                        expert_weight_memory_cache_statistics_after_reclamation,
                        memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                        expert_reclamation_target_bytes,
                        "reject",
                    );
                    return Err(
                        AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                            reason: format!(
                                "adaptive RAM growth rejected: stable projection of {} bytes, peak projection of {} bytes, or recovery projection of {} bytes remains above C={} bytes or P={} bytes after retained-expert reclamation",
                                projection_after_expert_reclamation.stable_projected_bytes(),
                                projection_after_expert_reclamation.peak_projected_bytes(),
                                projection_after_expert_reclamation.recovery_projected_bytes(),
                                projection_after_expert_reclamation.active_memory_limit_bytes(),
                                projection_after_expert_reclamation.allowed_active_memory_bytes(),
                            ),
                        },
                    );
                }
                log_adaptive_ram_growth_pressure(
                    &projection_after_expert_reclamation,
                    expert_weight_memory_cache_statistics_before_reclamation,
                    expert_weight_memory_cache_statistics_after_reclamation,
                    memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                    expert_reclamation_target_bytes,
                    "admit",
                );
                // Continue with the post-reclamation proof and ownership baseline.
                // Retaining either stale value would publish a reserve or teach
                // post-forward learning from a topology that no longer exists.
                first_forward_projection = projection_after_expert_reclamation;
                memory_snapshot_before_growth = memory_snapshot_after_reclamation;
                retained_expert_payload_bytes_before_growth =
                    expert_weight_memory_cache_statistics_after_reclamation
                        .resident_payload_byte_count;
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
