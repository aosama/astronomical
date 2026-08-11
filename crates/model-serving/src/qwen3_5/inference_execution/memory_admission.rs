use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5::decoder::RequestDecoderStateStack;
use crate::qwen3_5::model::Qwen3_5Model;
use crate::qwen3_5::model::adaptive_ram_growth_logging::log_adaptive_ram_growth_pressure;
use crate::qwen3_5::model::memory_admission::{
    combined_target_and_additional_persistent_growth_bytes,
    context_memory_admission_fits_without_expert_reclamation, invalid_request_error,
    validate_context_memory_admission,
};
use crate::qwen3_5_moe::{
    Qwen3_5ExpertResidencyTransitionReason, reclaim_retained_experts_for_request_memory_pressure,
};
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceCounter, PerformanceOperation,
};

use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

pub(in crate::qwen3_5) enum AdaptiveRamGrowthMemoryAdmissionError {
    InsufficientCapacity { reason: String },
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
                if let Some(model) = self.model.as_mut()
                    && let Err(resident_restoration_error) = model.try_promote_experts_to_resident(
                        Qwen3_5ExpertResidencyTransitionReason::RequestFinalization,
                        performance_attribution,
                    )
                {
                    tracing::warn!(
                        request_id = request_id.value(),
                        error = %resident_restoration_error,
                        "could not restore idle expert residency after request admission rejection"
                    );
                }
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
        let target_expert_payload_bytes_before_context_admission = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
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
        // First ask whether exact context, publication workspace, and any draft
        // page fit beside the complete owner. Demote only when that same request
        // would otherwise fail; small requests keep the faster resident path.
        let resident_model_requires_demote = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            model.resident_expert_weights.is_some()
                && !context_memory_admission_fits_without_expert_reclamation(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    total_context_tokens,
                    direct_publication_workspace_bytes,
                    additional_maximum_expert_page_reservation_bytes,
                )?
        };
        if resident_model_requires_demote {
            self.model
                .as_mut()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                .demote_resident_experts_to_paging(
                    Qwen3_5ExpertResidencyTransitionReason::RequestAdmission,
                    performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let context_admission_outcome = performance_attribution.measure_operation(
            PerformanceOperation::MemoryAdmissionSnapshot,
            |_performance_attribution| {
                validate_context_memory_admission(
                    model,
                    self.memory_limits,
                    self.context_memory_reservation_bytes_per_token,
                    total_context_tokens,
                    direct_publication_workspace_bytes,
                    additional_maximum_expert_page_reservation_bytes,
                )
            },
        );
        let target_expert_payload_bytes_after_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let target_expert_payload_bytes_reclaimed_during_context_admission =
            target_expert_payload_bytes_before_context_admission
                .saturating_sub(target_expert_payload_bytes_after_context_admission);
        if self.speculative_prefill.enabled {
            performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                target_expert_payload_bytes_reclaimed_during_context_admission,
            );
        }
        context_admission_outcome?;
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
    ) -> Result<usize, AdaptiveRamGrowthMemoryAdmissionError> {
        if !self.adaptive_ram_growth_guard_enabled {
            return Ok(usize::MAX);
        }
        performance_attribution.measure_operation(
            PerformanceOperation::AdaptiveRamGrowthMemoryAdmission,
            |_performance_attribution| {
                self.begin_adaptive_ram_growth(
                    adaptive_ram_growth_context,
                    request_decoder_state,
                    additional_persistent_state_growth_bytes,
                    exact_temporary_workspace_bytes,
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
    ) -> Result<usize, AdaptiveRamGrowthMemoryAdmissionError> {
        let (
            target_persistent_state_growth_bytes,
            mut routed_expert_page_reservation_bytes,
            mut memory_snapshot_before_growth,
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
                model
                    .expert_pager
                    .as_ref()
                    .map_or(0, |expert_pager| expert_pager.maximum_expert_page_bytes())
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
            (
                target_persistent_state_growth_bytes,
                routed_expert_page_reservation_bytes,
                memory_snapshot_before_growth,
            )
        };
        let exact_persistent_growth_bytes = combined_target_and_additional_persistent_growth_bytes(
            target_persistent_state_growth_bytes,
            additional_persistent_state_growth_bytes,
        )?;
        let mut initial_adaptive_ram_growth_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_persistent_growth_bytes,
                routed_expert_page_reservation_bytes,
                exact_temporary_workspace_bytes,
            )
            .map_err(|adaptive_ram_growth_projection_error| {
                tracing::warn!(
                    action = "reject",
                    current_active_memory_bytes =
                        memory_snapshot_before_growth.active_memory_bytes(),
                    exact_persistent_growth_bytes,
                    error = %adaptive_ram_growth_projection_error,
                    "stopped Qwen3.5 forward after adaptive RAM growth projection failed"
                );
                invalid_request_error(format!(
                    "adaptive RAM growth rejected: {adaptive_ram_growth_projection_error}"
                ))
            })?;

        if initial_adaptive_ram_growth_projection.fits_stable_and_peak_limits() {
            if !initial_adaptive_ram_growth_projection.has_full_recovery_reserve() {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
                let expert_weight_memory_cache_statistics =
                    model.expert_weight_memory_cache_statistics();
                let pressure_action =
                    if model.freeze_expert_retention_growth_for_request_memory_pressure() {
                        "freeze_retention_growth"
                    } else {
                        "admit"
                    };
                log_adaptive_ram_growth_pressure(
                    &initial_adaptive_ram_growth_projection,
                    expert_weight_memory_cache_statistics,
                    expert_weight_memory_cache_statistics,
                    memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                    0,
                    pressure_action,
                );
            }
        } else {
            // Later forwards can outgrow an initially safe resident request.
            // Retry from a fresh paged baseline before evicting individual pages
            // or rejecting, preserving the same growth evidence and user work.
            if let Some(resident_demotion) = self.demote_resident_experts_for_adaptive_growth(
                adaptive_ram_growth_context,
                exact_persistent_growth_bytes,
                exact_temporary_workspace_bytes,
            )? {
                adaptive_ram_growth_context = resident_demotion.adaptive_ram_growth_context;
                memory_snapshot_before_growth = resident_demotion.memory_snapshot;
                routed_expert_page_reservation_bytes =
                    resident_demotion.routed_expert_page_reservation_bytes;
                initial_adaptive_ram_growth_projection = resident_demotion.projection;
                if initial_adaptive_ram_growth_projection.fits_stable_and_peak_limits() {
                    let model_after_demotion = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    if !initial_adaptive_ram_growth_projection.has_full_recovery_reserve() {
                        model_after_demotion
                            .freeze_expert_retention_growth_for_request_memory_pressure();
                    }
                    model_after_demotion
                        .runtime()
                        .reset_peak_memory()
                        .map_err(qwen3_5_runtime_error)?;
                    return Ok(memory_snapshot_before_growth.active_memory_bytes());
                }
            }
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let expert_weight_memory_cache_statistics_before_reclamation =
                model.expert_weight_memory_cache_statistics();
            let required_reclamation_bytes =
                initial_adaptive_ram_growth_projection.required_reclamation_bytes();
            let Some(memory_snapshot_after_reclamation) =
                reclaim_retained_experts_for_request_memory_pressure(
                    model,
                    required_reclamation_bytes,
                )?
            else {
                log_adaptive_ram_growth_pressure(
                    &initial_adaptive_ram_growth_projection,
                    expert_weight_memory_cache_statistics_before_reclamation,
                    expert_weight_memory_cache_statistics_before_reclamation,
                    memory_snapshot_before_growth.allocator_cache_memory_bytes(),
                    required_reclamation_bytes,
                    "reject",
                );
                return Err(
                    AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                        reason: format!(
                            "adaptive RAM growth rejected: stable projection of {} bytes and peak projection of {} bytes do not fit C={} bytes and P={} bytes while retained expert paging is unavailable",
                            initial_adaptive_ram_growth_projection.stable_projected_bytes(),
                            initial_adaptive_ram_growth_projection.peak_projected_bytes(),
                            initial_adaptive_ram_growth_projection.active_memory_limit_bytes(),
                            initial_adaptive_ram_growth_projection.allowed_active_memory_bytes(),
                        ),
                    },
                );
            };
            let expert_weight_memory_cache_statistics_after_reclamation =
                model.expert_weight_memory_cache_statistics();
            log_adaptive_ram_growth_pressure(
                &initial_adaptive_ram_growth_projection,
                expert_weight_memory_cache_statistics_before_reclamation,
                expert_weight_memory_cache_statistics_after_reclamation,
                memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                required_reclamation_bytes,
                "reclaim_experts",
            );
            let post_reclamation_adaptive_ram_growth_projection = self
                .adaptive_ram_growth_guard
                .project_growth_for_context(
                    adaptive_ram_growth_context,
                    memory_snapshot_after_reclamation.active_memory_bytes(),
                    exact_persistent_growth_bytes,
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
            if !post_reclamation_adaptive_ram_growth_projection.fits_stable_and_peak_limits() {
                log_adaptive_ram_growth_pressure(
                    &post_reclamation_adaptive_ram_growth_projection,
                    expert_weight_memory_cache_statistics_before_reclamation,
                    expert_weight_memory_cache_statistics_after_reclamation,
                    memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                    required_reclamation_bytes,
                    "reject",
                );
                return Err(
                    AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                        reason: format!(
                            "adaptive RAM growth rejected: stable projection of {} bytes and peak projection of {} bytes remain above C={} bytes or P={} bytes after retained-expert reclamation",
                            post_reclamation_adaptive_ram_growth_projection
                                .stable_projected_bytes(),
                            post_reclamation_adaptive_ram_growth_projection.peak_projected_bytes(),
                            post_reclamation_adaptive_ram_growth_projection
                                .active_memory_limit_bytes(),
                            post_reclamation_adaptive_ram_growth_projection
                                .allowed_active_memory_bytes(),
                        ),
                    },
                );
            }
            log_adaptive_ram_growth_pressure(
                &post_reclamation_adaptive_ram_growth_projection,
                expert_weight_memory_cache_statistics_before_reclamation,
                expert_weight_memory_cache_statistics_after_reclamation,
                memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
                required_reclamation_bytes,
                "admit",
            );
            memory_snapshot_before_growth = memory_snapshot_after_reclamation;
        }
        // MLX's peak counter is process-global. Reset it only after admission so
        // the next sample measures this one forward pass rather than model loading
        // or an earlier prefill chunk.
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        Ok(memory_snapshot_before_growth.active_memory_bytes())
    }
}

/// Collects post-forward MLX telemetry and records adaptive learning when enabled.
pub(in crate::qwen3_5) fn collect_completed_forward_memory_snapshot(
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    should_retain_adaptive_ram_growth_observation: bool,
    model: &Qwen3_5Model,
    active_memory_bytes_before_growth: usize,
    exact_temporary_workspace_bytes: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<MlxMemorySnapshot>, InferenceEngineError> {
    let memory_snapshot_after_growth = performance_attribution.measure_operation(
        PerformanceOperation::CompletedForwardMemorySnapshot,
        |_performance_attribution| {
            model
                .runtime()
                .memory_snapshot()
                .map_err(qwen3_5_runtime_error)
        },
    )?;
    if active_memory_bytes_before_growth == usize::MAX {
        return Ok(Some(memory_snapshot_after_growth));
    }
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        adaptive_ram_growth_context,
        should_retain_adaptive_ram_growth_observation,
        active_memory_bytes_before_growth,
        memory_snapshot_after_growth.active_memory_bytes(),
        memory_snapshot_after_growth.peak_memory_bytes(),
        exact_temporary_workspace_bytes,
    );
    Ok(Some(memory_snapshot_after_growth))
}

/// Records adaptive learning without sampling when admission is disabled.
pub(in crate::qwen3_5) fn record_completed_adaptive_ram_growth(
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    should_retain_adaptive_ram_growth_observation: bool,
    model: &Qwen3_5Model,
    active_memory_bytes_before_growth: usize,
    exact_temporary_workspace_bytes: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    if active_memory_bytes_before_growth == usize::MAX {
        return Ok(());
    }
    collect_completed_forward_memory_snapshot(
        adaptive_ram_growth_guard,
        adaptive_ram_growth_context,
        should_retain_adaptive_ram_growth_observation,
        model,
        active_memory_bytes_before_growth,
        exact_temporary_workspace_bytes,
        performance_attribution,
    )?;
    Ok(())
}
