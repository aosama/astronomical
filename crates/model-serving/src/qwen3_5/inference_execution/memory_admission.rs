use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5::decoder::RequestDecoderStateStack;
use crate::qwen3_5::model::Qwen3_5Model;
use crate::qwen3_5::model::adaptive_ram_growth_logging::log_adaptive_ram_growth_pressure;
use crate::qwen3_5::model::memory_admission::{
    combined_target_and_mtp_persistent_growth_bytes, invalid_request_error,
};
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, InferenceEngineError, PerformanceAttribution,
    PerformanceOperation,
};

use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

pub(super) enum AdaptiveRamGrowthMemoryAdmissionError {
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
    /// Attributes adaptive admission, including any retained-expert reclamation.
    pub(super) fn measure_adaptive_ram_growth_memory_admission(
        &self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        performance_attribution: &mut PerformanceAttribution,
        request_decoder_state: &RequestDecoderStateStack,
        mtp_full_attention_growth_bytes: usize,
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
                    mtp_full_attention_growth_bytes,
                    exact_temporary_workspace_bytes,
                )
            },
        )
    }

    /// Admits one forward pass and starts an operation-local MLX peak sample.
    fn begin_adaptive_ram_growth(
        &self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        request_decoder_state: &RequestDecoderStateStack,
        mtp_full_attention_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<usize, AdaptiveRamGrowthMemoryAdmissionError> {
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
        let exact_persistent_growth_bytes = combined_target_and_mtp_persistent_growth_bytes(
            target_persistent_state_growth_bytes,
            mtp_full_attention_growth_bytes,
        )?;
        let mut memory_snapshot_before_growth = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let initial_adaptive_ram_growth_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_persistent_growth_bytes,
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
        model
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        Ok(memory_snapshot_before_growth.active_memory_bytes())
    }

    /// Drops request-owned arrays before asking MLX to return cached allocations.
    ///
    /// Clearing here is deliberately request-scoped: allocator reuse remains
    /// available inside prefill and decode, while completed work cannot leave a
    /// large cache resident indefinitely on a memory-constrained laptop.
    pub(crate) fn release_request_memory(
        &self,
        request_id: RequestId,
        should_capture_memory_snapshot: bool,
    ) -> Option<MlxMemorySnapshot> {
        let model = self.model.as_ref()?;
        if let Err(allocator_cache_error) = model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
        {
            tracing::warn!(request_id = request_id.value(), error = %allocator_cache_error,
                "failed to release reclaimable MLX request memory");
            return None;
        }
        if !should_capture_memory_snapshot {
            return None;
        }
        match model.runtime().memory_snapshot() {
            Ok(mlx_memory_snapshot) => {
                tracing::info!(
                    request_id = request_id.value(),
                    mlx_active_bytes = mlx_memory_snapshot.active_memory_bytes(),
                    mlx_allocator_cache_bytes = mlx_memory_snapshot.allocator_cache_memory_bytes(),
                    mlx_peak_bytes = mlx_memory_snapshot.peak_memory_bytes(),
                    "released reclaimable MLX request memory"
                );
                Some(mlx_memory_snapshot)
            }
            Err(snapshot_error) => {
                tracing::warn!(
                    request_id = request_id.value(), error = %snapshot_error,
                    "released MLX request memory but could not sample allocator metrics"
                );
                None
            }
        }
    }
}

/// Collects post-forward MLX telemetry and records adaptive learning when enabled.
pub(super) fn collect_completed_forward_memory_snapshot(
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
pub(super) fn record_completed_adaptive_ram_growth(
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
