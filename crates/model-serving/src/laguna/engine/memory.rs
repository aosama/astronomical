//! Laguna memory-budget composition and measured MLX telemetry.

use astronomical_runtime_integration::{MlxMemorySnapshot, MlxRuntime};

use crate::laguna::{LagunaDecoderState, LagunaModel};
use crate::memory::context_token_bucket;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, InferenceEngineError, MemoryPhase,
    MlxActiveMemoryBreakdown, MlxMemoryTelemetry, MlxRamBudget, MlxRamBudgetMeasurement,
    MlxRamBudgetSnapshot, PerformanceAttribution, PerformanceOperation,
    measured_non_expert_forward_growth_bytes,
};

use super::execution::LagunaInferenceExecution;

/// Active ownership sampled immediately before one Laguna forward.
pub(super) struct LagunaForwardMemoryBaseline {
    active_memory_bytes: u64,
    retained_expert_payload_bytes: u64,
    exact_temporary_workspace_bytes: usize,
}

/// Applies centralized stable/peak admission before one Laguna forward mutates state.
pub(super) fn admit_laguna_forward_memory(
    runtime: &MlxRuntime,
    model: &mut LagunaModel,
    adaptive_ram_growth_guard: &AdaptiveRamGrowthGuard,
    decoder_state: &LagunaDecoderState,
    forward_token_count: usize,
    operation_temporary_workspace_bytes: usize,
    memory_phase: MemoryPhase,
    context_token_count_after_forward: u64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(AdaptiveRamGrowthContext, LagunaForwardMemoryBaseline), InferenceEngineError> {
    let sparse_experts_are_paged = !matches!(
        model.expert_memory_mode(),
        astronomical_ipc_protocol::ExpertMemoryMode::Resident
    );
    let adaptive_ram_growth_context = match memory_phase {
        MemoryPhase::Prefill => AdaptiveRamGrowthContext::prefill(
            forward_token_count,
            context_token_bucket(context_token_count_after_forward),
            false,
            false,
            sparse_experts_are_paged,
        ),
        // The growth guard only observes prefill and decode windows.
        // GenerationPreparation shares the decode window (mapping contract in
        // memory/phase.rs), and Idle never reaches forward observation because
        // a completed forward always belongs to a work phase.
        MemoryPhase::GenerationPreparation | MemoryPhase::Idle | MemoryPhase::Decode => {
            AdaptiveRamGrowthContext::decode(forward_token_count, false, sparse_experts_are_paged)
        }
    };
    let decoder_memory_projection = decoder_state
        .projected_forward_memory(model.contract(), forward_token_count)
        .map_err(|growth_error| InferenceEngineError::Fatal {
            reason: format!("Laguna decoder growth projection failed: {growth_error}"),
        })?;
    let exact_persistent_growth_bytes = decoder_memory_projection.persistent_growth_bytes();
    let exact_temporary_workspace_bytes = decoder_memory_projection
        .sliding_temporary_workspace_bytes()
        .checked_add(operation_temporary_workspace_bytes)
        .ok_or(InferenceEngineError::InvalidRequest {
            reason: "Laguna forward temporary workspace projection overflowed".to_owned(),
        })?;
    let routed_expert_page_reservation_bytes = if sparse_experts_are_paged {
        usize::try_from(model.maximum_expert_page_bytes()).unwrap_or(usize::MAX)
    } else {
        0
    };
    let mut memory_snapshot = performance_attribution
        .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
            runtime.memory_snapshot()
        })
        .map_err(|memory_error| InferenceEngineError::Fatal {
            reason: format!("Laguna forward admission could not sample MLX memory: {memory_error}"),
        })?;
    let mut projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            adaptive_ram_growth_context,
            memory_snapshot.active_memory_bytes(),
            exact_persistent_growth_bytes,
            routed_expert_page_reservation_bytes,
            exact_temporary_workspace_bytes,
        )
        .map_err(|projection_error| InferenceEngineError::InvalidRequest {
            reason: format!("Laguna forward memory projection failed: {projection_error}"),
        })?;
    if !projection.fits_stable_and_peak_limits() {
        if model.native_routed_experts_are_resident() {
            model
                .demote_native_routed_experts(runtime, performance_attribution)
                .map_err(|demotion_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna forward admission could not demote experts: {demotion_error}"
                    ),
                })?;
        }
        let retained_payload_bytes = usize::try_from(
            model
                .expert_weight_memory_cache_statistics()
                .resident_payload_byte_count,
        )
        .unwrap_or(usize::MAX);
        let reclamation_plan = projection.expert_retention_reclamation_plan(retained_payload_bytes);
        if reclamation_plan.required_reclamation_bytes() > 0 {
            model.reclaim_retained_experts_for_request_pressure(
                u64::try_from(reclamation_plan.reclamation_target_bytes()).unwrap_or(u64::MAX),
            );
        }
        performance_attribution
            .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                runtime.synchronize_gpu_stream_and_clear_allocator_cache()
            })
            .map_err(|cleanup_error| InferenceEngineError::Fatal {
                reason: format!("Laguna forward-admission cleanup failed: {cleanup_error}"),
            })?;
        memory_snapshot = performance_attribution
            .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                runtime.memory_snapshot()
            })
            .map_err(|memory_error| InferenceEngineError::Fatal {
                reason: format!(
                    "Laguna forward re-admission could not sample memory: {memory_error}"
                ),
            })?;
        projection = adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context.with_sparse_experts_are_paged(true),
                memory_snapshot.active_memory_bytes(),
                exact_persistent_growth_bytes,
                usize::try_from(model.maximum_expert_page_bytes()).unwrap_or(usize::MAX),
                exact_temporary_workspace_bytes,
            )
            .map_err(|projection_error| InferenceEngineError::InvalidRequest {
                reason: format!("Laguna forward re-admission failed: {projection_error}"),
            })?;
        if !projection.fits_stable_and_peak_limits() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "Laguna forward cannot fit the configured MLX memory ceiling after expert reclamation"
                    .to_owned(),
            });
        }
    }
    tracing::debug!(
        forward_token_count,
        exact_temporary_workspace_bytes,
        stable_projected_bytes = projection.stable_projected_bytes(),
        peak_projected_bytes = projection.peak_projected_bytes(),
        recovery_projected_bytes = projection.recovery_projected_bytes(),
        recovery_reserve_shortfall_bytes = projection.recovery_reserve_shortfall_bytes(),
        "Laguna applied centralized adaptive RAM growth admission"
    );
    let baseline = begin_laguna_forward_memory_observation(
        runtime,
        model,
        exact_temporary_workspace_bytes,
        performance_attribution,
    )?;
    Ok((adaptive_ram_growth_context, baseline))
}

/// Resets the operation peak and captures owners needed to separate promotion from workspace.
pub(super) fn begin_laguna_forward_memory_observation(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    exact_temporary_workspace_bytes: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<LagunaForwardMemoryBaseline, InferenceEngineError> {
    let memory_snapshot = performance_attribution.measure_operation(
        PerformanceOperation::CompletedForwardMemorySnapshot,
        |_performance_attribution| {
            runtime
                .reset_peak_memory()
                .and_then(|()| runtime.memory_snapshot())
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna could not begin forward memory observation: {memory_error}"
                    ),
                })
        },
    )?;
    Ok(LagunaForwardMemoryBaseline {
        active_memory_bytes: u64::try_from(memory_snapshot.active_memory_bytes())
            .unwrap_or(u64::MAX),
        retained_expert_payload_bytes: model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count,
        exact_temporary_workspace_bytes,
    })
}

/// Applies measured forward pressure to the next mandatory expert-read plan.
pub(super) fn complete_laguna_forward_memory_observation(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    mlx_ram_budget: &mut MlxRamBudget,
    memory_baseline: LagunaForwardMemoryBaseline,
    context_token_count_after_forward: u64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxMemorySnapshot, InferenceEngineError> {
    let memory_snapshot = performance_attribution.measure_operation(
        PerformanceOperation::CompletedForwardMemorySnapshot,
        |_performance_attribution| {
            runtime
                .memory_snapshot()
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna could not complete forward memory observation: {memory_error}"
                    ),
                })
        },
    )?;
    let retained_expert_payload_bytes = model
        .expert_weight_memory_cache_statistics()
        .resident_payload_byte_count;
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        adaptive_ram_growth_context,
        true,
        usize::try_from(memory_baseline.active_memory_bytes).unwrap_or(usize::MAX),
        memory_snapshot.active_memory_bytes(),
        memory_snapshot.peak_memory_bytes(),
        memory_baseline.exact_temporary_workspace_bytes,
    );
    let observed_transient_high_water_bytes = adaptive_ram_growth_guard
        .observed_transient_high_water_bytes_for_context(adaptive_ram_growth_context);
    let phase = adaptive_ram_growth_context.memory_phase();
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase,
        context_token_count: context_token_count_after_forward,
        measured_context_and_activation_bytes: measured_non_expert_forward_growth_bytes(
            memory_baseline.active_memory_bytes,
            u64::try_from(memory_snapshot.peak_memory_bytes()).unwrap_or(u64::MAX),
            memory_baseline.retained_expert_payload_bytes,
            retained_expert_payload_bytes,
        ),
        observed_activation_headroom_bytes: u64::try_from(observed_transient_high_water_bytes)
            .unwrap_or(u64::MAX),
        exact_temporary_workspace_bytes: u64::try_from(
            memory_baseline.exact_temporary_workspace_bytes,
        )
        .unwrap_or(u64::MAX),
    });
    let retained_expert_budget_bytes =
        laguna_ram_budget_snapshot(mlx_ram_budget, phase, context_token_count_after_forward)
            .retained_expert_budget_bytes;
    model.update_expert_allocation_transient_high_water(
        u64::try_from(adaptive_ram_growth_guard.admission_transient_high_water_bytes())
            .unwrap_or(u64::MAX),
    );
    model
        .set_retained_expert_ceiling(retained_expert_budget_bytes)
        .map_err(|residency_error| InferenceEngineError::Fatal {
            reason: format!(
                "Laguna could not apply completed-forward expert residency: {residency_error:?}"
            ),
        })?;
    tracing::debug!(
        active_memory_bytes_before_forward = memory_baseline.active_memory_bytes,
        active_memory_bytes_after_forward = memory_snapshot.active_memory_bytes(),
        peak_memory_bytes = memory_snapshot.peak_memory_bytes(),
        retained_expert_payload_bytes_before_forward =
            memory_baseline.retained_expert_payload_bytes,
        retained_expert_payload_bytes_after_forward = retained_expert_payload_bytes,
        retained_expert_budget_bytes,
        "Laguna updated expert residency from completed-forward memory evidence"
    );
    Ok(memory_snapshot)
}

impl LagunaInferenceExecution {
    /// Publishes one allocator snapshot using family-owned model and expert owners.
    pub(super) fn collect_current_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let runtime = self.runtime.as_ref()?;
        let model = self.model.as_ref()?;
        let mlx_ram_budget = self.mlx_ram_budget.as_ref()?;
        let memory_snapshot = match runtime.memory_snapshot() {
            Ok(memory_snapshot) => memory_snapshot,
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "Laguna could not sample current MLX memory"
                );
                return None;
            }
        };
        let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).ok()?;
        let allocator_cache_memory_bytes =
            u64::try_from(memory_snapshot.allocator_cache_memory_bytes()).ok()?;
        let peak_memory_bytes = u64::try_from(memory_snapshot.peak_memory_bytes()).ok()?;
        let expert_payload_bytes = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let context_state_payload_bytes =
            self.active_request.as_ref().map_or(0, |active_request| {
                active_request.decoder_state.payload_byte_count()
            });
        let model_core_payload_bytes = mlx_ram_budget.model_geometry().model_core_payload_bytes;
        let active_memory_breakdown = MlxActiveMemoryBreakdown::reconcile(
            active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        );
        // This snapshot can run once per generated token. Keep detailed owner
        // reconciliation at trace level so ordinary serving does not add hot-path I/O.
        tracing::trace!(
            active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            reported_context_state_payload_bytes = context_state_payload_bytes,
            attributed_context_state_payload_bytes =
                active_memory_breakdown.context_state_payload_bytes,
            unattributed_active_memory_bytes = active_memory_bytes
                .saturating_sub(active_memory_breakdown.expert_payload_bytes)
                .saturating_sub(active_memory_breakdown.model_core_payload_bytes)
                .saturating_sub(active_memory_breakdown.context_state_payload_bytes),
            "Laguna reconciled one MLX active-memory snapshot"
        );
        Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            active_memory_breakdown,
        ))
    }
}

/// Composes Laguna geometry through the shared model-core, context, activation, and page budget.
pub(super) fn laguna_ram_budget_snapshot(
    mlx_ram_budget: &MlxRamBudget,
    phase: MemoryPhase,
    context_token_count: u64,
) -> MlxRamBudgetSnapshot {
    mlx_ram_budget.plan(phase, context_token_count, 0)
}
