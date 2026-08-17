//! Laguna memory-budget composition and measured MLX telemetry.

use astronomical_runtime_integration::{MlxMemorySnapshot, MlxRuntime};

use crate::laguna::{LagunaModel, laguna_retained_expert_budget_after_completed_forward};
use crate::{
    InferenceEngineError, MlxActiveMemoryBreakdown, MlxMemoryTelemetry, MlxRamBudget,
    MlxRamBudgetPhase, MlxRamBudgetSnapshot, PerformanceAttribution, PerformanceOperation,
};

use super::execution::LagunaInferenceExecution;

/// Active ownership sampled immediately before one Laguna forward.
pub(super) struct LagunaForwardMemoryBaseline {
    active_memory_bytes: u64,
    retained_expert_payload_bytes: u64,
}

/// Resets the operation peak and captures owners needed to separate promotion from workspace.
pub(super) fn begin_laguna_forward_memory_observation(
    runtime: &MlxRuntime,
    model: &LagunaModel,
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
    })
}

/// Applies measured forward pressure to the next mandatory expert-read plan.
pub(super) fn complete_laguna_forward_memory_observation(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    mlx_ram_budget: &MlxRamBudget,
    memory_baseline: LagunaForwardMemoryBaseline,
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
    let retained_expert_budget_bytes = laguna_retained_expert_budget_after_completed_forward(
        mlx_ram_budget,
        u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(u64::MAX),
        u64::try_from(memory_snapshot.peak_memory_bytes()).unwrap_or(u64::MAX),
        retained_expert_payload_bytes,
        mlx_ram_budget
            .model_geometry()
            .complete_expert_payload_bytes,
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

/// Composes Laguna's retained expert allowance with a laptop-relative reserve.
///
/// The common budget protects model core, context, activations, and one streamed
/// page. Laguna additionally reserves room for its first large prompt graph so
/// the policy adapts to the configured machine ceiling instead of one laptop.
pub(super) fn laguna_ram_budget_snapshot(
    mlx_ram_budget: &MlxRamBudget,
    phase: MlxRamBudgetPhase,
    context_token_count: u64,
) -> MlxRamBudgetSnapshot {
    let request_operational_reserve_bytes = mlx_ram_budget.mlx_active_memory_ceiling_bytes() / 8;
    let mut ram_budget_snapshot = mlx_ram_budget.plan(
        phase,
        context_token_count,
        request_operational_reserve_bytes,
    );
    let model_geometry = mlx_ram_budget.model_geometry();
    // Synchronous MLX evaluation detaches an evaluated layer when the next graph
    // consumes it. Paged execution therefore needs one slot for that boundary
    // output and a second slot for the next mandatory page allocation.
    let two_layer_stream_reserve_bytes = model_geometry
        .largest_complete_expert_layer_bytes
        .saturating_mul(2);
    let maximum_safe_retained_expert_payload_bytes = model_geometry
        .complete_expert_payload_bytes
        .saturating_sub(two_layer_stream_reserve_bytes);
    // Keep at least half the resolved MLX ceiling for attention, key/value state,
    // routing, and the evaluated expert boundary. The common budget may reserve more.
    let maximum_paged_expert_share_bytes = mlx_ram_budget.mlx_active_memory_ceiling_bytes() / 2;
    ram_budget_snapshot.retained_expert_budget_bytes = ram_budget_snapshot
        .retained_expert_budget_bytes
        .min(maximum_safe_retained_expert_payload_bytes)
        .min(maximum_paged_expert_share_bytes);
    ram_budget_snapshot
}
