//! Post-forward MLX sampling and memory-budget learning.
//!
//! Admission and observation are deliberately separate owners. Admission predicts
//! whether a forward may begin; this module runs only after the forward boundary
//! and teaches future admission from what MLX actually retained and peaked at.
//!
//! The sequence is important:
//!
//! 1. Sample active and peak MLX bytes under attribution.
//! 2. Teach the adaptive guard the residual transient window for this context.
//! 3. Publish the phase high-water to expert paging.
//! 4. Teach the composed RAM budget context and activation evidence.
//! 5. Recompute the retained-layer ceiling from that single-source budget.
//!
//! A disabled adaptive guard uses `usize::MAX` as its sentinel baseline. In that
//! case callers may still request a telemetry sample, but no learning or residency
//! limit changes are performed.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5::model::Qwen3_5Model;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthPhase, InferenceEngineError,
    MlxRamBudgetMeasurement, MlxRamBudgetPhase, PerformanceAttribution, PerformanceOperation,
};

use super::qwen3_5_runtime_error;

/// Collects post-forward MLX telemetry and records adaptive learning when enabled.
///
/// The returned snapshot is also used for user-visible memory telemetry. Returning
/// the exact sample used for learning prevents status from presenting a different
/// instant than the one that changed expert-retention policy.
pub(in crate::qwen3_5) fn collect_completed_forward_memory_snapshot(
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    should_retain_adaptive_ram_growth_observation: bool,
    model: &Qwen3_5Model,
    active_memory_bytes_before_growth: usize,
    retained_expert_payload_bytes_before_growth: u64,
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

    // Retained payload is part of the admitted operation snapshot even though
    // transient learning uses post-forward active memory as its stable baseline.
    // Keep it at this boundary so future diagnostics can compare topology before
    // and after a forward without changing every caller again.
    let _retained_expert_payload_bytes_before_growth = retained_expert_payload_bytes_before_growth;
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        adaptive_ram_growth_context,
        should_retain_adaptive_ram_growth_observation,
        active_memory_bytes_before_growth,
        memory_snapshot_after_growth.active_memory_bytes(),
        memory_snapshot_after_growth.peak_memory_bytes(),
        exact_temporary_workspace_bytes,
    );

    // Expert paging needs the phase high-water even when this exact context will
    // not be retained. This prevents warm layers from consuming activation space
    // already proven necessary by another context in the same phase.
    let phase_observed_transient_high_water_bytes = adaptive_ram_growth_guard
        .observed_transient_high_water_bytes(
            adaptive_ram_growth_context.adaptive_ram_growth_phase(),
        );
    model.update_expert_pager_transient_high_water_bytes(
        u64::try_from(phase_observed_transient_high_water_bytes).unwrap_or(u64::MAX),
    );

    if should_retain_adaptive_ram_growth_observation {
        let context_observed_transient_high_water_bytes = adaptive_ram_growth_guard
            .observed_transient_high_water_bytes_for_context(adaptive_ram_growth_context);
        record_composed_ram_budget_measurement(
            adaptive_ram_growth_context,
            model,
            active_memory_bytes_before_growth,
            context_observed_transient_high_water_bytes,
            exact_temporary_workspace_bytes,
            &memory_snapshot_after_growth,
        );
    }
    Ok(Some(memory_snapshot_after_growth))
}

/// Records adaptive learning for a completed operation that needs no snapshot result.
pub(in crate::qwen3_5) fn record_completed_adaptive_ram_growth(
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    should_retain_adaptive_ram_growth_observation: bool,
    model: &Qwen3_5Model,
    active_memory_bytes_before_growth: usize,
    retained_expert_payload_bytes_before_growth: u64,
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
        retained_expert_payload_bytes_before_growth,
        exact_temporary_workspace_bytes,
        performance_attribution,
    )?;
    Ok(())
}

fn record_composed_ram_budget_measurement(
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    model: &Qwen3_5Model,
    active_memory_bytes_before_growth: usize,
    observed_transient_high_water_bytes: usize,
    exact_temporary_workspace_bytes: usize,
    memory_snapshot_after_growth: &MlxMemorySnapshot,
) {
    let mlx_ram_budget_phase = match adaptive_ram_growth_context.adaptive_ram_growth_phase() {
        AdaptiveRamGrowthPhase::Prefill => MlxRamBudgetPhase::Prefill,
        AdaptiveRamGrowthPhase::Decode => MlxRamBudgetPhase::Decode,
    };
    // Peak minus the admitted stable baseline is the request workspace seen by
    // the composed RAM budget. Saturation treats a lower/reset peak as no positive
    // observation rather than unsigned wraparound.
    let measured_context_and_activation_bytes = u64::try_from(
        memory_snapshot_after_growth
            .peak_memory_bytes()
            .saturating_sub(active_memory_bytes_before_growth),
    )
    .unwrap_or(u64::MAX);
    model
        .mlx_ram_budget_mut()
        .record_measurement(MlxRamBudgetMeasurement {
            phase: mlx_ram_budget_phase,
            context_token_count: u64::try_from(adaptive_ram_growth_context.forward_token_count())
                .unwrap_or(u64::MAX),
            measured_context_and_activation_bytes,
            observed_activation_headroom_bytes: u64::try_from(observed_transient_high_water_bytes)
                .unwrap_or(u64::MAX),
            exact_temporary_workspace_bytes: u64::try_from(exact_temporary_workspace_bytes)
                .unwrap_or(u64::MAX),
        });

    // Publish the composed expert budget so retained-layer ownership stays inside
    // the single-source split. `may_grow` gates new warm fills; existing warm
    // layers may remain up to `retained_expert_budget_bytes`. Setting this ceiling
    // to zero merely because the completed operation was multi-token prefill would
    // evict reusable layers without evidence of live pressure.
    let multi_token_prefill = adaptive_ram_growth_context.forward_token_count() > 1
        && matches!(mlx_ram_budget_phase, MlxRamBudgetPhase::Prefill);
    let retained_expert_budget = model.mlx_ram_budget().plan(
        mlx_ram_budget_phase,
        u64::try_from(adaptive_ram_growth_context.forward_token_count()).unwrap_or(u64::MAX),
        0,
        multi_token_prefill,
    );
    if let Some(retained_expert_layers) = model.retained_expert_layers.as_ref() {
        retained_expert_layers
            .borrow_mut()
            .update_maximum_resident_payload_bytes(
                retained_expert_budget.retained_expert_budget_bytes,
            );
    }
}
