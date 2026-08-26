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
use crate::qwen3_5_moe::model::record_expert_reclamation_attribution;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthPhase, InferenceEngineError,
    MlxRamBudgetMeasurement, MlxRamBudgetPhase, PerformanceAttribution, PerformanceOperation,
    RetainedExpertReclamation, measured_non_expert_forward_growth_bytes,
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
        let budget_reclamation = record_composed_ram_budget_measurement(
            adaptive_ram_growth_context,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            context_observed_transient_high_water_bytes,
            exact_temporary_workspace_bytes,
            &memory_snapshot_after_growth,
        );
        record_expert_reclamation_attribution(performance_attribution, budget_reclamation);
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
    retained_expert_payload_bytes_before_growth: u64,
    observed_transient_high_water_bytes: usize,
    exact_temporary_workspace_bytes: usize,
    memory_snapshot_after_growth: &MlxMemorySnapshot,
) -> RetainedExpertReclamation {
    let mlx_ram_budget_phase = match adaptive_ram_growth_context.adaptive_ram_growth_phase() {
        AdaptiveRamGrowthPhase::Prefill => MlxRamBudgetPhase::Prefill,
        AdaptiveRamGrowthPhase::Decode => MlxRamBudgetPhase::Decode,
    };
    // Peak includes complete/routed pages promoted by mandatory reads. Subtract
    // only newly retained payload so the composed budget learns request workspace
    // without reserving the same expert ownership a second time.
    let retained_expert_payload_bytes_after_growth = model
        .expert_weight_memory_cache_statistics()
        .resident_payload_byte_count;
    let measured_context_and_activation_bytes = measured_non_expert_forward_growth_bytes(
        u64::try_from(active_memory_bytes_before_growth).unwrap_or(u64::MAX),
        u64::try_from(memory_snapshot_after_growth.peak_memory_bytes()).unwrap_or(u64::MAX),
        retained_expert_payload_bytes_before_growth,
        retained_expert_payload_bytes_after_growth,
    );
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
    // the single-source split. Execution policy—not this numeric owner—decides
    // whether a mandatory page remains operation-local or transfers to retention.
    let retained_expert_budget = model.mlx_ram_budget().plan(
        mlx_ram_budget_phase,
        u64::try_from(adaptive_ram_growth_context.forward_token_count()).unwrap_or(u64::MAX),
        0,
    );
    model.retained_experts.as_ref().map_or_else(
        RetainedExpertReclamation::default,
        |retained_experts| {
            retained_experts
                .borrow_mut()
                .update_maximum_resident_payload_bytes(
                    retained_expert_budget.retained_expert_budget_bytes,
                )
        },
    )
}
