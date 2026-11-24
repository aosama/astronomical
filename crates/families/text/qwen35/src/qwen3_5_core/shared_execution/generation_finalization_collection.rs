//! What a finished request reports for either Qwen3.5 engine (issue
//! #2473): the expert memory mode, and the finalized MLX memory telemetry
//! from one snapshot taken after the request's memory is released, so the
//! memory claim and the expert residency claim cannot straddle a promotion
//! or demote (issue #337). An engine adds its own figures through
//! `complete_generation_finalization`.

use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::{
    GenerationFinalization, MlxMemoryTelemetry, PerformanceAttribution, PerformanceOperation,
};

use super::generation_finalization_steps::Qwen3_5FinalizationHooks;
use super::text_model::Qwen3_5TextModel;

/// The finalization report of the request that just ended.
pub fn collect_generation_finalization<Engine: Qwen3_5FinalizationHooks>(
    engine: &Engine,
    performance_attribution: &mut PerformanceAttribution,
    release_peak_memory_bytes: Option<u64>,
) -> GenerationFinalization {
    let Some(model) = engine.execution_core().model.as_ref() else {
        return GenerationFinalization::default();
    };
    let expert_memory_mode = Some(model.observed_expert_memory_mode());
    let mlx_memory_telemetry = match performance_attribution.measure_operation(
        PerformanceOperation::FinalizedMlxMemorySnapshot,
        |_performance_attribution| model.memory_snapshot_for_telemetry(),
    ) {
        Ok(mlx_memory_snapshot) => match (
            u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
            u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes()),
            u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
        ) {
            (
                Ok(active_memory_bytes),
                Ok(mlx_buffer_cache_bytes),
                Ok(sampled_peak_memory_bytes),
            ) => {
                // The MLX peak counter resets at MLX buffer-cache cleanups,
                // so this post-cleanup sample can read a lower window peak
                // than the synchronized release sample taken moments
                // earlier. The finalized claim is a request maximum, so it
                // must retain the higher of the two windows.
                let peak_memory_bytes = match release_peak_memory_bytes {
                    Some(release_peak_memory_bytes) => {
                        release_peak_memory_bytes.max(sampled_peak_memory_bytes)
                    }
                    None => sampled_peak_memory_bytes,
                };
                // The finalized claim derives from the breakdown reconciling
                // this exact snapshot, so the pair cannot straddle a
                // promotion or demote (issue #337).
                let active_memory_breakdown = model.finalized_active_memory_breakdown(
                    active_memory_bytes,
                    engine.retained_live_context_state_payload_bytes(),
                    0,
                );
                let expert_residency_telemetry =
                    Some(model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown));
                (
                    Some(
                        MlxMemoryTelemetry::new(
                            active_memory_bytes,
                            mlx_buffer_cache_bytes,
                            peak_memory_bytes,
                            active_memory_breakdown,
                        )
                        .with_memory_ceiling_utilization(
                            model.between_request_memory_ceiling_utilization_for_breakdown(
                                active_memory_bytes,
                                active_memory_breakdown,
                            ),
                        )
                        .with_memory_ledger_reading(model.latest_memory_ledger_reading()),
                    ),
                    expert_residency_telemetry,
                )
            }
            memory_counter_results => {
                tracing::warn!(
                    ?memory_counter_results,
                    "could not publish finalized MLX memory because a runtime counter exceeds the u64 range"
                );
                (None, None)
            }
        },
        Err(memory_snapshot_error) => {
            tracing::warn!(
                error = %memory_snapshot_error,
                "could not capture finalized MLX memory after expert residency recovery"
            );
            (None, None)
        }
    };
    let (mlx_memory_telemetry, expert_residency_telemetry) = mlx_memory_telemetry;
    engine.complete_generation_finalization(
        GenerationFinalization::new(
            expert_memory_mode,
            mlx_memory_telemetry,
            expert_residency_telemetry,
        ),
        expert_memory_mode,
        expert_residency_telemetry,
    )
}
