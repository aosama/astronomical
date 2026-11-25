//! What a finished request reports for any text engine (issue
//! #2473): the expert memory mode, and the finalized MLX memory telemetry
//! from one snapshot taken after the request's memory is released, so the
//! memory claim and the expert residency claim cannot straddle a promotion
//! or demote (issue #337). An engine adds its own figures through
//! `complete_generation_finalization`.

use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::model::memory_breakdown::TextModelMemoryOwnership;
use crate::{
    GenerationFinalization, MlxMemoryTelemetry, PerformanceAttribution, PerformanceOperation,
};

use super::generation_finalization_steps::{
    MlxMemoryCounters, TextFinalizationHooks, between_request_mlx_memory_telemetry,
};
use crate::model::text_model::TextModel;

/// The finalized memory claim and the expert residency claim derived from
/// the same snapshot, both absent when the snapshot could not be read.
struct FinalizedMemoryClaims {
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
}

impl FinalizedMemoryClaims {
    const UNAVAILABLE: Self = Self {
        mlx_memory_telemetry: None,
        expert_residency_telemetry: None,
    };
}

/// The finalization report of the request that just ended.
pub fn collect_generation_finalization<Engine: TextFinalizationHooks>(
    engine: &Engine,
    performance_attribution: &mut PerformanceAttribution,
    release_peak_memory_bytes: Option<u64>,
) -> GenerationFinalization {
    let Some(model) = engine.execution_core().model.as_ref() else {
        return GenerationFinalization::default();
    };
    let expert_memory_mode = Some(model.observed_expert_memory_mode());
    let finalized_memory_claims = match performance_attribution.measure_operation(
        PerformanceOperation::FinalizedMlxMemorySnapshot,
        |_performance_attribution| model.memory_snapshot_for_telemetry(),
    ) {
        Ok(mlx_memory_snapshot) => finalized_memory_claims_from_snapshot(
            engine,
            &mlx_memory_snapshot,
            release_peak_memory_bytes,
        ),
        Err(memory_snapshot_error) => {
            tracing::warn!(
                error = %memory_snapshot_error,
                "could not capture finalized MLX memory after expert residency recovery"
            );
            FinalizedMemoryClaims::UNAVAILABLE
        }
    };
    engine.complete_generation_finalization(
        GenerationFinalization::new(
            expert_memory_mode,
            finalized_memory_claims.mlx_memory_telemetry,
            finalized_memory_claims.expert_residency_telemetry,
        ),
        expert_memory_mode,
        finalized_memory_claims.expert_residency_telemetry,
    )
}

/// Both claims from one snapshot, or none when a runtime counter does not fit
/// the u64 range the report carries.
fn finalized_memory_claims_from_snapshot<Engine: TextFinalizationHooks>(
    engine: &Engine,
    mlx_memory_snapshot: &runtime_integration::MlxMemorySnapshot,
    release_peak_memory_bytes: Option<u64>,
) -> FinalizedMemoryClaims {
    let memory_counter_results = (
        u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
        u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes()),
        u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
    );
    let (Ok(active_memory_bytes), Ok(mlx_buffer_cache_bytes), Ok(sampled_peak_memory_bytes)) =
        memory_counter_results
    else {
        tracing::warn!(
            ?memory_counter_results,
            "could not publish finalized MLX memory because a runtime counter exceeds the u64 range"
        );
        return FinalizedMemoryClaims::UNAVAILABLE;
    };
    // The MLX peak counter resets at MLX buffer-cache cleanups, so this
    // post-cleanup sample can read a lower window peak than the synchronized
    // release sample taken moments earlier. The finalized claim is a request
    // maximum, so it must retain the higher of the two windows.
    let peak_memory_bytes = release_peak_memory_bytes
        .map_or(sampled_peak_memory_bytes, |release_peak_memory_bytes| {
            release_peak_memory_bytes.max(sampled_peak_memory_bytes)
        });
    let Some(model) = engine.execution_core().model.as_ref() else {
        return FinalizedMemoryClaims::UNAVAILABLE;
    };
    // The finalized claim derives from the breakdown reconciling this exact
    // snapshot, so the pair cannot straddle a promotion or demote (issue #337).
    let active_memory_breakdown = model.finalized_active_memory_breakdown(
        active_memory_bytes,
        engine.retained_live_context_state_payload_bytes(),
        0,
    );
    let mlx_memory_telemetry = between_request_mlx_memory_telemetry(
        model,
        MlxMemoryCounters {
            active_memory_bytes,
            mlx_buffer_cache_bytes,
            peak_memory_bytes,
        },
        active_memory_breakdown,
    );
    FinalizedMemoryClaims {
        mlx_memory_telemetry: Some(mlx_memory_telemetry),
        expert_residency_telemetry: Some(
            model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
        ),
    }
}
