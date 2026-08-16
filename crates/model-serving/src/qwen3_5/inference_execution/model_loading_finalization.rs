use crate::{
    EngineLoadResult, InferenceEngineError, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionOutcome,
};

use super::{Qwen3_5EngineState, Qwen3_5MtpRuntimeState};

impl Qwen3_5EngineState {
    pub(super) fn engine_load_result_for_mtp_state(
        &self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> EngineLoadResult {
        let mtp_runtime_state = match self.mtp_runtime_state {
            Qwen3_5MtpRuntimeState::Disabled => {
                astronomical_ipc_protocol::MtpRuntimeState::Disabled
            }
            Qwen3_5MtpRuntimeState::TargetOnly => {
                astronomical_ipc_protocol::MtpRuntimeState::TargetOnly
            }
            Qwen3_5MtpRuntimeState::Active => astronomical_ipc_protocol::MtpRuntimeState::Active,
            Qwen3_5MtpRuntimeState::Unavailable => {
                astronomical_ipc_protocol::MtpRuntimeState::Unavailable
            }
        };
        // Build readiness only after startup promotion has selected the owner.
        // The worker forwards this value instead of inferring mode from memory.
        let mut engine_load_result = EngineLoadResult::new()
            .with_expert_memory_mode(
                self.model
                    .as_ref()
                    .map(|loaded_model| loaded_model.expert_memory_mode()),
            )
            .with_mtp_runtime_state(mtp_runtime_state)
            .with_mtp_depth_status(self.mtp_depth_status)
            .with_speculative_prefill_runtime(
                self.speculative_prefill_runtime_state,
                self.speculative_prefill_unavailable_reason.clone(),
                self.speculative_prefill.draft_model_id.clone(),
                self.speculative_prefill_draft_model_revision.clone(),
            );
        if self.mtp_runtime_state == Qwen3_5MtpRuntimeState::Unavailable {
            if let Some(mtp_unavailable_reason) = self.mtp_unavailable_reason.as_ref() {
                engine_load_result =
                    engine_load_result.with_mtp_unavailable_reason(mtp_unavailable_reason.clone());
            }
        }
        engine_load_result.with_minimum_mlx_memory_ceiling_bytes(minimum_mlx_memory_ceiling_bytes)
    }

    pub(super) fn record_model_loading_performance_attribution(
        &mut self,
        model_loading_performance_attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        model_id: Option<String>,
        model_revision: Option<String>,
        total_artifact_payload_bytes: Option<u64>,
        resident_model_payload_bytes: Option<u64>,
        model_shard_count: Option<usize>,
        mlx_memory_snapshot: Option<astronomical_runtime_integration::MlxMemorySnapshot>,
        failure_description: Option<String>,
    ) -> Result<(), InferenceEngineError> {
        let Some(performance_attribution_report) = model_loading_performance_attribution
            .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome,
                model_id,
                model_revision,
                prefill_transient_observation_completed: self
                    .adaptive_ram_growth_guard
                    .has_completed_growth_observation(crate::AdaptiveRamGrowthPhase::Prefill),
                prefill_observed_transient_high_water_bytes: u64::try_from(
                    self.adaptive_ram_growth_guard
                        .observed_transient_high_water_bytes(
                            crate::AdaptiveRamGrowthPhase::Prefill,
                        ),
                )
                .unwrap_or(u64::MAX),
                total_artifact_payload_bytes,
                resident_model_payload_bytes,
                model_shard_count,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description,
            })
        else {
            return Ok(());
        };
        if let Err(performance_attribution_write_error) = self
            .performance_attribution_log
            .record(&performance_attribution_report)
        {
            tracing::warn!(
                error = %performance_attribution_write_error,
                "failed to append model-loading performance attribution"
            );
        }
        Ok(())
    }
}
