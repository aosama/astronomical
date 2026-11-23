use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    InferenceEngineError, ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution,
    PerformanceAttributionOutcome,
};

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    pub(super) fn engine_load_result(
        &self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> EngineLoadResult {
        // Build readiness only after startup promotion has selected the owner.
        // The worker forwards this value instead of inferring mode from memory.
        EngineLoadResult::new()
            .with_expert_memory_mode(
                self.core
                    .model
                    .as_ref()
                    .map(|loaded_model| loaded_model.expert_memory_mode()),
            )
            .with_minimum_mlx_memory_ceiling_bytes(minimum_mlx_memory_ceiling_bytes)
            .with_neural_engine_prefill_status(self.neural_engine_prefill_status.clone())
            .with_draft_model_status(self.draft_model_status.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_model_loading_performance_attribution(
        &mut self,
        model_loading_performance_attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        model_id: Option<String>,
        model_revision: Option<String>,
        total_artifact_payload_bytes: Option<u64>,
        resident_model_payload_bytes: Option<u64>,
        model_shard_count: Option<usize>,
        mlx_memory_snapshot: Option<runtime_integration::MlxMemorySnapshot>,
        failure_description: Option<String>,
    ) -> Result<(), InferenceEngineError> {
        let Some(performance_attribution_report) = model_loading_performance_attribution
            .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome,
                model_id,
                model_revision,
                prefill_transient_observation_completed: self
                    .measured_prompt_chunk_mlx_working_buffer_bytes()
                    .is_some(),
                prefill_observed_transient_high_water_bytes: self
                    .measured_prompt_chunk_mlx_working_buffer_bytes()
                    .unwrap_or(0),
                total_artifact_payload_bytes,
                resident_model_payload_bytes,
                model_shard_count,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_buffer_cache_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.mlx_buffer_cache_bytes()).ok()),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description,
            })
        else {
            return Ok(());
        };
        if let Err(performance_attribution_write_error) = self
            .core
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
