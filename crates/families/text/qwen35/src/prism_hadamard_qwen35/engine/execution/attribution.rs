//! Model-load attribution report finalization for the folded-ternary engine.
//! A request's attribution is recorded by the shared Qwen 3.5 finalization.

use crate::{
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution,
    PerformanceAttributionOutcome,
};

use super::PrismHadamardQwen35InferenceExecution;

impl PrismHadamardQwen35InferenceExecution {
    pub(super) fn record_model_loading_performance_attribution(
        &mut self,
        performance_attribution: PerformanceAttribution,
    ) {
        if !performance_attribution.is_enabled() {
            return;
        }
        let mlx_memory_snapshot = self
            .execution_core
            .model
            .as_ref()
            .and_then(|model| model.runtime.memory_snapshot().ok());
        let Some(report) = performance_attribution.finish_model_loading(
            ModelLoadingPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id: self.execution_core.model_id.clone(),
                model_revision: self.execution_core.model_revision.clone(),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes: self
                    .execution_core
                    .model
                    .as_ref()
                    .map(|model| model.pack().total_payload_bytes()),
                resident_model_payload_bytes: self
                    .execution_core
                    .model
                    .as_ref()
                    .map(|model| model.pack().total_payload_bytes()),
                model_shard_count: Some(1),
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_buffer_cache_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.mlx_buffer_cache_bytes()).ok()),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: None,
            },
        ) else {
            return;
        };
        if let Err(write_error) = self
            .execution_core
            .performance_attribution_log
            .record(&report)
        {
            tracing::warn!(
                error = %write_error,
                "folded-ternary model-loading attribution could not be recorded"
            );
        }
    }
}
