//! Model-load and request attribution report finalization for the folded-ternary engine.

use astronomical_ipc_protocol::RequestId;

use crate::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionOutcome,
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
            .model
            .as_ref()
            .and_then(|model| model.runtime.memory_snapshot().ok());
        let Some(report) = performance_attribution.finish_model_loading(
            ModelLoadingPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id: self.attribution_model_id.clone(),
                model_revision: self.attribution_model_revision.clone(),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes: self
                    .model
                    .as_ref()
                    .map(|model| model.pack().total_payload_bytes()),
                resident_model_payload_bytes: self
                    .model
                    .as_ref()
                    .map(|model| model.pack().total_payload_bytes()),
                model_shard_count: Some(1),
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: None,
            },
        ) else {
            return;
        };
        if let Err(write_error) = self.performance_attribution_log.record(&report) {
            tracing::warn!(
                error = %write_error,
                "folded-ternary model-loading attribution could not be recorded"
            );
        }
    }

    pub(super) fn take_active_and_record_generation_attribution(&mut self) {
        let Some(mut active_generation) = self.active.take() else {
            return;
        };
        let request_id = active_generation.request_id;
        let configured_maximum_output_tokens = active_generation.request.max_output_tokens();
        let performance_attribution = std::mem::replace(
            &mut active_generation.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        self.record_generation_performance_attribution(
            performance_attribution,
            request_id,
            configured_maximum_output_tokens,
        );
    }

    fn record_generation_performance_attribution(
        &mut self,
        performance_attribution: PerformanceAttribution,
        request_id: RequestId,
        configured_maximum_output_tokens: u16,
    ) {
        if !performance_attribution.is_enabled() {
            return;
        }
        let (Some(model_id), Some(model_revision)) = (
            self.attribution_model_id.clone(),
            self.attribution_model_revision.clone(),
        ) else {
            tracing::warn!(
                request_id = request_id.value(),
                "folded-ternary generation attribution lacked loaded-model identity"
            );
            return;
        };
        let mlx_memory_snapshot = self
            .model
            .as_ref()
            .and_then(|model| model.runtime.memory_snapshot().ok());
        let Some(report) =
            performance_attribution.finish_generation(GenerationPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id,
                model_revision,
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                request_id: request_id.value(),
                configured_maximum_output_tokens,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: None,
            })
        else {
            return;
        };
        if let Err(write_error) = self.performance_attribution_log.record(&report) {
            tracing::warn!(
                request_id = request_id.value(),
                error = %write_error,
                "folded-ternary generation attribution could not be recorded"
            );
        }
    }
}
