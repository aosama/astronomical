//! Laguna model-load and generation attribution report finalization.

use astronomical_ipc_protocol::RequestId;

use crate::{
    GenerationPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionOutcome,
};

use super::execution::LagunaInferenceExecution;

impl LagunaInferenceExecution {
    /// Finalizes one request report after request-owned arrays have been released.
    pub(super) fn record_generation_performance_attribution(
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
                "Laguna generation attribution lacked loaded-model identity"
            );
            return;
        };
        let memory_snapshot = self
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.memory_snapshot().ok());
        let Some(report) =
            performance_attribution.finish_generation(GenerationPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id,
                model_revision,
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                request_id: request_id.value(),
                configured_maximum_output_tokens,
                mlx_active_memory_bytes: memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: memory_snapshot.as_ref().and_then(|snapshot| {
                    u64::try_from(snapshot.allocator_cache_memory_bytes()).ok()
                }),
                mlx_peak_memory_bytes: memory_snapshot
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
                "Laguna generation attribution could not be recorded"
            );
        }
    }
}
