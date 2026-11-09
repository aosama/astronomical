//! Payload-safe summaries for logging worker events without generated text or image bytes.

use crate::{RequestId, WorkerEvent};

impl WorkerEvent {
    /// Returns a bounded diagnostic summary without exposing model-generated payloads.
    ///
    /// The summary is the event's name followed by its request id, or by the
    /// model-scoped counters it carries; never by generated payload.
    #[must_use]
    pub fn diagnostic_summary(&self) -> String {
        let event_name = self.diagnostic_event_name();
        if let Some(request_id) = self.diagnostic_request_id() {
            return format!("{event_name} request_id={}", request_id.value());
        }
        match self {
            Self::ExpertStreamingStats { stats } => expert_streaming_summary(
                event_name,
                stats.model_id.as_deref(),
                stats.retained_route_assignment_count,
            ),
            Self::PromptCacheCleared { model_id, .. } => {
                format!("{event_name} model_id={model_id:?}")
            }
            _ => event_name.to_owned(),
        }
    }

    /// The event's snake-case name. Exhaustive on purpose: a new variant
    /// must be named here rather than fall into another event's label.
    fn diagnostic_event_name(&self) -> &'static str {
        match self {
            Self::RuntimeFeatureConfigurationApplied { .. } => {
                "runtime_feature_configuration_applied"
            }
            Self::Idle { .. } => "idle",
            Self::MlxMemorySample { .. } => "mlx_memory_sample",
            Self::MlxMemoryLimitChanged { .. } => "mlx_memory_limit_changed",
            Self::MlxMemoryLimitRejected { .. } => "mlx_memory_limit_rejected",
            Self::MlxMemoryLimitRestartRequired { .. } => "mlx_memory_limit_restart_required",
            Self::ExpertMemoryModeChanged { .. } => "expert_memory_mode_changed",
            Self::NeuralEnginePrefillChanged { .. } => "neural_engine_prefill_changed",
            Self::NeuralEngineChunkFinished { .. } => "neural_engine_chunk_finished",
            Self::ModelLoadInFlight { .. } => "model_load_in_flight",
            Self::GenerationFinalized { .. } => "generation_finalized",
            Self::ExpertStreamingStats { .. } => "expert_streaming_stats",
            Self::PersistentPromptCacheStats { .. } => "persistent_prompt_cache_stats",
            Self::ImageGenerationProgress { .. } => "image_generation_progress",
            Self::ImageGenerationCompleted { .. } => "image_generation_completed",
            Self::ImageGenerationFailed { .. } => "image_generation_failed",
            Self::ImageGenerationFinalized { .. } => "image_generation_finalized",
            Self::EmbeddingsCompleted { .. } => "embeddings_completed",
            Self::EmbeddingsFailed { .. } => "embeddings_failed",
            Self::EmbeddingsFinalized { .. } => "embeddings_finalized",
            Self::Ready { .. } => "ready",
            Self::Output { .. } => "output",
            Self::PrefillProgress { .. } => "prefill_progress",
            Self::GenerationPreparationStarted { .. } => "generation_preparation_started",
            Self::GenerationProgress { .. } => "generation_progress",
            Self::FirstDecodeCompleted { .. } => "first_decode_completed",
            Self::PromptWorkReuse { .. } => "prompt_work_reuse",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
            Self::ModelSwapped { .. } => "model_swapped",
            Self::ModelSwapFailed { .. } => "model_swap_failed",
            Self::PromptCacheCleared { .. } => "prompt_cache_cleared",
        }
    }

    /// The request a request-scoped event belongs to. Exhaustive for the
    /// same reason as the name: a new request event must decide here.
    fn diagnostic_request_id(&self) -> Option<RequestId> {
        match self {
            Self::GenerationFinalized { request_id, .. }
            | Self::ImageGenerationProgress { request_id, .. }
            | Self::ImageGenerationCompleted { request_id, .. }
            | Self::ImageGenerationFailed { request_id, .. }
            | Self::ImageGenerationFinalized { request_id, .. }
            | Self::EmbeddingsCompleted { request_id, .. }
            | Self::EmbeddingsFailed { request_id, .. }
            | Self::EmbeddingsFinalized { request_id, .. }
            | Self::Output { request_id, .. }
            | Self::PrefillProgress { request_id, .. }
            | Self::GenerationPreparationStarted { request_id, .. }
            | Self::GenerationProgress { request_id, .. }
            | Self::FirstDecodeCompleted { request_id, .. }
            | Self::PromptWorkReuse { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Failed { request_id, .. } => Some(*request_id),
            Self::RuntimeFeatureConfigurationApplied { .. }
            | Self::Idle { .. }
            | Self::MlxMemorySample { .. }
            | Self::MlxMemoryLimitChanged { .. }
            | Self::MlxMemoryLimitRejected { .. }
            | Self::MlxMemoryLimitRestartRequired { .. }
            | Self::ExpertMemoryModeChanged { .. }
            | Self::NeuralEnginePrefillChanged { .. }
            | Self::NeuralEngineChunkFinished { .. }
            | Self::ModelLoadInFlight { .. }
            | Self::ExpertStreamingStats { .. }
            | Self::PersistentPromptCacheStats { .. }
            | Self::Ready { .. }
            | Self::ModelSwapped { .. }
            | Self::ModelSwapFailed { .. }
            | Self::PromptCacheCleared { .. } => None,
        }
    }
}

/// The streaming-stats line names its model, or `<none>` before one is
/// loaded, and the assignments served from retained memory so far.
fn expert_streaming_summary(
    event_name: &str,
    model_id: Option<&str>,
    retained_route_assignment_count: u64,
) -> String {
    format!(
        "{event_name} model_id={} retained_assignments={retained_route_assignment_count}",
        model_id.unwrap_or("<none>")
    )
}

#[cfg(test)]
#[path = "worker_event_diagnostics_tests.rs"]
mod worker_event_diagnostics_tests;
