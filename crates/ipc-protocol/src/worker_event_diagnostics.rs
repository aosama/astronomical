//! Payload-safe summaries for logging worker events without generated text or image bytes.

use crate::WorkerEvent;

impl WorkerEvent {
    /// Returns a bounded diagnostic summary without exposing model-generated payloads.
    #[must_use]
    pub fn diagnostic_summary(&self) -> String {
        match self {
            Self::RuntimeFeatureConfigurationApplied { .. } => {
                "runtime_feature_configuration_applied".to_owned()
            }
            Self::Idle { .. } => "idle".to_owned(),
            Self::MlxMemorySample { .. } => "mlx_memory_sample".to_owned(),
            Self::MlxMemoryLimitChanged { .. } => "mlx_memory_limit_changed".to_owned(),
            Self::MlxMemoryLimitRejected { .. } => "mlx_memory_limit_rejected".to_owned(),
            Self::MlxMemoryLimitRestartRequired { .. } => {
                "mlx_memory_limit_restart_required".to_owned()
            }
            Self::ExpertMemoryModeChanged { .. } => "expert_memory_mode_changed".to_owned(),
            Self::GenerationFinalized { request_id, .. } => {
                format!("generation_finalized request_id={}", request_id.value())
            }
            Self::ExpertStreamingStats { .. } | Self::PersistentPromptCacheStats { .. } => {
                observatory_event_diagnostic_summary(self)
            }
            Self::ImageGenerationProgress { request_id, .. } => format!(
                "image_generation_progress request_id={}",
                request_id.value()
            ),
            Self::ImageGenerationCompleted { request_id, .. } => format!(
                "image_generation_completed request_id={}",
                request_id.value()
            ),
            Self::ImageGenerationFailed { request_id, .. } => {
                format!("image_generation_failed request_id={}", request_id.value())
            }
            Self::ImageGenerationFinalized { request_id, .. } => format!(
                "image_generation_finalized request_id={}",
                request_id.value()
            ),
            Self::EmbeddingsCompleted { request_id, .. } => {
                format!("embeddings_completed request_id={}", request_id.value())
            }
            Self::EmbeddingsFailed { request_id, .. } => {
                format!("embeddings_failed request_id={}", request_id.value())
            }
            Self::EmbeddingsFinalized { request_id, .. } => {
                format!("embeddings_finalized request_id={}", request_id.value())
            }
            Self::Ready { .. } => "ready".to_owned(),
            Self::Output { request_id, .. } => {
                format!("output request_id={}", request_id.value())
            }
            Self::PrefillProgress { request_id, .. } => {
                format!("prefill_progress request_id={}", request_id.value())
            }
            Self::GenerationPreparationStarted { request_id, .. } => format!(
                "generation_preparation_started request_id={}",
                request_id.value()
            ),
            Self::GenerationProgress { request_id, .. } => {
                format!("generation_progress request_id={}", request_id.value())
            }
            Self::FirstDecodeCompleted { request_id, .. } => {
                format!("first_decode_completed request_id={}", request_id.value())
            }
            Self::PromptWorkReuse { request_id, .. } => {
                format!("prompt_work_reuse request_id={}", request_id.value())
            }
            Self::Completed { request_id, .. } => {
                format!("completed request_id={}", request_id.value())
            }
            Self::Failed { request_id, .. } => {
                format!("failed request_id={}", request_id.value())
            }
            Self::ModelSwapped { .. } => "model_swapped".to_owned(),
            Self::ModelSwapFailed { .. } => "model_swap_failed".to_owned(),
            Self::PromptCacheCleared { model_id, .. } => {
                format!("prompt_cache_cleared model_id={model_id:?}")
            }
        }
    }
}

/// Observatory events share one arm of [`WorkerEvent::diagnostic_summary`].
/// The caller routes only these two variants; the final arm keeps this helper
/// exhaustive without printing generated text or image bytes.
fn observatory_event_diagnostic_summary(event: &WorkerEvent) -> String {
    match event {
        WorkerEvent::ExpertStreamingStats { stats } => format!(
            "expert_streaming_stats model_id={} retained_assignments={}",
            stats.model_id.as_deref().unwrap_or("<none>"),
            stats.retained_route_assignment_count
        ),
        WorkerEvent::PersistentPromptCacheStats { .. } => {
            "persistent_prompt_cache_stats".to_owned()
        }
        _ => "worker_event".to_owned(),
    }
}
