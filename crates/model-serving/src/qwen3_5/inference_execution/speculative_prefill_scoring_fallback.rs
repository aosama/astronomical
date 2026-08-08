#[cfg(feature = "direct-mlx")]
use crate::PerformanceCounter;

#[cfg(feature = "direct-mlx")]
use super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn record_speculative_prefill_scoring_fallback(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        draft_model: &crate::Qwen3_5Model,
        speculative_prefill_error: impl std::fmt::Display,
    ) {
        active_request.should_use_speculative_prefill = false;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::SpeculativePrefillFallbackCount, 1);
        tracing::warn!(
            request_id = active_request.request_id.value(),
            error = %speculative_prefill_error,
            "optional speculative-prefill scoring failed; continuing target-only"
        );
        if let Err(allocator_cleanup_error) = draft_model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
        {
            tracing::debug!(
                error = %allocator_cleanup_error,
                "speculative-prefill draft allocator cleanup failed after fallback"
            );
        }
    }
}
