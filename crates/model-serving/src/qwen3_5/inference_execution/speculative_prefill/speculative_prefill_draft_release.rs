use astronomical_runtime_integration::MlxArray;

use crate::{InferenceEngineError, PerformanceOperation, Qwen3_5Model, RequestDecoderStateStack};

use super::super::{
    Qwen3_5EngineState, engine_request::Qwen3_5EngineRequest, qwen3_5_runtime_error,
};
use super::speculative_prefill_failure::configured_speculative_prefill_failure;

impl Qwen3_5EngineState {
    /// Samples drafter memory while every logical owner is still live, then
    /// releases decoder state at a synchronized allocator-cleanup boundary.
    pub(in crate::qwen3_5) fn capture_speculative_prefill_draft_memory_and_release_decoder_state(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        draft_model: &Qwen3_5Model,
        draft_request_decoder_state: RequestDecoderStateStack,
        draft_visual_embeddings: Option<&MlxArray>,
    ) -> Result<(), InferenceEngineError> {
        active_request.speculative_prefill_draft_memory_telemetry = self
            .speculative_prefill_draft_memory_telemetry(
                active_request,
                draft_model,
                &draft_request_decoder_state,
                draft_visual_embeddings,
            );
        drop(draft_request_decoder_state);
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    draft_model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )
            .map_err(|draft_allocator_cleanup_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter cleanup",
                    draft_allocator_cleanup_error,
                )
            })
    }

    /// Drops every request-scoped draft owner, then restores complete target
    /// residency whenever the target and live request state fit together.
    pub(in crate::qwen3_5) fn release_speculative_prefill_draft_and_restore_target_residency(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        draft_visual_embeddings: Option<MlxArray>,
        draft_model: Qwen3_5Model,
    ) -> Result<(), InferenceEngineError> {
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::SpeculativePrefillRequestScopedDraftRelease,
                |_performance_attribution| {
                    drop(draft_visual_embeddings);
                    drop(draft_model);
                    let target_model = self.model.as_ref().ok_or_else(|| {
                        qwen3_5_runtime_error("Qwen3.5 engine lost its loaded target model")
                    })?;
                    target_model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                        .map_err(qwen3_5_runtime_error)
                },
            )
            .map_err(|draft_release_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter release",
                    draft_release_error,
                )
            })?;
        let resumed_target_expert_retention = self.model.as_ref().is_some_and(|target_model| {
            target_model.resume_expert_retention_after_request_memory_pressure()
        });
        let target_residency_promotion_outcome = self
            .model
            .as_mut()
            .ok_or_else(|| qwen3_5_runtime_error("Qwen3.5 engine lost its loaded target model"))?
            .try_promote_experts_to_resident(
                crate::qwen3_5_moe::Qwen3_5ExpertResidencyTransitionReason::SpeculativePrefillDraftLoading,
                &mut active_request.performance_attribution,
            )
            .map_err(|target_residency_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "target expert residency restoration after drafter release",
                    target_residency_error,
                )
            })?;
        let target_expert_payload_bytes_after_draft_release = self
            .model
            .as_ref()
            .ok_or_else(|| qwen3_5_runtime_error("Qwen3.5 engine lost its loaded target model"))?
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        active_request.speculative_prefill_target_expert_payload_bytes_after_draft_release =
            Some(target_expert_payload_bytes_after_draft_release);
        tracing::info!(
            request_id = active_request.request_id.value(),
            resumed_target_expert_retention,
            ?target_residency_promotion_outcome,
            target_expert_payload_bytes_after_draft_release,
            "released request-scoped speculative-prefill draft and restored fitting target residency"
        );
        Ok(())
    }
}
