//! The streaming engine's half of the shared prompt-cache capture.

use super::engine_request::{Qwen3_5EngineRequest, StreamingRequestExtension};
use super::{Qwen3_5EngineState, Qwen3_5Model};
use crate::qwen3_5_core::prompt_cache_capture::Qwen3_5PromptCacheCapture;
use crate::qwen3_5_streaming;
use crate::{InferenceEngineError, PerformanceOperation};

impl Qwen3_5PromptCacheCapture for Qwen3_5EngineState {
    type Model = Qwen3_5Model;
    type RequestExtension = StreamingRequestExtension;

    /// Evicts retained expert pages to cover the deficit, measured on its own
    /// so performance reports can attribute why publication paused, and adds
    /// the expert bytes freed to the request's diagnostics.
    fn reclaim_before_publication_retry(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError> {
        let expert_payload_bytes_before_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        active_request.performance_attribution.measure_operation(
            PerformanceOperation::ExpertRetentionReclamation,
            |_performance_attribution| {
                qwen3_5_streaming::reclaim_retained_payload_for_request_memory_pressure(
                    model,
                    active_memory_deficit_bytes,
                )
            },
        )?;
        let expert_payload_bytes_after_reclamation = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        if let Some(persistent_prompt_cache_diagnostics) =
            active_request.persistent_prompt_cache_diagnostics.as_mut()
        {
            persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_publication =
                persistent_prompt_cache_diagnostics
                    .expert_bytes_reclaimed_for_publication
                    .saturating_add(
                        expert_payload_bytes_before_reclamation
                            .saturating_sub(expert_payload_bytes_after_reclamation),
                    );
        }
        Ok(())
    }
}
