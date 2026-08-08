use crate::MlxActiveMemoryBreakdown;

use super::{Qwen3_5Model, Qwen3_5VisionModel, RequestDecoderStateStack};

impl Qwen3_5Model {
    #[must_use]
    pub(crate) fn active_memory_breakdown(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        additional_context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let context_state_payload_bytes = request_decoder_state
            .payload_byte_count()
            .saturating_add(additional_context_state_payload_bytes);
        self.active_memory_breakdown_with_context_state_payload_bytes(
            context_state_payload_bytes,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    #[must_use]
    pub(crate) fn finalized_active_memory_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        self.active_memory_breakdown_with_context_state_payload_bytes(
            0,
            mlx_active_memory_bytes,
            additional_model_core_payload_bytes,
        )
    }

    fn active_memory_breakdown_with_context_state_payload_bytes(
        &self,
        context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
        additional_model_core_payload_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let paged_expert_payload_bytes = self
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let expert_payload_bytes = paged_expert_payload_bytes;
        let model_core_payload_bytes = self
            .resident_model_payload_byte_count()
            .saturating_add(
                self.vision_model
                    .as_ref()
                    .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
            )
            .saturating_add(additional_model_core_payload_bytes);
        MlxActiveMemoryBreakdown::reconcile(
            mlx_active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        )
    }
}
