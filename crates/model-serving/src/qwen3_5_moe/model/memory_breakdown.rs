use crate::MlxActiveMemoryBreakdown;
use crate::qwen3_5_moe::Qwen3_5MoEMtpRequestState;

use super::{Qwen3_5MoEModel, Qwen3_5MoEVisionModel, RequestDecoderStateStack};

impl Qwen3_5MoEModel {
    #[must_use]
    pub(in crate::qwen3_5_moe) fn active_memory_breakdown(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        mtp_request_state: Option<&Qwen3_5MoEMtpRequestState>,
        mlx_active_memory_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let context_state_payload_bytes =
            request_decoder_state.payload_byte_count().saturating_add(
                mtp_request_state.map_or(0, Qwen3_5MoEMtpRequestState::payload_byte_count),
            );
        self.active_memory_breakdown_with_context_state_payload_bytes(
            context_state_payload_bytes,
            mlx_active_memory_bytes,
        )
    }

    #[must_use]
    pub(in crate::qwen3_5_moe) fn finalized_active_memory_breakdown(
        &self,
        mlx_active_memory_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        self.active_memory_breakdown_with_context_state_payload_bytes(0, mlx_active_memory_bytes)
    }

    fn active_memory_breakdown_with_context_state_payload_bytes(
        &self,
        context_state_payload_bytes: u64,
        mlx_active_memory_bytes: u64,
    ) -> MlxActiveMemoryBreakdown {
        let paged_expert_payload_bytes = self
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let expert_payload_bytes = paged_expert_payload_bytes;
        let model_core_payload_bytes = self.resident_model_payload_byte_count().saturating_add(
            self.vision_model
                .as_ref()
                .map_or(0, Qwen3_5MoEVisionModel::resident_payload_bytes),
        );
        MlxActiveMemoryBreakdown::reconcile(
            mlx_active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        )
    }
}
