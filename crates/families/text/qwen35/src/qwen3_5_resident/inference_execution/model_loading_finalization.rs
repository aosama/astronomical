use crate::inference_engine::load_result::EngineLoadResult;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    pub(super) fn engine_load_result(
        &self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> EngineLoadResult {
        // Build readiness only after startup promotion has selected the owner.
        // The worker forwards this value instead of inferring mode from memory.
        EngineLoadResult::new()
            .with_expert_memory_mode(
                self.core
                    .model
                    .as_ref()
                    .map(|loaded_model| loaded_model.expert_memory_mode()),
            )
            .with_minimum_mlx_memory_ceiling_bytes(minimum_mlx_memory_ceiling_bytes)
            .with_neural_engine_prefill_status(self.neural_engine_prefill.status().clone())
            .with_draft_model_status(self.draft_model_status.clone())
    }
}
