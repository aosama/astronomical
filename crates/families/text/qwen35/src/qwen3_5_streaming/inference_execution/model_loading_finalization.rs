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
            .with_startup_residency_facts(self.startup_residency_facts)
            .with_neural_engine_prefill_status(
                EngineLoadResult::neural_engine_prefill_unsupported_when_enabled(
                    self.core
                        .chunking
                        .experimental_neural_engine_prefill_enabled,
                ),
            )
            .with_draft_model_status(self.draft_model_status())
    }

    /// This engine never drafts; status names a drafter it would have used.
    fn draft_model_status(&self) -> ipc_protocol::DraftModelStatus {
        match self.core.model.as_ref() {
            Some(loaded_model) => {
                engine_core::speculative_decoding::unused_drafter_status::streaming_load_status(
                    &self.core.model_directory,
                    &engine_core::speculative_decoding::drafter_discovery::DrafterPairing {
                        target_layer_count: loaded_model.config.layer_count() as i32,
                        target_hidden_size: loaded_model.config.hidden_size() as i32,
                        target_vocabulary_size: loaded_model.config.vocabulary_size() as i32,
                    },
                )
            }
            None => ipc_protocol::DraftModelStatus::None,
        }
    }
}
