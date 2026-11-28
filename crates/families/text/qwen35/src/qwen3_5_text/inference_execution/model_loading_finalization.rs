use crate::inference_engine::load_result::EngineLoadResult;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    pub(super) fn engine_load_result(
        &self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> EngineLoadResult {
        // Build readiness only after startup has selected the owner.
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
            .with_draft_model_status(self.draft_model_status())
            .with_custom_metal_kernel_report(self.custom_metal_kernel_report())
            .with_prompt_processing_chunk_report(self.prompt_processing_chunk_report)
    }

    /// What this worker's capability probes decided and which custom kernels
    /// the loaded model holds. Before any probe ran (no model has loaded the
    /// kernels yet) the verdicts are empty rather than invented.
    fn custom_metal_kernel_report(&self) -> ipc_protocol::CustomMetalKernelReport {
        let probe_verdicts =
            engine_core::kernel_capability_registry::retained_worker_kernel_capabilities()
                .map(engine_core::kernel_capability_report::probe_verdict_report)
                .unwrap_or_default();
        let in_use = self
            .core
            .model
            .as_ref()
            .map(|loaded_model| loaded_model.custom_metal_kernels_in_use())
            .unwrap_or_default();
        ipc_protocol::CustomMetalKernelReport {
            probe_verdicts,
            in_use,
        }
    }

    /// What loading found about a drafter: drafting with the one it hosts,
    /// or the reason it hosts none. No model loaded, no drafter.
    fn draft_model_status(&self) -> ipc_protocol::DraftModelStatus {
        self.core
            .model
            .as_ref()
            .and_then(|loaded_model| {
                loaded_model
                    .base
                    .optional_accelerators
                    .draft_model_status
                    .clone()
            })
            .unwrap_or(ipc_protocol::DraftModelStatus::None)
    }
}
