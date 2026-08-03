use crate::InferenceEngineError;
use astronomical_ipc_protocol::RequestId;

use crate::MlxInferenceEngine;

use super::{Qwen3_5MoEInferenceExecution, fatal_engine_error, qwen3_5_moe_runtime_error};

impl MlxInferenceEngine<Qwen3_5MoEInferenceExecution> {
    /// Arms one deterministic draft rejection through the production acceptance path.
    pub async fn force_next_mtp_draft_rejection_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.force_next_mtp_draft_rejection_for_tests(request_id)
        })
        .await
    }

    /// Disables adaptive growth guarding and its memory sampling for benchmarks.
    #[doc(hidden)]
    pub async fn disable_adaptive_ram_growth_memory_guard_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution.adaptive_ram_growth_guard_enabled = false;
            Ok(())
        })
        .await
    }

    /// Resets the process-global MLX peak counter on the engine owner thread.
    #[doc(hidden)]
    pub async fn reset_mlx_peak_memory_for_tests(&self) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution
                .model
                .as_ref()
                .ok_or_else(|| {
                    fatal_engine_error("cannot reset MLX peak memory before the model is loaded")
                })?
                .runtime()
                .reset_peak_memory()
                .map_err(qwen3_5_moe_runtime_error)
        })
        .await
    }
}
