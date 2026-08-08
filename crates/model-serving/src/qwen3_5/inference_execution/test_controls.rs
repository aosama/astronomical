use crate::InferenceEngineError;
use astronomical_ipc_protocol::RequestId;

use crate::MlxInferenceEngine;

use super::{
    Qwen3_5InferenceExecution, Qwen3_5SpeculativePrefillFailureStageForTests,
    fatal_engine_error, qwen3_5_runtime_error,
};

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Returns truthful per-model prompt work for an active qualification request.
    #[doc(hidden)]
    pub async fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<astronomical_ipc_protocol::WorkerPromptWorkReuse, InferenceEngineError> {
        let (prompt_work_reuse_sender, prompt_work_reuse_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let prompt_work_reuse =
                qwen_inference_execution.prompt_work_reuse_for_tests(request_id)?;
            prompt_work_reuse_sender
                .send(prompt_work_reuse)
                .map_err(|_| fatal_engine_error("prompt work reuse receiver stopped unexpectedly"))
        })
        .await?;
        prompt_work_reuse_receiver.recv().map_err(|_| {
            fatal_engine_error("prompt work reuse owner operation returned no response")
        })
    }

    /// Returns the target conversation positions selected for an active qualification request.
    #[doc(hidden)]
    pub async fn speculative_prefill_selected_token_positions_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<Option<Vec<usize>>, InferenceEngineError> {
        let (selected_token_positions_sender, selected_token_positions_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let selected_token_positions = qwen_inference_execution
                .speculative_prefill_selected_token_positions_for_tests(request_id)?;
            selected_token_positions_sender
                .send(selected_token_positions)
                .map_err(|_| {
                    fatal_engine_error(
                        "selected speculative-prefill positions receiver stopped unexpectedly",
                    )
                })
        })
        .await?;
        selected_token_positions_receiver.recv().map_err(|_| {
            fatal_engine_error(
                "selected speculative-prefill positions owner operation returned no response",
            )
        })
    }

    /// Leaves prompt-cache reads available while removing the optional writer.
    pub async fn disable_persistent_prompt_cache_writes_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution
                .persistent_prompt_cache_write_queue
                .take();
            Ok(())
        })
        .await
    }

    /// Rejects one completed prefill attempt so qualification can verify full retry rollback.
    pub async fn force_next_prefill_capacity_rejection_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.force_next_prefill_capacity_rejection_for_tests(request_id)
        })
        .await
    }

    /// Forces one draft-prefix restore failure so qualification can verify uncached retry.
    pub async fn force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution
                .force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(request_id)
        })
        .await
    }

    /// Forces one configured SpecPrefill stage to fail through its production error boundary.
    #[doc(hidden)]
    pub async fn force_next_speculative_prefill_failure_for_tests(
        &self,
        request_id: RequestId,
        failure_stage: Qwen3_5SpeculativePrefillFailureStageForTests,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution
                .force_next_speculative_prefill_failure_for_tests(request_id, failure_stage)
        })
        .await
    }

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
                .map_err(qwen3_5_runtime_error)
        })
        .await
    }
}
