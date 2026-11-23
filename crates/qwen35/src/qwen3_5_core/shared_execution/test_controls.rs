//! Test controls both Qwen3.5 engines offer, written once for any engine
//! that owns a [`Qwen3_5ExecutionCore`]. Each runs on the MLX owner thread,
//! where the engine's state lives.

use ipc_protocol::{RequestId, WorkerPromptWorkReuse};

use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};
use crate::{InferenceEngineError, MlxInferenceEngine, MlxInferenceExecution};

use super::execution_core::Qwen3_5ExecutionCoreOwner;

impl<Execution> MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution + Qwen3_5ExecutionCoreOwner,
{
    /// How much prompt work the active request reused, for the request named.
    #[doc(hidden)]
    pub async fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<WorkerPromptWorkReuse, InferenceEngineError> {
        let (prompt_work_reuse_sender, prompt_work_reuse_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |inference_execution| {
            let prompt_work_reuse =
                active_request_prompt_work_reuse(inference_execution, request_id)?;
            prompt_work_reuse_sender
                .send(prompt_work_reuse)
                .map_err(|_| fatal_engine_error("prompt work reuse receiver stopped unexpectedly"))
        })
        .await?;
        prompt_work_reuse_receiver.recv().map_err(|_| {
            fatal_engine_error("prompt work reuse owner operation returned no response")
        })
    }

    /// Resets the process-global MLX peak counter on the engine owner thread.
    #[doc(hidden)]
    pub async fn reset_mlx_peak_memory_for_tests(&self) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|inference_execution| {
            inference_execution
                .execution_core()
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

fn active_request_prompt_work_reuse<Execution: Qwen3_5ExecutionCoreOwner>(
    inference_execution: &Execution,
    request_id: RequestId,
) -> Result<WorkerPromptWorkReuse, InferenceEngineError> {
    let active_request = inference_execution
        .execution_core()
        .active_request
        .as_ref()
        .ok_or_else(|| {
            fatal_engine_error("cannot inspect prompt work reuse without an active request")
        })?;
    if active_request.request_id != request_id {
        return Err(fatal_engine_error(
            "cannot inspect prompt work reuse for a different request",
        ));
    }
    Ok(active_request.prompt_work_reuse)
}
