//! Test controls every text engine offer, written once for any engine
//! that owns a [`TextExecutionCore`]. Each runs on the MLX owner thread,
//! where the engine's state lives.

use crate::model::TextServingModel;
use ipc_protocol::{ExpertMemoryMode, RequestId, WorkerPromptWorkReuse};

use crate::engine_errors::{fatal_engine_error, fatal_runtime_error};
use crate::{InferenceEngineError, MlxInferenceEngine, MlxInferenceExecution};

use super::execution_core::TextExecutionCoreOwner;
use crate::model::text_model::TextModel;

/// Test controls of the `MlxInferenceEngine` this family runs on. A trait,
/// because only the engine core may add inherent methods to that type.
// Every caller awaits these in place, so the futures' Send bound never matters.
#[allow(async_fn_in_trait)]
pub trait TextMlxEngineTestControls {
    /// How much prompt work the active request reused, for the request named.
    async fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<WorkerPromptWorkReuse, InferenceEngineError>;

    /// Resets the process-global MLX peak counter on the engine owner thread.
    async fn reset_mlx_peak_memory_for_tests(&self) -> Result<(), InferenceEngineError>;
}

impl<Execution> TextMlxEngineTestControls for MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution + TextExecutionCoreOwner,
{
    async fn prompt_work_reuse_for_tests(
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

    async fn reset_mlx_peak_memory_for_tests(&self) -> Result<(), InferenceEngineError> {
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
                .map_err(fatal_runtime_error)
        })
        .await
    }
}

/// Controls that read the loaded text model.
/// Test controls of the `MlxInferenceEngine` this family runs on. A trait,
/// because only the engine core may add inherent methods to that type.
// Every caller awaits these in place, so the futures' Send bound never matters.
#[allow(async_fn_in_trait)]
pub trait TextModelTestControls {
    /// Returns the currently loaded model's truthful expert-memory mode.
    async fn expert_memory_mode_for_tests(
        &self,
    ) -> Result<Option<ExpertMemoryMode>, InferenceEngineError>;
}

impl<Execution> TextModelTestControls for MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution + TextExecutionCoreOwner<Model: TextModel>,
{
    async fn expert_memory_mode_for_tests(
        &self,
    ) -> Result<Option<ExpertMemoryMode>, InferenceEngineError> {
        let (expert_memory_mode_sender, expert_memory_mode_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            expert_memory_mode_sender
                .send(loaded_expert_memory_mode(qwen_inference_execution))
                .map_err(|_| fatal_engine_error("expert-memory mode receiver stopped unexpectedly"))
        })
        .await?;
        expert_memory_mode_receiver.recv().map_err(|_| {
            fatal_engine_error("expert-memory mode owner operation returned no response")
        })
    }
}

fn loaded_expert_memory_mode<Execution>(inference_execution: &Execution) -> Option<ExpertMemoryMode>
where
    Execution: TextExecutionCoreOwner<Model: TextModel>,
{
    inference_execution
        .execution_core()
        .model
        .as_ref()
        .map(TextModel::expert_memory_mode)
}

fn active_request_prompt_work_reuse<Execution: TextExecutionCoreOwner>(
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
