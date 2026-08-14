use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use astronomical_ipc_protocol::{RequestId, WorkerEvent};
use tokio::sync::oneshot;

use crate::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};

use super::{
    EngineGenerationStart, EngineLoadResult, GeneratedToken, GenerationFinalization,
    InferenceEngine, InferenceEngineError, MlxInferenceExecution,
};

const MLX_OWNER_COMMAND_CAPACITY: usize = 1;

/// Generic asynchronous facade for one architecture-specific MLX owner thread.
pub struct MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution,
{
    command_sender: SyncSender<MlxInferenceCommand<Execution>>,
    owner_thread: Option<JoinHandle<()>>,
}

impl<Execution> MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution,
{
    /// Starts an MLX owner thread and constructs the architecture implementation on that thread.
    pub fn new(
        create_inference_execution: impl FnOnce() -> Execution + Send + 'static,
    ) -> Result<Self, InferenceEngineError> {
        let (command_sender, command_receiver) = sync_channel(MLX_OWNER_COMMAND_CAPACITY);
        let owner_thread = thread::Builder::new()
            .name("astronomical-mlx-inference-owner".to_owned())
            .spawn(move || run_mlx_owner(create_inference_execution(), command_receiver))
            .map_err(|source| InferenceEngineError::Fatal {
                reason: format!("failed to start MLX inference owner thread: {source}"),
            })?;
        Ok(Self {
            command_sender,
            owner_thread: Some(owner_thread),
        })
    }

    fn send_command(
        &self,
        inference_command: MlxInferenceCommand<Execution>,
    ) -> Result<(), InferenceEngineError> {
        self.command_sender
            .send(inference_command)
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "MLX inference owner thread stopped unexpectedly".to_owned(),
            })
    }

    #[cfg(feature = "direct-mlx")]
    pub(crate) async fn run_owner_test_operation(
        &self,
        owner_test_operation: impl FnOnce(&mut Execution) -> Result<(), InferenceEngineError>
        + Send
        + 'static,
    ) -> Result<(), InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::RunOwnerTestOperation {
            owner_test_operation: Box::new(owner_test_operation),
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }
}

impl<Execution> InferenceEngine for MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution,
{
    type Request = Execution::Request;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::Load(completion_sender))?;
        receive_owner_result(completion_receiver).await
    }

    async fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::StartGeneration {
            inference_request,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }

    async fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::DecodeNextToken {
            request_id,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }

    async fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::InjectInputTokens {
            request_id,
            input_token_ids,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }

    async fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::CancelGeneration {
            request_id,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }

    async fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::CollectPersistentPromptCacheStats(
            completion_sender,
        ))?;
        receive_owner_result(completion_receiver).await
    }

    async fn clear_persistent_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::ClearPersistentPromptCache {
            model_id,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }

    async fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::CollectMlxMemoryTelemetry(
            completion_sender,
        ))?;
        receive_owner_result(completion_receiver).await
    }

    async fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        let (completion_sender, completion_receiver) = oneshot::channel();
        self.send_command(MlxInferenceCommand::UpdateMlxMemoryLimit {
            requested_mlx_memory_ceiling_bytes,
            completion_sender,
        })?;
        receive_owner_result(completion_receiver).await
    }
}

impl<Execution> Drop for MlxInferenceEngine<Execution>
where
    Execution: MlxInferenceExecution,
{
    fn drop(&mut self) {
        let _owner_was_already_stopped = self
            .command_sender
            .send(MlxInferenceCommand::Shutdown)
            .is_err();
        if let Some(owner_thread) = self.owner_thread.take() {
            let _owner_thread_panicked = owner_thread.join().is_err();
        }
    }
}

enum MlxInferenceCommand<Execution>
where
    Execution: MlxInferenceExecution,
{
    Load(OwnerResultSender<EngineLoadResult>),
    StartGeneration {
        inference_request: Execution::Request,
        completion_sender: OwnerResultSender<EngineGenerationStart>,
    },
    DecodeNextToken {
        request_id: RequestId,
        completion_sender: OwnerResultSender<GeneratedToken>,
    },
    InjectInputTokens {
        request_id: RequestId,
        input_token_ids: Vec<u32>,
        completion_sender: OwnerResultSender<()>,
    },
    CancelGeneration {
        request_id: RequestId,
        completion_sender: OwnerResultSender<GenerationFinalization>,
    },
    CollectPersistentPromptCacheStats(OwnerResultSender<Option<WorkerEvent>>),
    ClearPersistentPromptCache {
        model_id: Option<String>,
        completion_sender: OwnerResultSender<Option<WorkerEvent>>,
    },
    CollectMlxMemoryTelemetry(OwnerResultSender<Option<MlxMemoryTelemetry>>),
    UpdateMlxMemoryLimit {
        requested_mlx_memory_ceiling_bytes: u64,
        completion_sender: OwnerResultSender<MlxMemoryLimitAdjustment>,
    },
    #[cfg(feature = "direct-mlx")]
    RunOwnerTestOperation {
        owner_test_operation:
            Box<dyn FnOnce(&mut Execution) -> Result<(), InferenceEngineError> + Send>,
        completion_sender: OwnerResultSender<()>,
    },
    Shutdown,
}

type OwnerResultSender<Output> = oneshot::Sender<Result<Output, InferenceEngineError>>;

fn run_mlx_owner<Execution>(
    mut execution: Execution,
    command_receiver: Receiver<MlxInferenceCommand<Execution>>,
) where
    Execution: MlxInferenceExecution,
{
    while let Ok(inference_command) = command_receiver.recv() {
        match inference_command {
            MlxInferenceCommand::Load(completion_sender) => {
                let _receiver_was_dropped = completion_sender.send(execution.load()).is_err();
            }
            MlxInferenceCommand::StartGeneration {
                inference_request,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.start_generation(inference_request))
                    .is_err();
            }
            MlxInferenceCommand::DecodeNextToken {
                request_id,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.decode_next_token(request_id))
                    .is_err();
            }
            MlxInferenceCommand::InjectInputTokens {
                request_id,
                input_token_ids,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.inject_input_tokens(request_id, input_token_ids))
                    .is_err();
            }
            MlxInferenceCommand::CancelGeneration {
                request_id,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.cancel_generation(request_id))
                    .is_err();
            }
            MlxInferenceCommand::CollectPersistentPromptCacheStats(completion_sender) => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.collect_persistent_prompt_cache_stats())
                    .is_err();
            }
            MlxInferenceCommand::ClearPersistentPromptCache {
                model_id,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.clear_persistent_prompt_cache(model_id))
                    .is_err();
            }
            MlxInferenceCommand::CollectMlxMemoryTelemetry(completion_sender) => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.collect_mlx_memory_telemetry())
                    .is_err();
            }
            MlxInferenceCommand::UpdateMlxMemoryLimit {
                requested_mlx_memory_ceiling_bytes,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(execution.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes))
                    .is_err();
            }
            #[cfg(feature = "direct-mlx")]
            MlxInferenceCommand::RunOwnerTestOperation {
                owner_test_operation,
                completion_sender,
            } => {
                let _receiver_was_dropped = completion_sender
                    .send(owner_test_operation(&mut execution))
                    .is_err();
            }
            MlxInferenceCommand::Shutdown => return,
        }
    }
}

async fn receive_owner_result<Output>(
    completion_receiver: oneshot::Receiver<Result<Output, InferenceEngineError>>,
) -> Result<Output, InferenceEngineError> {
    completion_receiver
        .await
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "MLX inference owner stopped before reporting command completion".to_owned(),
        })?
}
