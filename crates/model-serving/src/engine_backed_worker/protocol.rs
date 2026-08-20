//! Protocol startup and command-loop orchestration for the engine-backed worker.

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ImageGenerationFailureReason,
    MlxMemorySnapshotSource, MtpRuntimeState, ProtocolReader, ProtocolWriter,
    SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent, WorkerModelCapabilities,
    WorkerRuntimeFeatureConfiguration,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::support::{ActiveWorkerRequest, ModelFactory, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime};
use crate::{ImageGenerationEngine, InferenceEngine, ModelGenerationProcessor};

impl<Processor, Engine, Factory, ImageEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
{
    /// Loads the engine, reports readiness, and serves commands until stdin closes.
    pub async fn run<ReadTransport, WriteTransport>(
        self,
        read_transport: ReadTransport,
        write_transport: WriteTransport,
    ) -> Result<(), WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        self.run_with_protocol(
            ProtocolReader::new(read_transport),
            ProtocolWriter::new(write_transport),
        )
        .await
    }

    /// Serves commands using already-created protocol transports.
    pub async fn run_with_protocol<ReadTransport, WriteTransport>(
        mut self,
        command_reader: ProtocolReader<ReadTransport>,
        mut event_writer: ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Autoregressive(loaded_model)) => {
                let engine_load_result =
                    loaded_model.engine.load().await.map_err(|engine_error| {
                        WorkerRuntimeError::InferenceEngineInitializationFailed {
                            reason: engine_error.to_string(),
                        }
                    })?;
                self.minimum_mlx_memory_ceiling_bytes =
                    engine_load_result.minimum_mlx_memory_ceiling_bytes();
                event_writer
                    .send_event(
                        &loaded_model.processor.ready_event(
                            engine_load_result.mtp_runtime_state(),
                            engine_load_result
                                .mtp_unavailable_reason()
                                .map(String::from),
                            engine_load_result.mtp_depth_status(),
                            engine_load_result.speculative_prefill_runtime_state(),
                            engine_load_result
                                .speculative_prefill_unavailable_reason()
                                .map(String::from),
                            engine_load_result
                                .speculative_prefill_draft_model_id()
                                .map(String::from),
                            engine_load_result
                                .speculative_prefill_draft_model_revision()
                                .map(String::from),
                        ),
                    )
                    .await?;
                if let Some(worker_runtime_feature_configuration) =
                    self.worker_runtime_feature_configuration()
                {
                    event_writer
                        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                            worker_runtime_feature_configuration,
                        })
                        .await?;
                }
                self.emit_mlx_memory_sample(
                    MlxMemorySnapshotSource::ModelLoaded,
                    &mut event_writer,
                )
                .await?;
                self.emit_persistent_prompt_cache_stats(&mut event_writer)
                    .await?;
            }
            Some(LoadedRuntime::Image(image_engine)) => {
                let image_load_result = image_engine.load().map_err(|failure_reason| {
                    WorkerRuntimeError::InferenceEngineInitializationFailed {
                        reason: format!("image engine initialization failed: {failure_reason:?}"),
                    }
                })?;
                self.minimum_mlx_memory_ceiling_bytes =
                    image_load_result.minimum_mlx_memory_ceiling_bytes();
                event_writer
                    .send_event(&WorkerEvent::Ready {
                        model_id: image_load_result.model_id().to_owned(),
                        capabilities: WorkerModelCapabilities::image_generation(
                            image_load_result.capabilities().clone(),
                        ),
                        mtp_runtime_state: MtpRuntimeState::Disabled,
                        mtp_unavailable_reason: None,
                        mtp_depth_status: Default::default(),
                        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
                        speculative_prefill_unavailable_reason: None,
                        speculative_prefill_draft_model_id: None,
                        speculative_prefill_draft_model_revision: None,
                    })
                    .await?;
                self.emit_mlx_memory_sample(
                    MlxMemorySnapshotSource::ModelLoaded,
                    &mut event_writer,
                )
                .await?;
            }
            None => {
                event_writer
                    .send_event(&WorkerEvent::Idle {
                        machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                        effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                    })
                    .await?;
                if let Some(worker_runtime_feature_configuration) =
                    self.worker_runtime_feature_configuration()
                {
                    event_writer
                        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                            worker_runtime_feature_configuration,
                        })
                        .await?;
                }
            }
        }
        self.serve_protocol(command_reader, event_writer).await
    }

    pub(super) fn worker_runtime_feature_configuration(
        &self,
    ) -> Option<WorkerRuntimeFeatureConfiguration> {
        self.worker_runtime_feature_configuration.clone()
    }

    async fn serve_protocol<ReadTransport, WriteTransport>(
        mut self,
        mut command_reader: ProtocolReader<ReadTransport>,
        mut event_writer: ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        let mut active_request = None;
        loop {
            let Some(current_request) = active_request.take() else {
                let Some(worker_command) = command_reader.next_command().await? else {
                    return Ok(());
                };
                active_request = self
                    .serve_idle_command(worker_command, &mut event_writer)
                    .await?;
                continue;
            };
            match current_request {
                ActiveWorkerRequest::Autoregressive(mut current_generation) => tokio::select! {
                    biased;
                    next_command = command_reader.next_command() => {
                        let Some(worker_command) = next_command? else { return Ok(()); };
                        match worker_command {
                            WorkerCommand::InitializeWorker(_) => active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation)),
                            WorkerCommand::Cancel { request_id } if request_id == current_generation.request_id => {
                                let generation_finalization = self.cancel_active_engine_request(request_id).await?;
                                self.emit_generation_finalization(
                                    &mut current_generation,
                                    generation_finalization,
                                    &mut event_writer,
                                ).await?;
                                self.send_completion(&current_generation, ChatGenerationCompletionReason::Cancelled, &mut event_writer).await?;
                            }
                            WorkerCommand::Cancel { .. } => active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation)),
                            WorkerCommand::Generate(generation_command) => {
                                event_writer.send_event(&WorkerEvent::Failed { request_id: generation_command.request_id, reason: ChatGenerationFailureReason::EngineBusy }).await?;
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                            WorkerCommand::GenerateImage(generation_command) => {
                                self.emit_image_failure_and_finalization(generation_command.request_id, ImageGenerationFailureReason::EngineBusy, 0, &mut event_writer).await?;
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                            WorkerCommand::SwapModel { .. } => {
                                tracing::warn!(request_id = current_generation.request_id.value(), "received SwapModel command while generation is active; ignoring");
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                            WorkerCommand::SampleMlxMemory => {
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                            WorkerCommand::UpdateMlxMemoryLimit {
                                effective_mlx_memory_ceiling_bytes,
                            } => {
                                event_writer
                                    .send_event(&WorkerEvent::MlxMemoryLimitRejected {
                                        requested_mlx_memory_ceiling_bytes:
                                            effective_mlx_memory_ceiling_bytes,
                                        minimum_mlx_memory_ceiling_bytes:
                                            self.minimum_mlx_memory_ceiling_bytes,
                                        machine_mlx_memory_ceiling_bytes:
                                            self.machine_mlx_memory_ceiling_bytes,
                                        reason: "memory limits cannot change during generation"
                                            .to_owned(),
                                    })
                                    .await?;
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                            WorkerCommand::ClearPromptCache { .. } => {
                                tracing::warn!(request_id = current_generation.request_id.value(), "received ClearPromptCache command while generation is active; ignoring");
                                active_request = Some(ActiveWorkerRequest::Autoregressive(current_generation));
                            }
                        }
                    }
                    () = tokio::task::yield_now() => {
                        active_request = self.advance_generation(*current_generation, &mut event_writer).await?.map(Box::new).map(ActiveWorkerRequest::Autoregressive);
                    }
                },
                ActiveWorkerRequest::Image(current_generation) => tokio::select! {
                    biased;
                    next_command = command_reader.next_command() => {
                        let Some(worker_command) = next_command? else {
                            self.cancel_image_generation(current_generation, &mut event_writer).await?;
                            return Ok(());
                        };
                        match worker_command {
                            WorkerCommand::Cancel { request_id } if request_id == current_generation.request_id => {
                                self.cancel_image_generation(current_generation, &mut event_writer).await?;
                            }
                            WorkerCommand::Generate(generation_command) => {
                                event_writer.send_event(&WorkerEvent::Failed { request_id: generation_command.request_id, reason: ChatGenerationFailureReason::EngineBusy }).await?;
                                active_request = Some(ActiveWorkerRequest::Image(current_generation));
                            }
                            WorkerCommand::GenerateImage(generation_command) => {
                                self.emit_image_failure_and_finalization(generation_command.request_id, ImageGenerationFailureReason::EngineBusy, 0, &mut event_writer).await?;
                                active_request = Some(ActiveWorkerRequest::Image(current_generation));
                            }
                            WorkerCommand::UpdateMlxMemoryLimit { effective_mlx_memory_ceiling_bytes } => {
                                event_writer.send_event(&WorkerEvent::MlxMemoryLimitRejected {
                                    requested_mlx_memory_ceiling_bytes: effective_mlx_memory_ceiling_bytes,
                                    minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                                    machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                                    reason: "memory limits cannot change during generation".to_owned(),
                                }).await?;
                                active_request = Some(ActiveWorkerRequest::Image(current_generation));
                            }
                            WorkerCommand::Cancel { .. } | WorkerCommand::InitializeWorker(_) | WorkerCommand::SampleMlxMemory | WorkerCommand::SwapModel { .. } | WorkerCommand::ClearPromptCache { .. } => {
                                active_request = Some(ActiveWorkerRequest::Image(current_generation));
                            }
                        }
                    }
                    () = tokio::task::yield_now() => {
                        active_request = self.advance_image_generation(current_generation, &mut event_writer).await?.map(ActiveWorkerRequest::Image);
                    }
                },
            }
        }
    }
}
