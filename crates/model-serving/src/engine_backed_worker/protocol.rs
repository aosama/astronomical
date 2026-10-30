//! Protocol startup and command-loop orchestration for the engine-backed worker.

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, EmbeddingsFailureReason,
    ImageGenerationFailureReason, MlxMemorySnapshotSource, ProtocolReader, ProtocolWriter,
    WorkerCommand, WorkerEvent, WorkerModelCapabilities, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::support::{
    ActiveEngineGeneration, ActiveImageGeneration, ActiveWorkerRequest, ModelFactory,
    WorkerRuntimeError,
};
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
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
        // The runtime detaches for the announcement so the announce units
        // can take &mut self; it is reinstalled before any command is served.
        let mut loaded_runtime = self.loaded_runtime.take();
        match loaded_runtime.as_mut() {
            Some(LoadedRuntime::Autoregressive(loaded_model)) => {
                self.announce_autoregressive_readiness(loaded_model, &mut event_writer)
                    .await?;
            }
            Some(LoadedRuntime::Image(image_engine)) => {
                self.announce_image_readiness(image_engine, &mut event_writer)
                    .await?;
            }
            Some(LoadedRuntime::Embeddings(embedding_engine)) => {
                self.announce_embeddings_readiness(embedding_engine, &mut event_writer)
                    .await?;
            }
            None => self.announce_idle_readiness(&mut event_writer).await?,
        }
        self.loaded_runtime = loaded_runtime;
        self.serve_protocol(command_reader, event_writer).await
    }

    /// Loads the autoregressive engine and announces readiness with the
    /// processor's ready event, the expert-mode fact, any pending runtime
    /// feature configuration, and the first memory/prompt-cache samples.
    async fn announce_autoregressive_readiness<WriteTransport>(
        &mut self,
        loaded_model: &mut LoadedModel<Processor, Engine>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let engine_load_result = loaded_model.engine.load().await.map_err(|engine_error| {
            WorkerRuntimeError::InferenceEngineInitializationFailed {
                reason: engine_error.to_string(),
            }
        })?;
        self.minimum_mlx_memory_ceiling_bytes =
            engine_load_result.minimum_mlx_memory_ceiling_bytes();
        event_writer
            .send_event(&loaded_model.processor.ready_event())
            .await?;
        // The menu's expert-mode label must be reported fact from the moment readiness
        // is published. Without this event the label stays a daemon-side fallback
        // guess until the first generation starts, which mislabels paged loads.
        if let Some(expert_memory_mode) = engine_load_result.expert_memory_mode() {
            event_writer
                .send_event(&WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode })
                .await?;
        }
        if let Some(worker_runtime_feature_configuration) =
            self.worker_runtime_feature_configuration()
        {
            event_writer
                .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                    worker_runtime_feature_configuration,
                })
                .await?;
        }
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await?;
        self.emit_persistent_prompt_cache_stats(event_writer).await
    }

    /// Loads the image engine and announces its readiness event and the
    /// first memory sample.
    async fn announce_image_readiness<WriteTransport>(
        &mut self,
        image_engine: &mut ImageEngine,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
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
            })
            .await?;
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await
    }

    /// Loads the embeddings engine and announces its readiness event and the
    /// first memory sample.
    async fn announce_embeddings_readiness<WriteTransport>(
        &mut self,
        embedding_engine: &mut EmbeddingsEngine,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let embedding_load_result = embedding_engine.load().map_err(|failure_reason| {
            WorkerRuntimeError::InferenceEngineInitializationFailed {
                reason: format!("embedding engine initialization failed: {failure_reason:?}"),
            }
        })?;
        self.minimum_mlx_memory_ceiling_bytes =
            embedding_load_result.minimum_mlx_memory_ceiling_bytes();
        event_writer
            .send_event(&WorkerEvent::Ready {
                model_id: embedding_load_result.model_id().to_owned(),
                capabilities: WorkerModelCapabilities::embeddings(
                    *embedding_load_result.capabilities(),
                ),
            })
            .await?;
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await
    }

    /// Announces the idle envelope: the worker serves no model yet, so only
    /// the memory ceilings and any pending runtime feature configuration
    /// are published.
    async fn announce_idle_readiness<WriteTransport>(
        &mut self,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::Idle {
                machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                wired_residency_budget_bytes: self.wired_residency_budget_bytes,
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
        Ok(())
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
            active_request = match current_request {
                ActiveWorkerRequest::Autoregressive(current_generation) => {
                    self.serve_active_autoregressive_request(
                        current_generation,
                        &mut command_reader,
                        &mut event_writer,
                    )
                    .await?
                }
                ActiveWorkerRequest::Image(current_generation) => {
                    self.serve_active_image_request(
                        current_generation,
                        &mut command_reader,
                        &mut event_writer,
                    )
                    .await?
                }
            };
        }
    }

    /// Runs one autoregressive generation to its next yield: a worker
    /// command either replaces the active request state or completes it,
    /// and every otherwise-idle loop tick advances the generation.
    async fn serve_active_autoregressive_request<ReadTransport, WriteTransport>(
        &mut self,
        current_generation: Box<
            ActiveEngineGeneration<Processor::RequestOutput, Processor::InferenceRequest>,
        >,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<
        Option<ActiveWorkerRequest<Processor::RequestOutput, Processor::InferenceRequest>>,
        WorkerRuntimeError,
    >
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        let mut current_generation = current_generation;
        tokio::select! {
            biased;
            next_command = command_reader.next_command() => {
                let Some(worker_command) = next_command? else { return Ok(None); };
                match worker_command {
                    WorkerCommand::InitializeWorker(_) => Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation))),
                    WorkerCommand::Cancel { request_id } if request_id == current_generation.request_id => {
                        let generation_finalization = self.cancel_active_engine_request(request_id).await?;
                        self.emit_generation_finalization(
                            &mut current_generation,
                            generation_finalization,
                            event_writer,
                        ).await?;
                        self.send_completion(&current_generation, ChatGenerationCompletionReason::Cancelled, event_writer).await?;
                        Ok(None)
                    }
                    WorkerCommand::Cancel { .. } => Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation))),
                    WorkerCommand::Generate(generation_command) => {
                        event_writer.send_event(&WorkerEvent::Failed { request_id: generation_command.request_id, reason: ChatGenerationFailureReason::EngineBusy }).await?;
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::GenerateImage(generation_command) => {
                        self.emit_image_failure_and_finalization(generation_command.request_id, ImageGenerationFailureReason::EngineBusy, 0, event_writer).await?;
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::GenerateEmbeddings(embeddings_command) => {
                        event_writer.send_event(&WorkerEvent::EmbeddingsFailed { request_id: embeddings_command.request_id, reason: EmbeddingsFailureReason::EngineBusy }).await?;
                        event_writer.send_event(&WorkerEvent::EmbeddingsFinalized { request_id: embeddings_command.request_id, elapsed_millis: 0, mlx_memory_snapshot: None }).await?;
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::SwapModel { .. } => {
                        tracing::warn!(request_id = current_generation.request_id.value(), "received SwapModel command while generation is active; ignoring");
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::SampleMlxMemory => {
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::UpdateMlxMemoryLimit {
                        effective_mlx_memory_ceiling_bytes,
                        configuration_generation: _,
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
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                    WorkerCommand::ClearPromptCache { .. } => {
                        tracing::warn!(request_id = current_generation.request_id.value(), "received ClearPromptCache command while generation is active; ignoring");
                        Ok(Some(ActiveWorkerRequest::Autoregressive(current_generation)))
                    }
                }
            }
            () = tokio::task::yield_now() => {
                Ok(self
                    .advance_generation(*current_generation, event_writer)
                    .await?
                    .map(Box::new)
                    .map(ActiveWorkerRequest::Autoregressive))
            }
        }
    }

    /// Runs one image generation to its next yield: a worker command either
    /// keeps the generation active or cancels/completes it, and every
    /// otherwise-idle loop tick advances the generation.
    async fn serve_active_image_request<ReadTransport, WriteTransport>(
        &mut self,
        current_generation: ActiveImageGeneration,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<
        Option<ActiveWorkerRequest<Processor::RequestOutput, Processor::InferenceRequest>>,
        WorkerRuntimeError,
    >
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        tokio::select! {
            biased;
            next_command = command_reader.next_command() => {
                let Some(worker_command) = next_command? else {
                    self.cancel_image_generation(current_generation, event_writer).await?;
                    return Ok(None);
                };
                match worker_command {
                    WorkerCommand::Cancel { request_id } if request_id == current_generation.request_id => {
                        self.cancel_image_generation(current_generation, event_writer).await?;
                        Ok(None)
                    }
                    WorkerCommand::Generate(generation_command) => {
                        event_writer.send_event(&WorkerEvent::Failed { request_id: generation_command.request_id, reason: ChatGenerationFailureReason::EngineBusy }).await?;
                        Ok(Some(ActiveWorkerRequest::Image(current_generation)))
                    }
                    WorkerCommand::GenerateImage(generation_command) => {
                        self.emit_image_failure_and_finalization(generation_command.request_id, ImageGenerationFailureReason::EngineBusy, 0, event_writer).await?;
                        Ok(Some(ActiveWorkerRequest::Image(current_generation)))
                    }
                    WorkerCommand::GenerateEmbeddings(embeddings_command) => {
                        event_writer.send_event(&WorkerEvent::EmbeddingsFailed { request_id: embeddings_command.request_id, reason: EmbeddingsFailureReason::EngineBusy }).await?;
                        event_writer.send_event(&WorkerEvent::EmbeddingsFinalized { request_id: embeddings_command.request_id, elapsed_millis: 0, mlx_memory_snapshot: None }).await?;
                        Ok(Some(ActiveWorkerRequest::Image(current_generation)))
                    }
                    WorkerCommand::UpdateMlxMemoryLimit { effective_mlx_memory_ceiling_bytes, configuration_generation: _ } => {
                        event_writer.send_event(&WorkerEvent::MlxMemoryLimitRejected {
                            requested_mlx_memory_ceiling_bytes: effective_mlx_memory_ceiling_bytes,
                            minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                            machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                            reason: "memory limits cannot change during generation".to_owned(),
                        }).await?;
                        Ok(Some(ActiveWorkerRequest::Image(current_generation)))
                    }
                    WorkerCommand::Cancel { .. } | WorkerCommand::InitializeWorker(_) | WorkerCommand::SampleMlxMemory | WorkerCommand::SwapModel { .. } | WorkerCommand::ClearPromptCache { .. } => {
                        Ok(Some(ActiveWorkerRequest::Image(current_generation)))
                    }
                }
            }
            () = tokio::task::yield_now() => {
                Ok(self
                    .advance_image_generation(current_generation, event_writer)
                    .await?
                    .map(ActiveWorkerRequest::Image))
            }
        }
    }
}
