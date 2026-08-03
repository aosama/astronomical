//! Protocol startup and command-loop orchestration for the engine-backed worker.

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, MlxMemorySnapshotSource,
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerEvent,
};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::engine_backed_worker::EngineBackedWorker;
use crate::engine_backed_worker_support::{ModelFactory, WorkerRuntimeError};
use crate::{InferenceEngine, ModelGenerationProcessor};

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine> + Send + 'static,
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
        if let Some(loaded_model) = self.loaded_model.as_mut() {
            let engine_load_result = loaded_model.engine.load().await.map_err(|engine_error| {
                WorkerRuntimeError::InferenceEngineInitializationFailed {
                    reason: engine_error.to_string(),
                }
            })?;
            self.minimum_mlx_memory_ceiling_bytes =
                engine_load_result.minimum_mlx_memory_ceiling_bytes();
            event_writer
                .send_event(
                    &loaded_model.processor.ready_event(
                        engine_load_result.expert_storage_format(),
                        engine_load_result.mtp_runtime_state(),
                        engine_load_result
                            .mtp_unavailable_reason()
                            .map(String::from),
                    ),
                )
                .await?;
            self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, &mut event_writer)
                .await?;
            self.emit_persistent_prompt_cache_stats(&mut event_writer)
                .await?;
        } else {
            event_writer
                .send_event(&WorkerEvent::Idle {
                    machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                    effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                })
                .await?;
        }
        self.serve_protocol(command_reader, event_writer).await
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
        let mut active_generation = None;
        loop {
            let Some(mut current_generation) = active_generation.take() else {
                let Some(worker_command) = command_reader.next_command().await? else {
                    return Ok(());
                };
                active_generation = self
                    .serve_idle_command(worker_command, &mut event_writer)
                    .await?;
                continue;
            };
            tokio::select! {
                biased;
                next_command = command_reader.next_command() => {
                    let Some(worker_command) = next_command? else { return Ok(()); };
                    match worker_command {
                        WorkerCommand::InitializeWorker(_) => active_generation = Some(current_generation),
                        WorkerCommand::Cancel { request_id } if request_id == current_generation.request_id => {
                            let generation_finalization = self.cancel_active_engine_request(request_id).await?;
                            self.emit_generation_finalization(
                                &mut current_generation,
                                generation_finalization,
                                &mut event_writer,
                            ).await?;
                            self.send_completion(&current_generation, ChatGenerationCompletionReason::Cancelled, &mut event_writer).await?;
                        }
                        WorkerCommand::Cancel { .. } => active_generation = Some(current_generation),
                        WorkerCommand::Generate(generation_command) => {
                            event_writer.send_event(&WorkerEvent::Failed { request_id: generation_command.request_id, reason: ChatGenerationFailureReason::EngineBusy }).await?;
                            active_generation = Some(current_generation);
                        }
                        WorkerCommand::SwapModel { .. } => {
                            tracing::warn!(request_id = current_generation.request_id.value(), "received SwapModel command while generation is active; ignoring");
                            active_generation = Some(current_generation);
                        }
                        WorkerCommand::SampleMlxMemory => {
                            active_generation = Some(current_generation);
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
                            active_generation = Some(current_generation);
                        }
                    }
                }
                () = tokio::task::yield_now() => {
                    active_generation = self.advance_generation(current_generation, &mut event_writer).await?;
                }
            }
        }
    }
}
