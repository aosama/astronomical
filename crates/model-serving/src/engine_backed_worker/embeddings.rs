//! Serializes one synchronous embeddings request and publishes its cleanup event.

use astronomical_ipc_protocol::{
    EmbeddingsCommand, EmbeddingsFailureReason, MlxMemorySnapshotSource, ProtocolWriter, RequestId,
    WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::WorkerRuntimeError;
use super::output::worker_memory_snapshot;
use super::{EngineBackedWorker, LoadedRuntime};
use crate::EmbeddingEngine;
use crate::InferenceEngine;
use crate::ModelGenerationProcessor;
use crate::engine_backed_worker::support::ModelFactory;

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: crate::ImageGenerationEngine,
    EmbeddingsEngine: EmbeddingEngine,
{
    pub(crate) async fn start_embeddings<WriteTransport>(
        &mut self,
        embeddings_command: EmbeddingsCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = embeddings_command.request_id;
        if let Err(validation_error) = embeddings_command.validate() {
            return self
                .emit_embeddings_failure_and_finalization(
                    request_id,
                    EmbeddingsFailureReason::invalid_request(validation_error.to_string()),
                    0,
                    event_writer,
                )
                .await;
        }
        let started_at = std::time::Instant::now();
        let embed_outcome = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Embeddings(embedding_engine)) => {
                embedding_engine.embed(&embeddings_command)
            }
            Some(LoadedRuntime::Autoregressive(_)) | Some(LoadedRuntime::Image(_)) | None => {
                Err(EmbeddingsFailureReason::FatalExecution {
                    reason: "the loaded model does not support embeddings".to_owned(),
                })
            }
        };
        match embed_outcome {
            Ok(engine_output) => {
                event_writer
                    .send_event(&WorkerEvent::EmbeddingsCompleted {
                        request_id,
                        embeddings: engine_output.embeddings,
                        input_token_counts: engine_output.input_token_counts,
                        elapsed_millis: engine_output.elapsed_millis,
                    })
                    .await?;
                self.emit_embeddings_finalization(
                    request_id,
                    engine_output
                        .elapsed_millis
                        .max(Self::elapsed_since(started_at)),
                    event_writer,
                )
                .await
            }
            Err(failure_reason) => {
                self.emit_embeddings_failure_and_finalization(
                    request_id,
                    failure_reason,
                    Self::elapsed_since(started_at),
                    event_writer,
                )
                .await
            }
        }
    }

    async fn emit_embeddings_failure_and_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        failure_reason: EmbeddingsFailureReason,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::EmbeddingsFailed {
                request_id,
                reason: bounded_embeddings_failure_reason(failure_reason),
            })
            .await?;
        self.emit_embeddings_finalization(request_id, elapsed_millis, event_writer)
            .await
    }

    async fn emit_embeddings_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        // The engine captured its post-cleanup observation during the embed
        // call, so the finalization publishes the same split the menu paints
        // (issue #510).
        let mlx_memory_snapshot = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Embeddings(embedding_engine)) => embedding_engine
                .take_post_cleanup_memory_telemetry()
                .map(|mlx_memory_telemetry| {
                    worker_memory_snapshot(MlxMemorySnapshotSource::Finalized, mlx_memory_telemetry)
                }),
            _ => None,
        };
        event_writer
            .send_event(&WorkerEvent::EmbeddingsFinalized {
                request_id,
                elapsed_millis,
                mlx_memory_snapshot,
            })
            .await?;
        Ok(())
    }

    fn elapsed_since(started_at: std::time::Instant) -> u64 {
        u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

fn bounded_embeddings_failure_reason(
    failure_reason: EmbeddingsFailureReason,
) -> EmbeddingsFailureReason {
    match failure_reason {
        EmbeddingsFailureReason::InvalidRequest { reason } => {
            EmbeddingsFailureReason::InvalidRequest {
                reason: reason.chars().take(256).collect(),
            }
        }
        EmbeddingsFailureReason::FatalExecution { reason } => {
            EmbeddingsFailureReason::FatalExecution {
                reason: reason.chars().take(256).collect(),
            }
        }
        other_failure_reason => other_failure_reason,
    }
}
