use astronomical_ipc_protocol::{
    ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent,
};
use tokio::io::AsyncWrite;

use crate::engine_backed_worker_output::worker_memory_snapshot;
use crate::engine_backed_worker_support::{
    ModelFactory, WorkerRuntimeError, engine_generation_error,
};
use crate::{InferenceEngine, InferenceEngineError, ModelGenerationProcessor};

use super::EngineBackedWorker;

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine> + Send + 'static,
{
    pub(crate) async fn update_mlx_memory_limit<WriteTransport>(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if requested_mlx_memory_ceiling_bytes == 0
            || requested_mlx_memory_ceiling_bytes > self.machine_mlx_memory_ceiling_bytes
        {
            return self
                .emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    "requested memory ceiling is outside the worker machine limit",
                    event_writer,
                )
                .await;
        }
        let Some(loaded_model) = self.loaded_model.as_mut() else {
            let Some(model_factory) = self.model_factory.as_mut() else {
                return self
                    .emit_mlx_memory_limit_rejection(
                        requested_mlx_memory_ceiling_bytes,
                        "model factory is unavailable",
                        event_writer,
                    )
                    .await;
            };
            model_factory.update_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
            self.effective_mlx_memory_ceiling_bytes = requested_mlx_memory_ceiling_bytes;
            event_writer
                .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                    effective_mlx_memory_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                    expert_memory_mode: ExpertMemoryMode::Resident,
                    mlx_memory_snapshot: None,
                })
                .await?;
            return Ok(());
        };
        match loaded_model
            .engine
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
            .await
        {
            Ok(mlx_memory_limit_adjustment) => {
                self.effective_mlx_memory_ceiling_bytes =
                    mlx_memory_limit_adjustment.effective_mlx_memory_ceiling_bytes();
                self.minimum_mlx_memory_ceiling_bytes =
                    mlx_memory_limit_adjustment.minimum_mlx_memory_ceiling_bytes();
                if let Some(model_factory) = self.model_factory.as_mut() {
                    model_factory
                        .update_mlx_memory_ceiling_bytes(self.effective_mlx_memory_ceiling_bytes);
                }
                event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                        expert_memory_mode: mlx_memory_limit_adjustment.expert_memory_mode(),
                        mlx_memory_snapshot: mlx_memory_limit_adjustment
                            .mlx_memory_telemetry()
                            .map(|mlx_memory_telemetry| {
                                worker_memory_snapshot(
                                    MlxMemorySnapshotSource::MemoryLimitAdjusted,
                                    mlx_memory_telemetry,
                                )
                            }),
                    })
                    .await?;
                Ok(())
            }
            Err(InferenceEngineError::MlxMemoryLimitRejected {
                minimum_mlx_memory_ceiling_bytes,
                reason,
                ..
            }) => {
                self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
                self.emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    &reason,
                    event_writer,
                )
                .await
            }
            Err(InferenceEngineError::EngineBusy) => {
                self.emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    "memory limits cannot change during generation",
                    event_writer,
                )
                .await
            }
            Err(engine_error) => Err(engine_generation_error(engine_error)),
        }
    }

    async fn emit_mlx_memory_limit_rejection<WriteTransport>(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        reason: &str,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                reason: reason.chars().take(256).collect(),
            })
            .await?;
        Ok(())
    }
}
