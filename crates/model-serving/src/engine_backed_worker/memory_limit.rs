use astronomical_ipc_protocol::{
    ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::output::{worker_expert_residency_snapshot, worker_memory_snapshot};
use super::support::{ModelFactory, WorkerRuntimeError, engine_generation_error};
use crate::{
    ImageGenerationEngine, InferenceEngine, InferenceEngineError, ModelGenerationProcessor,
};

use super::{EngineBackedWorker, LoadedRuntime};

impl<Processor, Engine, Factory, ImageEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
{
    pub(crate) async fn update_mlx_memory_limit<WriteTransport>(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
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
        let Some(loaded_runtime) = self.loaded_runtime.as_mut() else {
            let Some(model_factory) = self.model_factory.as_mut() else {
                return self
                    .emit_mlx_memory_limit_rejection(
                        requested_mlx_memory_ceiling_bytes,
                        "model factory is unavailable",
                        event_writer,
                    )
                    .await;
            };
            model_factory.update_mlx_memory_limits(
                requested_mlx_memory_ceiling_bytes,
                requested_mlx_memory_ceiling_bytes,
            );
            self.effective_mlx_memory_ceiling_bytes = requested_mlx_memory_ceiling_bytes;
            self.record_configuration_generation(configuration_generation);
            event_writer
                .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                    effective_mlx_memory_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                    expert_memory_mode: ExpertMemoryMode::Resident,
                    mlx_memory_snapshot: None,
                    expert_residency: None,
                })
                .await?;
            return Ok(());
        };
        let memory_limit_adjustment = match loaded_runtime {
            LoadedRuntime::Autoregressive(loaded_model) => {
                loaded_model
                    .engine
                    .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
                    .await
            }
            LoadedRuntime::Image(image_engine) => image_engine
                .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
                .map_err(|failure_reason| match failure_reason {
                    astronomical_ipc_protocol::ImageGenerationFailureReason::EngineBusy => {
                        InferenceEngineError::EngineBusy
                    }
                    astronomical_ipc_protocol::ImageGenerationFailureReason::InvalidRequest {
                        reason,
                    }
                    | astronomical_ipc_protocol::ImageGenerationFailureReason::EncodingFailed {
                        reason,
                    }
                    | astronomical_ipc_protocol::ImageGenerationFailureReason::FatalExecution {
                        reason,
                    } => InferenceEngineError::Fatal { reason },
                    other => InferenceEngineError::Fatal {
                        reason: format!("image memory-limit update failed: {other:?}"),
                    },
                }),
        };
        match memory_limit_adjustment {
            Ok(mlx_memory_limit_adjustment) => {
                self.effective_mlx_memory_ceiling_bytes =
                    mlx_memory_limit_adjustment.effective_mlx_memory_ceiling_bytes();
                self.minimum_mlx_memory_ceiling_bytes =
                    mlx_memory_limit_adjustment.minimum_mlx_memory_ceiling_bytes();
                if let Some(model_factory) = self.model_factory.as_mut() {
                    model_factory.update_mlx_memory_limits(
                        self.effective_mlx_memory_ceiling_bytes,
                        mlx_memory_limit_adjustment.allocator_cache_memory_limit_bytes(),
                    );
                }
                self.record_configuration_generation(configuration_generation);
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
                        expert_residency: mlx_memory_limit_adjustment
                            .expert_residency_telemetry()
                            .map(worker_expert_residency_snapshot),
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

    fn record_configuration_generation(&mut self, configuration_generation: String) {
        if let Some(worker_runtime_feature_configuration) =
            self.worker_runtime_feature_configuration.as_mut()
        {
            worker_runtime_feature_configuration.configuration_generation =
                configuration_generation;
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
