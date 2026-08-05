use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, ProtocolWriter, WorkerCommand, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::support::{ActiveEngineGeneration, ModelFactory, WorkerRuntimeError};
use crate::InferenceEngine;
use crate::model_generation_processor::ModelGenerationProcessor;

use super::EngineBackedWorker;

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine> + Send + 'static,
{
    pub(crate) async fn serve_idle_command<WriteTransport>(
        &mut self,
        worker_command: WorkerCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match worker_command {
            WorkerCommand::InitializeWorker(_) => Ok(None),
            WorkerCommand::Generate(generation_command) => {
                self.start_generation(generation_command, event_writer)
                    .await
            }
            WorkerCommand::Cancel { .. } => Ok(None),
            WorkerCommand::SampleMlxMemory => {
                self.emit_mlx_memory_sample(MlxMemorySnapshotSource::IdlePoll, event_writer)
                    .await?;
                Ok(None)
            }
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
            } => {
                self.update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes, event_writer)
                    .await?;
                Ok(None)
            }
            WorkerCommand::SwapModel {
                model_directory,
                max_output_tokens,
            } => {
                if let Err(swap_error) = self
                    .swap_model(&model_directory, max_output_tokens, event_writer)
                    .await
                {
                    let loaded_model_remains_ready = self.loaded_model.is_some();
                    let model_load_failure_reason = match &swap_error {
                        WorkerRuntimeError::ModelSwapFailed {
                            model_load_failure_reason,
                        } => model_load_failure_reason.clone(),
                        _ => "model initialization failed".to_owned(),
                    };
                    tracing::error!(
                        model_directory,
                        error = %swap_error,
                        loaded_model_remains_ready,
                        "model swap failed"
                    );
                    event_writer
                        .send_event(&WorkerEvent::ModelSwapFailed {
                            loaded_model_remains_ready,
                            model_load_failure_reason,
                        })
                        .await?;
                }
                Ok(None)
            }
        }
    }
}
