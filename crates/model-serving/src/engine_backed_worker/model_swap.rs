use astronomical_ipc_protocol::{MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent};
use tokio::io::AsyncWrite;

use super::support::{ModelFactory, WorkerRuntimeError};
use crate::InferenceEngine;
use crate::model_generation_processor::ModelGenerationProcessor;

use super::{EngineBackedWorker, LoadedModel};

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine> + Send + 'static,
{
    pub(super) async fn swap_model<WriteTransport>(
        &mut self,
        model_directory: &str,
        max_output_tokens: u32,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(model_factory) = self.model_factory.as_ref() else {
            tracing::error!("received SwapModel command but no model factory is configured");
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason: "model swapping is unavailable".to_owned(),
            });
        };
        tracing::info!(model_directory, max_output_tokens, "starting model swap");
        let (new_processor, new_engine) = model_factory
            .create(model_directory, max_output_tokens)
            .await
            .map_err(|model_load_failure_reason| {
                tracing::error!(
                    model_directory,
                    model_load_failure_reason = %model_load_failure_reason,
                    "model swap creation failed"
                );
                WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason,
                }
            })?;
        drop(self.loaded_model.take());
        let mut replacement_model = LoadedModel {
            processor: new_processor,
            engine: new_engine,
        };
        let engine_load_result = replacement_model
            .engine
            .load()
            .await
            .map_err(|engine_error| {
                tracing::error!(
                    model_directory,
                    error = %engine_error,
                    "model engine load failed after swap creation"
                );
                WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason: "model engine initialization failed".to_owned(),
                }
            })?;
        let minimum_mlx_memory_ceiling_bytes =
            engine_load_result.minimum_mlx_memory_ceiling_bytes();
        let mtp_runtime_state = engine_load_result.mtp_runtime_state();
        let mtp_unavailable_reason = engine_load_result
            .mtp_unavailable_reason()
            .map(String::from);
        let model_swapped_event = match replacement_model
            .processor
            .ready_event(mtp_runtime_state, mtp_unavailable_reason)
        {
            WorkerEvent::Ready {
                model_id,
                capabilities,
                mtp_runtime_state,
                mtp_unavailable_reason,
            } => WorkerEvent::ModelSwapped {
                model_id,
                capabilities,
                minimum_mlx_memory_ceiling_bytes,
                mtp_runtime_state,
                mtp_unavailable_reason,
            },
            other => {
                tracing::error!(?other, "expected Ready event from new processor after swap");
                return Err(WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason: "model processor did not become ready".to_owned(),
                });
            }
        };
        tracing::info!(model_id = ?model_swapped_event, "model swap completed successfully");
        self.loaded_model = Some(replacement_model);
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        event_writer.send_event(&model_swapped_event).await?;
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await?;
        self.emit_persistent_prompt_cache_stats(event_writer)
            .await?;
        Ok(())
    }
}
