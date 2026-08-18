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
                model_id,
                model_directory,
                max_output_tokens,
            } => {
                if let Err(swap_error) = self
                    .swap_model(&model_id, &model_directory, max_output_tokens, event_writer)
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
            WorkerCommand::ClearPromptCache { model_id } => {
                self.clear_prompt_cache(model_id, event_writer).await?;
                Ok(None)
            }
        }
    }

    /// Clears the persistent prompt-cache footprint for a scoped or global model identity.
    ///
    /// A loaded engine gets first ownership so its in-memory index is reset on
    /// the MLX owner thread. Engines without an open cache use the factory's
    /// startup root, which also supports disabled-cache and idle-worker clears.
    async fn clear_prompt_cache<WriteTransport>(
        &mut self,
        model_id: Option<String>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let _performance_log = PromptCacheClearPerformanceLog::start(
            self.model_factory
                .as_ref()
                .is_some_and(ModelFactory::performance_attribution_enabled),
            model_id.as_deref(),
        );
        if let Some(loaded_model) = self.loaded_model.as_mut()
            && let Some(clear_event) = loaded_model
                .engine
                .clear_persistent_prompt_cache(model_id.clone())
                .await
                .map_err(
                    |engine_error| WorkerRuntimeError::PersistentPromptCacheClearFailed {
                        reason: engine_error.to_string(),
                    },
                )?
        {
            self.emit_persistent_prompt_cache_stats(event_writer)
                .await?;
            event_writer.send_event(&clear_event).await?;
            return Ok(());
        }

        #[cfg(feature = "direct-mlx")]
        let clear_event = {
            let clear_outcome = match self
                .model_factory
                .as_ref()
                .and_then(|factory| factory.global_prompt_cache_root_directory())
            {
                Some(global_prompt_cache_root_directory) => {
                    crate::clear_persistent_prompt_cache_directory(
                        global_prompt_cache_root_directory,
                        model_id.as_deref(),
                    )
                    .map_err(|clear_error| {
                        WorkerRuntimeError::PersistentPromptCacheClearFailed {
                            reason: clear_error.to_string(),
                        }
                    })?
                }
                None => crate::PersistentPromptCacheClearOutcome {
                    model_id: model_id.clone(),
                    blocks_removed: 0,
                    bytes_freed: 0,
                },
            };
            WorkerEvent::PromptCacheCleared {
                model_id: clear_outcome.model_id,
                blocks_removed: clear_outcome.blocks_removed,
                bytes_freed: clear_outcome.bytes_freed,
            }
        };

        #[cfg(not(feature = "direct-mlx"))]
        let clear_event = WorkerEvent::PromptCacheCleared {
            model_id,
            blocks_removed: 0,
            bytes_freed: 0,
        };

        event_writer.send_event(&clear_event).await?;
        Ok(())
    }
}

/// Emits paired timing boundaries only when performance attribution is enabled.
struct PromptCacheClearPerformanceLog {
    active_operation: Option<(std::time::Instant, Option<String>)>,
}

impl PromptCacheClearPerformanceLog {
    fn start(is_enabled: bool, model_id: Option<&str>) -> Self {
        let active_operation =
            is_enabled.then(|| (std::time::Instant::now(), model_id.map(str::to_owned)));
        if active_operation.is_some() {
            tracing::info!(
                operation = "persistent_prompt_cache_clear",
                phase = "start",
                model_id,
                "performance attribution operation started"
            );
        }
        Self { active_operation }
    }
}

impl Drop for PromptCacheClearPerformanceLog {
    fn drop(&mut self) {
        if let Some((operation_started_at, model_id)) = self.active_operation.as_ref() {
            tracing::info!(
                operation = "persistent_prompt_cache_clear",
                phase = "end",
                elapsed_millis = operation_started_at.elapsed().as_millis(),
                model_id,
                "performance attribution operation completed"
            );
        }
    }
}
