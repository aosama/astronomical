use astronomical_ipc_protocol::{ChatGenerationFailureReason, ProtocolWriter, WorkerEvent};
use tokio::io::AsyncWrite;

use super::fatal::report_fatal_engine_error;
use super::support::{ActiveEngineGeneration, ModelFactory, WorkerRuntimeError};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{InferenceEngine, InferenceEngineError, PreparedInferenceRequest};

use super::EngineBackedWorker;

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine> + Send + 'static,
{
    pub(super) async fn start_generation<WriteTransport>(
        &mut self,
        generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = generation_command.request_id;
        if let Err(validation_error) = generation_command.validate() {
            tracing::warn!(request_id = request_id.value(), error = %validation_error,
                "rejected invalid worker generation command");
            event_writer
                .send_event(&WorkerEvent::Failed {
                    request_id,
                    reason: ChatGenerationFailureReason::invalid_request(
                        validation_error.to_string(),
                    ),
                })
                .await?;
            return Ok(None);
        }

        let Some(loaded_model) = self.loaded_model.as_mut() else {
            return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "generation was requested before a model was loaded".to_owned(),
            });
        };
        let prepared_generation = match loaded_model
            .processor
            .prepare_chat_generation(&generation_command)
        {
            Ok(prepared_generation) => prepared_generation,
            Err(failure_reason) => {
                event_writer
                    .send_event(&WorkerEvent::Failed {
                        request_id,
                        reason: failure_reason,
                    })
                    .await?;
                return Ok(None);
            }
        };
        let prompt_token_count = u32::try_from(
            prepared_generation.inference_request.prompt_token_count(),
        )
        .map_err(|_| WorkerRuntimeError::InferenceEngineGenerationFailed {
            reason: "chat prompt token count exceeds the protocol range".to_owned(),
        })?;
        let inference_request = prepared_generation.inference_request;
        let generation_start = match loaded_model
            .engine
            .start_generation(inference_request)
            .await
        {
            Ok(generation_start) => generation_start,
            Err(InferenceEngineError::EngineBusy) => {
                event_writer
                    .send_event(&WorkerEvent::Failed {
                        request_id,
                        reason: ChatGenerationFailureReason::EngineBusy,
                    })
                    .await?;
                return Ok(None);
            }
            Err(InferenceEngineError::InvalidRequest { reason }) => {
                Self::send_invalid_request_failure(request_id, reason, event_writer).await?;
                return Ok(None);
            }
            Err(engine_error) => {
                return report_fatal_engine_error(request_id, engine_error, event_writer).await;
            }
        };

        self.emit_persistent_prompt_cache_stats(event_writer)
            .await?;
        let cached_prompt_token_count = generation_start.cached_token_count();
        let restored_prompt_prefix_token_count =
            generation_start.restored_prompt_prefix_token_count();
        let initial_expert_memory_mode = generation_start.expert_memory_mode();
        if let Some(expert_memory_mode) = initial_expert_memory_mode {
            tracing::info!(
                request_id = request_id.value(),
                expert_memory_mode = ?expert_memory_mode,
                generation_phase = "start",
                "emitting expert-memory mode event"
            );
            event_writer
                .send_event(&WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode })
                .await?;
        }
        let required_prompt_processing_token_count =
            prompt_token_count.saturating_sub(restored_prompt_prefix_token_count);
        if required_prompt_processing_token_count > 1
            && let Some(prompt_processing_phase) = generation_start.prompt_processing_phase()
        {
            event_writer
                .send_event(&WorkerEvent::PrefillProgress {
                    request_id,
                    prompt_processing_phase,
                    processed_tokens: 0,
                    total_tokens: required_prompt_processing_token_count,
                    elapsed_millis: 0,
                    forward_prefill_chunk_elapsed_millis: None,
                    completed_prefill_chunk_tokens: None,
                    mlx_memory_snapshot: None,
                    expert_residency: None,
                    speculative_prefill_draft_memory_snapshot: None,
                })
                .await?;
        }

        let mut active_engine_generation = ActiveEngineGeneration::new(
            &generation_command,
            prompt_token_count,
            cached_prompt_token_count,
            required_prompt_processing_token_count,
            prepared_generation.request_output,
        );
        active_engine_generation.last_reported_expert_memory_mode = initial_expert_memory_mode;
        active_engine_generation.persistent_prompt_cache_diagnostics = generation_start
            .persistent_prompt_cache_diagnostics()
            .cloned();
        Ok(Some(active_engine_generation))
    }
}
