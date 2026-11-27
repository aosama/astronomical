use super::worker_engines::{InferenceRequestOf, RequestOutputOf, WorkerEngines};
use ipc_protocol::{ChatGenerationFailureReason, ProtocolWriter, WorkerEvent};
use tokio::io::AsyncWrite;

use super::fatal;
use super::support::{ActiveEngineGeneration, WorkerRuntimeError};
use crate::inference_engine::contract::EngineGenerationStart;
use crate::inference_engine::contract::PreparedInferenceRequest;
use crate::model_generation_processor::{ModelGenerationProcessor, PreparedModelGeneration};
use crate::{InferenceEngine, InferenceEngineError};

use super::{EngineBackedWorker, LoadedRuntime};

impl<Engines: WorkerEngines> EngineBackedWorker<Engines> {
    pub(super) async fn start_generation<WriteTransport>(
        &mut self,
        generation_command: ipc_protocol::ChatGenerationCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<RequestOutputOf<Engines>>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = generation_command.request_id;
        let Some(prepared_generation) = self
            .prepare_validated_chat_generation(&generation_command, request_id, event_writer)
            .await?
        else {
            return Ok(None);
        };
        let prompt_token_count = protocol_prompt_token_count(
            prepared_generation.inference_request.prompt_token_count(),
        )?;
        let inference_request = prepared_generation.inference_request;
        let generation_start_result = {
            let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut()
            else {
                return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                    reason: "the loaded model was removed before generation started".to_owned(),
                });
            };
            loaded_model
                .engine
                .start_generation(inference_request)
                .await
        };
        let generation_start = match generation_start_result {
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
                return fatal::report_fatal_engine_error(request_id, engine_error, event_writer)
                    .await;
            }
        };

        self.emit_startup_events_and_assemble_generation(
            request_id,
            &generation_command,
            generation_start,
            prompt_token_count,
            prepared_generation.request_output,
            event_writer,
        )
        .await
    }

    /// Validates the command, proves the loaded runtime is autoregressive,
    /// and prepares the generation. `Ok(None)` means the request was already
    /// rejected to the client with its failure event.
    pub(super) async fn prepare_validated_chat_generation<WriteTransport>(
        &self,
        generation_command: &ipc_protocol::ChatGenerationCommand,
        request_id: ipc_protocol::RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<
        Option<PreparedModelGeneration<InferenceRequestOf<Engines>, RequestOutputOf<Engines>>>,
        WorkerRuntimeError,
    >
    where
        WriteTransport: AsyncWrite + Unpin,
    {
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

        let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_ref() else {
            event_writer
                .send_event(&WorkerEvent::Failed {
                    request_id,
                    reason: ChatGenerationFailureReason::invalid_request(
                        "the loaded model does not support chat generation",
                    ),
                })
                .await?;
            return Ok(None);
        };
        match loaded_model
            .processor
            .prepare_chat_generation(generation_command)
        {
            Ok(prepared_generation) => Ok(Some(prepared_generation)),
            Err(failure_reason) => {
                event_writer
                    .send_event(&WorkerEvent::Failed {
                        request_id,
                        reason: failure_reason,
                    })
                    .await?;
                Ok(None)
            }
        }
    }

    /// Emits the post-start events (`astro_ssd_prompt_cache` statistics, expert-memory mode,
    /// first prefill progress) and assembles the active generation state.
    #[allow(clippy::too_many_arguments)]
    async fn emit_startup_events_and_assemble_generation<WriteTransport>(
        &mut self,
        request_id: ipc_protocol::RequestId,
        generation_command: &ipc_protocol::ChatGenerationCommand,
        generation_start: EngineGenerationStart,
        prompt_token_count: u32,
        prepared_request_output: RequestOutputOf<Engines>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<RequestOutputOf<Engines>>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        self.emit_astro_ssd_prompt_cache_stats(event_writer).await?;
        let astro_ssd_prompt_cache_restored_token_count =
            generation_start.astro_ssd_prompt_cache_restored_token_count();
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
                    prompt_layer_progress: None,
                })
                .await?;
        }

        let mut active_engine_generation = ActiveEngineGeneration::new(
            generation_command,
            prompt_token_count,
            astro_ssd_prompt_cache_restored_token_count,
            required_prompt_processing_token_count,
            prepared_request_output,
        );
        active_engine_generation.last_reported_expert_memory_mode = initial_expert_memory_mode;
        active_engine_generation.astro_ssd_prompt_cache_diagnostics = generation_start
            .astro_ssd_prompt_cache_diagnostics()
            .cloned();
        Ok(Some(active_engine_generation))
    }
}

/// Pure: the wire protocol carries the prompt length as u32; a longer prompt
/// is an engine-generation failure, not a silent truncation.
pub(super) fn protocol_prompt_token_count(
    prompt_token_count: usize,
) -> Result<u32, WorkerRuntimeError> {
    u32::try_from(prompt_token_count).map_err(|_| {
        WorkerRuntimeError::InferenceEngineGenerationFailed {
            reason: "chat prompt token count exceeds the protocol range".to_owned(),
        }
    })
}

#[cfg(test)]
#[path = "generation_start_tests.rs"]
mod generation_start_tests;
