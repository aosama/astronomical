use std::time::Instant;

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::fatal::report_fatal_engine_error;
use super::support::{ActiveEngineGeneration, ModelFactory, WorkerRuntimeError};
use crate::model_generation_processor::{ModelGenerationOutputError, ModelGenerationProcessor};
use crate::{GeneratedToken, ImageGenerationEngine, InferenceEngine, InferenceEngineError};

use super::{EngineBackedWorker, LoadedRuntime};

impl<Processor, Engine, Factory, ImageEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
{
    pub(crate) async fn advance_generation<WriteTransport>(
        &mut self,
        mut active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if active_generation.generated_token_count >= active_generation.max_output_tokens {
            return self
                .finish_generation(
                    active_generation,
                    ChatGenerationCompletionReason::MaximumOutputTokens,
                    event_writer,
                )
                .await;
        }

        let request_id = active_generation.request_id;
        let generated_token = {
            let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut()
            else {
                return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                    reason: "generation continued after the loaded model was removed".to_owned(),
                });
            };
            match loaded_model.engine.decode_next_token(request_id).await {
                Ok(generated_token) => generated_token,
                Err(InferenceEngineError::InvalidRequest { reason }) => {
                    Self::send_invalid_request_failure(request_id, reason, event_writer).await?;
                    return Ok(None);
                }
                Err(engine_error) => {
                    return report_fatal_engine_error(request_id, engine_error, event_writer).await;
                }
            }
        };
        match generated_token {
            GeneratedToken::TokenId {
                token_id: generated_token_id,
                is_reasoning_token,
                expert_memory_mode,
                mlx_memory_telemetry,
                first_decode_forward_elapsed_millis,
                generation_finalization,
            } => {
                if let Some(elapsed_millis) = first_decode_forward_elapsed_millis {
                    event_writer
                        .send_event(&WorkerEvent::FirstDecodeCompleted {
                            request_id,
                            elapsed_millis,
                        })
                        .await?;
                }
                if let Some(generation_finalization) = generation_finalization {
                    active_generation.engine_has_finalized_generation = true;
                    self.emit_generation_finalization(
                        &mut active_generation,
                        generation_finalization,
                        event_writer,
                    )
                    .await?;
                } else if let Some(expert_memory_mode) = expert_memory_mode
                    && active_generation.last_reported_expert_memory_mode
                        != Some(expert_memory_mode)
                {
                    tracing::info!(
                        request_id = request_id.value(),
                        expert_memory_mode = ?expert_memory_mode,
                        generation_phase = "decode",
                        "emitting expert-memory mode event"
                    );
                    event_writer
                        .send_event(&WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode })
                        .await?;
                    active_generation.last_reported_expert_memory_mode = Some(expert_memory_mode);
                }
                if active_generation.generation_started_at.is_none() {
                    active_generation.generation_started_at = Some(Instant::now());
                }
                active_generation.generated_token_count = active_generation
                    .generated_token_count
                    .checked_add(1)
                    .ok_or_else(|| WorkerRuntimeError::InferenceEngineGenerationFailed {
                        reason: "generated token counter overflowed".to_owned(),
                    })?;
                if is_reasoning_token {
                    active_generation.reasoning_token_count = active_generation
                        .reasoning_token_count
                        .checked_add(1)
                        .ok_or_else(|| WorkerRuntimeError::InferenceEngineGenerationFailed {
                            reason: "reasoning token counter overflowed".to_owned(),
                        })?;
                }
                let (is_end_of_sequence, model_translation) = {
                    let Some(LoadedRuntime::Autoregressive(loaded_model)) =
                        self.loaded_runtime.as_mut()
                    else {
                        return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                            reason: "generation continued after the loaded model was removed"
                                .to_owned(),
                        });
                    };
                    (
                        loaded_model
                            .processor
                            .is_end_of_sequence_token(generated_token_id),
                        loaded_model.processor.translate_generated_token(
                            &mut active_generation.request_output,
                            generated_token_id,
                        ),
                    )
                };
                let model_translation = match model_translation {
                    Ok(model_translation) => model_translation,
                    Err(ModelGenerationOutputError::MalformedOutput { diagnostic }) => {
                        tracing::warn!(
                            request_id = active_generation.request_id.value(),
                            generated_token_count = active_generation.generated_token_count,
                            diagnostic_json = %serde_json::to_string(&diagnostic).unwrap_or_else(|serialization_error| format!(
                                r#"{{"diagnostic_serialization_error":"{}"}}"#,
                                serialization_error
                            )),
                            "malformed model output while translating generated token"
                        );
                        self.send_generation_progress(&active_generation, None, event_writer)
                            .await?;
                        self.fail_malformed_generation(&mut active_generation, event_writer)
                            .await?;
                        return Ok(None);
                    }
                    Err(ModelGenerationOutputError::Fatal { reason }) => {
                        return Err(WorkerRuntimeError::InferenceEngineGenerationFailed { reason });
                    }
                };
                let (model_outputs, model_feedback_token_ids) = model_translation.into_parts();
                if model_outputs.is_empty() {
                    self.send_generation_progress(
                        &active_generation,
                        mlx_memory_telemetry,
                        event_writer,
                    )
                    .await?;
                }
                self.emit_model_outputs(
                    &mut active_generation,
                    model_outputs,
                    mlx_memory_telemetry,
                    event_writer,
                )
                .await?;
                if !model_feedback_token_ids.is_empty() {
                    let Some(LoadedRuntime::Autoregressive(loaded_model)) =
                        self.loaded_runtime.as_mut()
                    else {
                        return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                            reason: "generation continued after the loaded model was removed"
                                .to_owned(),
                        });
                    };
                    match loaded_model
                        .engine
                        .inject_input_tokens(active_generation.request_id, model_feedback_token_ids)
                        .await
                    {
                        Ok(()) => {}
                        Err(InferenceEngineError::InvalidRequest { reason }) => {
                            Self::send_invalid_request_failure(request_id, reason, event_writer)
                                .await?;
                            return Ok(None);
                        }
                        Err(engine_error) => {
                            return report_fatal_engine_error(
                                request_id,
                                engine_error,
                                event_writer,
                            )
                            .await;
                        }
                    }
                }
                if is_end_of_sequence {
                    return self
                        .finish_generation(
                            active_generation,
                            ChatGenerationCompletionReason::EndOfSequence,
                            event_writer,
                        )
                        .await;
                }
                if active_generation.generated_token_count >= active_generation.max_output_tokens {
                    return self
                        .finish_generation(
                            active_generation,
                            ChatGenerationCompletionReason::MaximumOutputTokens,
                            event_writer,
                        )
                        .await;
                }
                Ok(Some(active_generation))
            }
            GeneratedToken::EndOfSequence => {
                self.finish_generation(
                    active_generation,
                    ChatGenerationCompletionReason::EndOfSequence,
                    event_writer,
                )
                .await
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                forward_prefill_chunk_elapsed_millis,
                completed_prefill_chunk_tokens,
                mlx_memory_telemetry,
                expert_residency_telemetry,
                speculative_prefill_draft_memory_telemetry,
                expert_memory_mode,
                prompt_work_reuse,
                persistent_prompt_cache_diagnostics,
            } => {
                if prompt_work_reuse.target_eligible_token_count > 0 {
                    let required_target_prompt_processing_token_count = prompt_work_reuse
                        .target_eligible_token_count
                        .saturating_sub(prompt_work_reuse.target_restored_token_count);
                    active_generation.required_prompt_processing_token_count =
                        u32::try_from(required_target_prompt_processing_token_count)
                            .unwrap_or(u32::MAX);
                }
                active_generation.prompt_work_reuse = prompt_work_reuse;
                if persistent_prompt_cache_diagnostics.is_some() {
                    active_generation.persistent_prompt_cache_diagnostics =
                        persistent_prompt_cache_diagnostics;
                }
                if let Some(expert_memory_mode) = expert_memory_mode
                    && active_generation.last_reported_expert_memory_mode
                        != Some(expert_memory_mode)
                {
                    tracing::info!(
                        request_id = request_id.value(),
                        expert_memory_mode = ?expert_memory_mode,
                        generation_phase = "prefill",
                        "emitting expert-memory mode event"
                    );
                    event_writer
                        .send_event(&WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode })
                        .await?;
                    active_generation.last_reported_expert_memory_mode = Some(expert_memory_mode);
                }
                active_generation.prefill_processed_tokens = active_generation
                    .prefill_processed_tokens
                    .saturating_add(processed_token_count);
                active_generation.prefill_elapsed_millis = active_generation
                    .prefill_elapsed_millis
                    .saturating_add(elapsed_millis);
                let required_prompt_processing_token_count =
                    active_generation.required_prompt_processing_token_count;
                if required_prompt_processing_token_count == 0 {
                    return Ok(Some(active_generation));
                }
                event_writer
                    .send_event(&WorkerEvent::PrefillProgress {
                        request_id: active_generation.request_id,
                        prompt_processing_phase:
                            astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target,
                        processed_tokens: active_generation
                            .prefill_processed_tokens
                            .min(required_prompt_processing_token_count),
                        total_tokens: required_prompt_processing_token_count,
                        elapsed_millis: active_generation.prefill_elapsed_millis,
                        forward_prefill_chunk_elapsed_millis: Some(
                            forward_prefill_chunk_elapsed_millis,
                        ),
                        completed_prefill_chunk_tokens: Some(completed_prefill_chunk_tokens),
                        mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                            super::output::worker_memory_snapshot(
                                MlxMemorySnapshotSource::Prefill,
                                mlx_memory_telemetry,
                            )
                        }),
                        expert_residency: expert_residency_telemetry
                            .map(super::output::worker_expert_residency_snapshot),
                        speculative_prefill_draft_memory_snapshot:
                            speculative_prefill_draft_memory_telemetry.map(
                                |speculative_prefill_draft_memory_telemetry| {
                                    super::output::worker_memory_snapshot(
                                        MlxMemorySnapshotSource::SpeculativePrefillDraftScoring,
                                        speculative_prefill_draft_memory_telemetry,
                                    )
                                },
                            ),
                    })
                    .await?;
                Ok(Some(active_generation))
            }
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                total_token_count,
            } => {
                event_writer
                    .send_event(&WorkerEvent::PrefillProgress {
                        request_id: active_generation.request_id,
                        prompt_processing_phase,
                        processed_tokens: 0,
                        total_tokens: total_token_count,
                        elapsed_millis: 0,
                        forward_prefill_chunk_elapsed_millis: None,
                        completed_prefill_chunk_tokens: None,
                        mlx_memory_snapshot: None,
                        expert_residency: None,
                        speculative_prefill_draft_memory_snapshot: None,
                    })
                    .await?;
                Ok(Some(active_generation))
            }
            GeneratedToken::GenerationPreparationStarted {
                total_layer_count,
                resident_expert_count,
                resident_expert_payload_bytes,
                mlx_memory_telemetry,
            } => {
                event_writer
                    .send_event(&WorkerEvent::GenerationPreparationStarted {
                        request_id: active_generation.request_id,
                        total_layer_count,
                        resident_expert_count,
                        resident_expert_payload_bytes,
                        mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                            super::output::worker_memory_snapshot(
                                MlxMemorySnapshotSource::DecodeSubmitted,
                                mlx_memory_telemetry,
                            )
                        }),
                    })
                    .await?;
                Ok(Some(active_generation))
            }
        }
    }
}
