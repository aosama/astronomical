use std::time::Instant;

use ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationOutput, ProtocolWriter, RequestId, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::fatal;
use super::support::{ActiveEngineGeneration, ModelFactory, WorkerRuntimeError};
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::model_generation_processor::{ModelGenerationOutputError, ModelGenerationProcessor};
use crate::{
    ExpertMemoryMode, GeneratedToken, ImageGenerationEngine, InferenceEngine, InferenceEngineError,
};

use super::{EngineBackedWorker, LoadedRuntime};

/// The outcome of translating one decoded token: either a usable translation
/// split into public outputs and model-visible feedback tokens, or the
/// malformed-output path already handled (progress sent, generation failed).
enum DecodedTokenTranslation {
    Translated {
        is_end_of_sequence: bool,
        model_outputs: Vec<ChatGenerationOutput>,
        model_feedback_token_ids: Vec<u32>,
    },
    MalformedOutputHandled,
}

/// Increments the generation and reasoning token counters after one decoded
/// token. Overflow on either counter is a generation failure, with the same
/// reason strings the inline code carried.
fn increment_decode_token_counters<RequestOutput>(
    active_generation: &mut ActiveEngineGeneration<RequestOutput>,
    is_reasoning_token: bool,
) -> Result<(), WorkerRuntimeError> {
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
    Ok(())
}

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    EngineBackedWorker<
        Processor,
        Engine,
        Factory,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        > + Send
        + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
    TextToSpeechEngine: crate::SpeechEngine,
{
    pub(crate) async fn advance_generation<WriteTransport>(
        &mut self,
        active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
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
        let decode_result = match self.decode_next_generated_token(request_id).await {
            Ok(engine_decode_result) => engine_decode_result,
            Err(runtime_error) => return Err(runtime_error),
        };
        let generated_token = match decode_result {
            Ok(generated_token) => generated_token,
            Err(InferenceEngineError::InvalidRequest { reason }) => {
                Self::send_invalid_request_failure(request_id, reason, event_writer).await?;
                return Ok(None);
            }
            Err(engine_error) => {
                return fatal::report_fatal_engine_error(request_id, engine_error, event_writer)
                    .await;
            }
        };
        self.advance_generated_token(active_generation, generated_token, request_id, event_writer)
            .await
    }

    /// Decodes the next token through the loaded autoregressive engine.
    ///
    /// Two error layers, exactly as the inlined block had them: the outer
    /// `WorkerRuntimeError` is the no-loaded-model condition (an early return
    /// from the whole advance), while the inner `InferenceEngineError` flows
    /// into the caller's decode-error dispatch.
    async fn decode_next_generated_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<Result<GeneratedToken, InferenceEngineError>, WorkerRuntimeError> {
        let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut() else {
            return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "generation continued after the loaded model was removed".to_owned(),
            });
        };
        Ok(loaded_model.engine.decode_next_token(request_id).await)
    }

    /// Advances the active generation by one decoded engine token, dispatching
    /// on the token kind: decode steps and end of sequence here, and every
    /// token that belongs to reading the prompt (prefill progress chunks and
    /// the phase/preparation start notifications) in
    /// `advance_prompt_phase_token`. The body of each arm lives in its own
    /// unit; this is the dispatch the inlined advance carried as its final
    /// `match`.
    async fn advance_generated_token<WriteTransport>(
        &mut self,
        active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
        generated_token: GeneratedToken,
        request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match generated_token {
            GeneratedToken::TokenId {
                token_id: generated_token_id,
                is_reasoning_token,
                expert_memory_mode,
                mlx_memory_telemetry,
                first_decode_forward_elapsed_millis,
                generation_finalization,
            } => {
                self.advance_decoded_token(
                    active_generation,
                    generated_token_id,
                    is_reasoning_token,
                    expert_memory_mode,
                    mlx_memory_telemetry,
                    first_decode_forward_elapsed_millis,
                    generation_finalization,
                    request_id,
                    event_writer,
                )
                .await
            }
            GeneratedToken::EndOfSequence => {
                self.finish_generation(
                    active_generation,
                    ChatGenerationCompletionReason::EndOfSequence,
                    event_writer,
                )
                .await
            }
            // The three prompt-phase variants are named, not caught by a
            // wildcard, so a new `GeneratedToken` variant fails to compile here
            // until someone decides which handler it belongs to.
            prompt_phase_token @ (GeneratedToken::PrefillProgress { .. }
            | GeneratedToken::PromptProcessingPhaseStarted { .. }
            | GeneratedToken::GenerationPreparationStarted { .. }) => {
                self.advance_prompt_phase_token(active_generation, prompt_phase_token, event_writer)
                    .await
            }
        }
    }

    /// Advances the active generation by one token the engine sends while it
    /// reads the prompt or prepares to decode: a prefill progress chunk, or
    /// the start of a prompt phase or of generation preparation. Decode tokens
    /// are routed away before this point; one arriving here means the
    /// dispatcher and this handler disagree, which is reported as a failed
    /// generation rather than guessed at.
    async fn advance_prompt_phase_token<WriteTransport>(
        &mut self,
        active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
        prompt_phase_token: GeneratedToken,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match prompt_phase_token {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                forward_prefill_chunk_elapsed_millis,
                completed_prefill_chunk_tokens,
                mlx_memory_telemetry,
                expert_residency_telemetry,
                expert_memory_mode,
                prompt_work_reuse,
                astro_ssd_prompt_cache_diagnostics,
                neural_engine_chunk,
                prompt_layer_progress,
            } => {
                self.advance_prefill_progress(
                    active_generation,
                    processed_token_count,
                    elapsed_millis,
                    forward_prefill_chunk_elapsed_millis,
                    completed_prefill_chunk_tokens,
                    mlx_memory_telemetry,
                    expert_residency_telemetry,
                    expert_memory_mode,
                    prompt_work_reuse,
                    astro_ssd_prompt_cache_diagnostics,
                    neural_engine_chunk,
                    prompt_layer_progress,
                    event_writer,
                )
                .await
            }
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                total_token_count,
            } => {
                super::generation_advance_prefill::emit_prompt_processing_phase_started(
                    &active_generation,
                    prompt_processing_phase,
                    total_token_count,
                    event_writer,
                )
                .await?;
                Ok(Some(active_generation))
            }
            GeneratedToken::GenerationPreparationStarted {
                expert_residency,
                mlx_memory_telemetry,
            } => {
                super::generation_advance_prefill::emit_generation_preparation_started(
                    &active_generation,
                    expert_residency,
                    mlx_memory_telemetry,
                    event_writer,
                )
                .await?;
                Ok(Some(active_generation))
            }
            // `advance_generated_token` never routes these two variants here;
            // an impossible state is reported as a failed generation, the way
            // the rest of this worker reports one, rather than a panic.
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => {
                Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                    reason: "a decode token reached the prompt-phase handler".to_owned(),
                })
            }
        }
    }

    /// Advances one decoded token: records telemetry events, increments the
    /// counters, translates the token through the processor, emits the public
    /// outputs, injects any model feedback tokens, and finishes on end of
    /// sequence or the token budget. Mirrors the inlined `TokenId` arm.
    #[allow(clippy::too_many_arguments)]
    async fn advance_decoded_token<WriteTransport>(
        &mut self,
        mut active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
        generated_token_id: u32,
        is_reasoning_token: bool,
        expert_memory_mode: Option<ExpertMemoryMode>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        first_decode_forward_elapsed_millis: Option<u64>,
        generation_finalization: Option<Box<crate::GenerationFinalization>>,
        request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        self.note_decode_step_telemetry(
            &mut active_generation,
            first_decode_forward_elapsed_millis,
            generation_finalization,
            expert_memory_mode,
            request_id,
            event_writer,
        )
        .await?;
        increment_decode_token_counters(&mut active_generation, is_reasoning_token)?;
        let token_translation = self
            .translate_decoded_token(&mut active_generation, generated_token_id, event_writer)
            .await?;
        let DecodedTokenTranslation::Translated {
            is_end_of_sequence,
            model_outputs,
            model_feedback_token_ids,
        } = token_translation
        else {
            // The malformed-output path already sent its progress and failure
            // events inside translate_decoded_token.
            return Ok(None);
        };
        if model_outputs.is_empty() {
            self.send_generation_progress(&active_generation, mlx_memory_telemetry, event_writer)
                .await?;
        }
        self.emit_model_outputs(
            &mut active_generation,
            model_outputs,
            mlx_memory_telemetry,
            event_writer,
        )
        .await?;
        if !model_feedback_token_ids.is_empty()
            && self
                .inject_model_feedback_tokens(
                    &mut active_generation,
                    request_id,
                    model_feedback_token_ids,
                    event_writer,
                )
                .await?
                .is_none()
        {
            return Ok(None);
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

    /// Emits the exactly-once first-decode event, then either the engine's
    /// generation finalization or a changed expert-memory-mode event. Mirrors
    /// the telemetry head of the inlined `TokenId` arm.
    #[allow(clippy::too_many_arguments)]
    async fn note_decode_step_telemetry<WriteTransport>(
        &mut self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        first_decode_forward_elapsed_millis: Option<u64>,
        generation_finalization: Option<Box<crate::GenerationFinalization>>,
        expert_memory_mode: Option<ExpertMemoryMode>,
        request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if let Some(elapsed_millis) = first_decode_forward_elapsed_millis {
            event_writer
                .send_event(&WorkerEvent::FirstDecodeCompleted {
                    request_id,
                    elapsed_millis,
                })
                .await?;
        }
        if let Some(generation_finalization) = generation_finalization {
            let generation_finalization = *generation_finalization;
            active_generation.engine_has_finalized_generation = true;
            self.emit_generation_finalization(
                active_generation,
                generation_finalization,
                event_writer,
            )
            .await?;
        } else if let Some(expert_memory_mode) = expert_memory_mode
            && active_generation.last_reported_expert_memory_mode != Some(expert_memory_mode)
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
        Ok(())
    }

    /// Translates one decoded token through the loaded processor: checks the
    /// end-of-sequence condition, translates, and either splits the
    /// translation into outputs and feedback tokens or fully handles a
    /// malformed output (warn log, progress event, generation failure).
    /// Mirrors the translation block of the inlined `TokenId` arm.
    async fn translate_decoded_token<WriteTransport>(
        &mut self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        generated_token_id: u32,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<DecodedTokenTranslation, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let (is_end_of_sequence, model_translation) = {
            let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut()
            else {
                return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                    reason: "generation continued after the loaded model was removed".to_owned(),
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
                self.send_generation_progress(active_generation, None, event_writer)
                    .await?;
                self.fail_malformed_generation(active_generation, event_writer)
                    .await?;
                return Ok(DecodedTokenTranslation::MalformedOutputHandled);
            }
            Err(ModelGenerationOutputError::Fatal { reason }) => {
                return Err(WorkerRuntimeError::InferenceEngineGenerationFailed { reason });
            }
        };
        let (model_outputs, model_feedback_token_ids) = model_translation.into_parts();
        Ok(DecodedTokenTranslation::Translated {
            is_end_of_sequence,
            model_outputs,
            model_feedback_token_ids,
        })
    }

    /// Injects model-visible feedback tokens (tool-call results) back into the
    /// engine. Mirrors the feedback block of the inlined `TokenId` arm,
    /// including the invalid-request and fatal dispatch on injection errors:
    /// `Ok(None)` means the generation is over (rejected or fatally reported).
    async fn inject_model_feedback_tokens<WriteTransport>(
        &mut self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        request_id: RequestId,
        model_feedback_token_ids: Vec<u32>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<()>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut() else {
            return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "generation continued after the loaded model was removed".to_owned(),
            });
        };
        match loaded_model
            .engine
            .inject_input_tokens(active_generation.request_id, model_feedback_token_ids)
            .await
        {
            Ok(()) => Ok(Some(())),
            Err(InferenceEngineError::InvalidRequest { reason }) => {
                Self::send_invalid_request_failure(request_id, reason, event_writer).await?;
                Ok(None)
            }
            Err(engine_error) => {
                fatal::report_fatal_engine_error(request_id, engine_error, event_writer).await
            }
        }
    }
}
