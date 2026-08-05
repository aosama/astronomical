use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    MlxMemorySnapshotSource, ProtocolWriter, RequestId, WorkerEvent, WorkerMlxMemorySnapshot,
};
use tokio::io::AsyncWrite;

use super::support::{ActiveEngineGeneration, engine_generation_error};
use crate::model_generation_processor::{ModelGenerationOutputError, ModelGenerationProcessor};
use crate::{
    EngineBackedWorker, GenerationFinalization, InferenceEngine, MlxMemoryTelemetry,
    WorkerRuntimeError,
};

impl<Processor, Engine, Factory> EngineBackedWorker<Processor, Engine, Factory>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
{
    pub(crate) async fn emit_model_outputs<WriteTransport>(
        &self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        model_outputs: Vec<ChatGenerationOutput>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if model_outputs.is_empty() {
            return Ok(());
        }

        for model_output in &model_outputs {
            if let ChatGenerationOutput::ToolCall {
                tool_call_index, ..
            } = model_output
            {
                if *tool_call_index != active_generation.next_tool_call_index {
                    return Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                        reason: "structured chat tool-call indexes were not contiguous".to_owned(),
                    });
                }
                active_generation.next_tool_call_index = active_generation
                    .next_tool_call_index
                    .checked_add(1)
                    .ok_or_else(|| WorkerRuntimeError::InferenceEngineGenerationFailed {
                        reason: "structured chat tool-call sequence overflowed".to_owned(),
                    })?;
                active_generation.has_emitted_tool_call = true;
            }
        }

        let output_count = u16::try_from(model_outputs.len()).map_err(|_| {
            WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "batched output count exceeds the u16 range".to_owned(),
            }
        })?;
        event_writer
            .send_event(&WorkerEvent::Output {
                request_id: active_generation.request_id,
                sequence_number: active_generation.next_sequence_number,
                generated_token_count: active_generation.generated_token_count,
                outputs: model_outputs,
                mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                    worker_memory_snapshot(
                        MlxMemorySnapshotSource::DecodeSubmitted,
                        mlx_memory_telemetry,
                    )
                }),
            })
            .await?;
        active_generation.next_sequence_number = active_generation
            .next_sequence_number
            .checked_add(output_count)
            .ok_or_else(|| WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "generation output sequence overflowed".to_owned(),
            })?;
        Ok(())
    }

    pub(crate) async fn cancel_active_engine_request(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, WorkerRuntimeError> {
        let loaded_model = self.loaded_model.as_mut().ok_or_else(|| {
            WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "cancellation was requested before a model was loaded".to_owned(),
            }
        })?;
        loaded_model
            .engine
            .cancel_generation(request_id)
            .await
            .map_err(engine_generation_error)
    }

    pub(crate) async fn emit_generation_finalization<WriteTransport>(
        &self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        generation_finalization: GenerationFinalization,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if !generation_finalization.has_reportable_state() {
            return Ok(());
        }
        let expert_memory_mode = generation_finalization.expert_memory_mode();
        let mlx_memory_snapshot =
            generation_finalization
                .mlx_memory_telemetry()
                .map(|mlx_memory_telemetry| {
                    worker_memory_snapshot(MlxMemorySnapshotSource::Finalized, mlx_memory_telemetry)
                });
        tracing::info!(
            request_id = active_generation.request_id.value(),
            expert_memory_mode = ?expert_memory_mode,
            mlx_active_memory_bytes = ?mlx_memory_snapshot.as_ref().map(|snapshot| snapshot.active_memory_bytes),
            "emitting final generation residency and MLX memory telemetry"
        );
        event_writer
            .send_event(&WorkerEvent::GenerationFinalized {
                request_id: active_generation.request_id,
                expert_memory_mode,
                mlx_memory_snapshot,
            })
            .await?;
        active_generation.last_reported_expert_memory_mode = expert_memory_mode;
        Ok(())
    }

    pub(crate) async fn send_generation_progress<WriteTransport>(
        &self,
        active_generation: &ActiveEngineGeneration<Processor::RequestOutput>,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let elapsed_millis = active_generation
            .generation_started_at
            .map_or(0, |generation_started_at| {
                generation_started_at.elapsed().as_millis() as u64
            });
        event_writer
            .send_event(&WorkerEvent::GenerationProgress {
                request_id: active_generation.request_id,
                generated_token_count: active_generation.generated_token_count,
                maximum_output_tokens: active_generation.max_output_tokens,
                elapsed_millis,
                mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                    worker_memory_snapshot(
                        MlxMemorySnapshotSource::DecodeSubmitted,
                        mlx_memory_telemetry,
                    )
                }),
            })
            .await?;
        Ok(())
    }

    pub(crate) async fn finish_generation<WriteTransport>(
        &mut self,
        mut active_generation: ActiveEngineGeneration<Processor::RequestOutput>,
        completion_reason: ChatGenerationCompletionReason,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveEngineGeneration<Processor::RequestOutput>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let loaded_model = self.loaded_model.as_mut().ok_or_else(|| {
            WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "generation completed after the loaded model was removed".to_owned(),
            }
        })?;
        let final_outputs = loaded_model
            .processor
            .finish_request_output(&mut active_generation.request_output);
        let final_outputs = match final_outputs {
            Ok(final_outputs) => final_outputs,
            Err(ModelGenerationOutputError::MalformedOutput { diagnostic }) => {
                tracing::warn!(
                    request_id = active_generation.request_id.value(),
                    generated_token_count = active_generation.generated_token_count,
                    diagnostic_json = %serde_json::to_string(&diagnostic).unwrap_or_else(|serialization_error| format!(
                        r#"{{"diagnostic_serialization_error":"{}"}}"#,
                        serialization_error
                    )),
                    "malformed model output while finishing generation"
                );
                self.fail_malformed_generation(&mut active_generation, event_writer)
                    .await?;
                return Ok(None);
            }
            Err(ModelGenerationOutputError::Fatal { reason }) => {
                return Err(WorkerRuntimeError::InferenceEngineGenerationFailed { reason });
            }
        };
        self.emit_model_outputs(&mut active_generation, final_outputs, None, event_writer)
            .await?;
        if !active_generation.engine_has_finalized_generation {
            let generation_finalization = self
                .cancel_active_engine_request(active_generation.request_id)
                .await?;
            self.emit_generation_finalization(
                &mut active_generation,
                generation_finalization,
                event_writer,
            )
            .await?;
        }
        let final_reason = if active_generation.has_emitted_tool_call {
            ChatGenerationCompletionReason::ToolCalls
        } else {
            completion_reason
        };
        self.send_completion(&active_generation, final_reason, event_writer)
            .await?;
        Ok(None)
    }

    pub(crate) async fn fail_malformed_generation<WriteTransport>(
        &mut self,
        active_generation: &mut ActiveEngineGeneration<Processor::RequestOutput>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if !active_generation.engine_has_finalized_generation {
            let generation_finalization = self
                .cancel_active_engine_request(active_generation.request_id)
                .await?;
            self.emit_generation_finalization(
                active_generation,
                generation_finalization,
                event_writer,
            )
            .await?;
        }
        event_writer
            .send_event(&WorkerEvent::Failed {
                request_id: active_generation.request_id,
                reason: ChatGenerationFailureReason::MalformedModelOutput,
            })
            .await?;
        Ok(())
    }

    pub(crate) async fn send_invalid_request_failure<WriteTransport>(
        request_id: RequestId,
        reason: String,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::Failed {
                request_id,
                reason: ChatGenerationFailureReason::invalid_request(reason),
            })
            .await?;
        Ok(())
    }

    pub(crate) async fn send_completion<WriteTransport>(
        &self,
        active_generation: &ActiveEngineGeneration<Processor::RequestOutput>,
        completion_reason: ChatGenerationCompletionReason,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::Completed {
                request_id: active_generation.request_id,
                prompt_token_count: active_generation.prompt_token_count,
                generated_token_count: active_generation.generated_token_count,
                reasoning_token_count: active_generation.reasoning_token_count,
                cached_token_count: active_generation.cached_token_count,
                reason: completion_reason,
            })
            .await?;
        self.emit_persistent_prompt_cache_stats(event_writer)
            .await?;
        Ok(())
    }

    /// Emits a `WorkerEvent::PersistentPromptCacheStats` event when the engine
    /// has a persistent prompt cache. Silently skips when the engine returns
    /// `None` (no cache open) or when collection fails.
    pub(crate) async fn emit_persistent_prompt_cache_stats<WriteTransport>(
        &self,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(loaded_model) = self.loaded_model.as_ref() else {
            return Ok(());
        };
        match loaded_model
            .engine
            .collect_persistent_prompt_cache_stats()
            .await
        {
            Ok(Some(persistent_prompt_cache_stats_event)) => {
                event_writer
                    .send_event(&persistent_prompt_cache_stats_event)
                    .await?;
            }
            Ok(None) => {}
            Err(collection_error) => {
                tracing::warn!(
                    "persistent prompt-cache stats collection failed: {collection_error}"
                );
            }
        }
        Ok(())
    }
    pub(crate) async fn emit_mlx_memory_sample<WriteTransport>(
        &self,
        mlx_memory_snapshot_source: MlxMemorySnapshotSource,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let mlx_memory_snapshot = match self.loaded_model.as_ref() {
            Some(loaded_model) => loaded_model
                .engine
                .collect_mlx_memory_telemetry()
                .await
                .map_err(engine_generation_error)?
                .map(|mlx_memory_telemetry| {
                    worker_memory_snapshot(mlx_memory_snapshot_source, mlx_memory_telemetry)
                }),
            None => None,
        };
        let Some(mlx_memory_snapshot) = mlx_memory_snapshot else {
            return Ok(());
        };
        event_writer
            .send_event(&WorkerEvent::MlxMemorySample {
                mlx_memory_snapshot: Some(mlx_memory_snapshot),
            })
            .await?;
        Ok(())
    }
}

pub(crate) fn worker_memory_snapshot(
    mlx_memory_snapshot_source: MlxMemorySnapshotSource,
    mlx_memory_telemetry: MlxMemoryTelemetry,
) -> WorkerMlxMemorySnapshot {
    let mlx_active_memory_breakdown = mlx_memory_telemetry.active_memory_breakdown;
    WorkerMlxMemorySnapshot {
        source: mlx_memory_snapshot_source,
        active_memory_bytes: mlx_memory_telemetry.active_memory_bytes,
        allocator_cache_memory_bytes: mlx_memory_telemetry.allocator_cache_memory_bytes,
        peak_memory_bytes: mlx_memory_telemetry.peak_memory_bytes,
        expert_payload_bytes: mlx_active_memory_breakdown.expert_payload_bytes,
        model_core_payload_bytes: mlx_active_memory_breakdown.model_core_payload_bytes,
        context_state_payload_bytes: mlx_active_memory_breakdown.context_state_payload_bytes,
    }
}
