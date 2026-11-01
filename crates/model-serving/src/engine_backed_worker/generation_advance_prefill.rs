//! The prefill/preparation side of generation advance: folding prefill
//! progress chunks into the active request and emitting the phase and
//! preparation start events. Split from generation_advance so each file stays
//! within the repository's source-size limit.

use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent,
    WorkerPersistentPromptCacheRequestDiagnostics, WorkerPromptProcessingPhase,
    WorkerPromptWorkReuse,
};
use tokio::io::AsyncWrite;

use super::support::{ActiveEngineGeneration, ModelFactory, WorkerRuntimeError};
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ExpertMemoryMode, ImageGenerationEngine, InferenceEngine};

use super::EngineBackedWorker;

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
{
    /// Advances one prefill progress chunk: folds the reuse statistics,
    /// diagnostics, and expert-memory mode into the active request,
    /// accumulates the prefill cursor, and emits the target-phase progress
    /// event while required prompt processing remains. Mirrors the inlined
    /// `PrefillProgress` arm.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn advance_prefill_progress<WriteTransport>(
        &mut self,
        mut active_generation: ActiveEngineGeneration<
            Processor::RequestOutput,
            Processor::InferenceRequest,
        >,
        processed_token_count: u32,
        elapsed_millis: u64,
        forward_prefill_chunk_elapsed_millis: u64,
        completed_prefill_chunk_tokens: u32,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
        expert_residency_telemetry: Option<
            crate::inference_engine::contract::ExpertResidencyTelemetry,
        >,
        expert_memory_mode: Option<ExpertMemoryMode>,
        prompt_work_reuse: WorkerPromptWorkReuse,
        persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<
        Option<ActiveEngineGeneration<Processor::RequestOutput, Processor::InferenceRequest>>,
        WorkerRuntimeError,
    >
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if prompt_work_reuse.target_eligible_token_count > 0 {
            let required_target_prompt_processing_token_count = prompt_work_reuse
                .target_eligible_token_count
                .saturating_sub(prompt_work_reuse.target_restored_token_count);
            active_generation.required_prompt_processing_token_count =
                u32::try_from(required_target_prompt_processing_token_count).unwrap_or(u32::MAX);
        }
        active_generation.prompt_work_reuse = prompt_work_reuse;
        if persistent_prompt_cache_diagnostics.is_some() {
            active_generation.persistent_prompt_cache_diagnostics =
                persistent_prompt_cache_diagnostics;
        }
        if let Some(expert_memory_mode) = expert_memory_mode
            && active_generation.last_reported_expert_memory_mode != Some(expert_memory_mode)
        {
            tracing::info!(
                request_id = active_generation.request_id.value(),
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
                prompt_processing_phase: WorkerPromptProcessingPhase::Target,
                processed_tokens: active_generation
                    .prefill_processed_tokens
                    .min(required_prompt_processing_token_count),
                total_tokens: required_prompt_processing_token_count,
                elapsed_millis: active_generation.prefill_elapsed_millis,
                forward_prefill_chunk_elapsed_millis: Some(forward_prefill_chunk_elapsed_millis),
                completed_prefill_chunk_tokens: Some(completed_prefill_chunk_tokens),
                mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                    super::output::worker_memory_snapshot(
                        MlxMemorySnapshotSource::Prefill,
                        mlx_memory_telemetry,
                    )
                }),
                expert_residency: expert_residency_telemetry
                    .map(super::output::worker_expert_residency_snapshot),
            })
            .await?;
        Ok(Some(active_generation))
    }
}

/// Emits the phase-started prefill progress event. Mirrors the inlined
/// `PromptProcessingPhaseStarted` arm.
pub(super) async fn emit_prompt_processing_phase_started<
    WriteTransport,
    RequestOutput,
    RetryRequest,
>(
    active_generation: &ActiveEngineGeneration<RequestOutput, RetryRequest>,
    prompt_processing_phase: WorkerPromptProcessingPhase,
    total_token_count: u32,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
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
        })
        .await?;
    Ok(())
}

/// Emits the generation preparation started event. Mirrors the inlined
/// `GenerationPreparationStarted` arm.
pub(super) async fn emit_generation_preparation_started<
    WriteTransport,
    RequestOutput,
    RetryRequest,
>(
    active_generation: &ActiveEngineGeneration<RequestOutput, RetryRequest>,
    expert_residency: crate::inference_engine::contract::ExpertResidencyTelemetry,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::GenerationPreparationStarted {
            request_id: active_generation.request_id,
            expert_residency: super::output::worker_expert_residency_snapshot(expert_residency),
            mlx_memory_snapshot: mlx_memory_telemetry.map(|mlx_memory_telemetry| {
                super::output::worker_memory_snapshot(
                    MlxMemorySnapshotSource::DecodeSubmitted,
                    mlx_memory_telemetry,
                )
            }),
        })
        .await?;
    Ok(())
}
