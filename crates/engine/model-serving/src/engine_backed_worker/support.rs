//! Shared factory, generation state, and typed failures for the engine-backed worker.

use std::path::Path;
use std::time::Instant;

use ipc_protocol::{
    ChatGenerationCommand, ExpertMemoryMode, ProtocolError, RequestId,
    WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerModelConfiguration, WorkerPromptWorkReuse,
};
use thiserror::Error;

use crate::{EnginePlacement, ImageGenerationUnavailableEngine, InferenceEngineError};

/// Unloaded runtime selected by the model factory for one typed model configuration.
pub enum ModelFactoryRuntime<
    Processor,
    Engine,
    ImageEngine = ImageGenerationUnavailableEngine,
    EmbeddingEngine = crate::EmbeddingUnavailableEngine,
    DecisionsEngine = crate::DecisionUnavailableEngine,
    TextToSpeechEngine = crate::SpeechUnavailableEngine,
> {
    Autoregressive {
        processor: Processor,
        engine: Engine,
        /// The startup placement this runtime was built under. The worker
        /// adopts it when the runtime loads.
        engine_placement: EnginePlacement,
    },
    Image(ImageEngine),
    Embeddings(EmbeddingEngine),
    Decisions(DecisionsEngine),
    Speech(TextToSpeechEngine),
}

impl<Processor, Engine, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    ModelFactoryRuntime<
        Processor,
        Engine,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
{
    #[must_use]
    pub fn autoregressive(
        processor: Processor,
        engine: Engine,
        engine_placement: EnginePlacement,
    ) -> Self {
        Self::Autoregressive {
            processor,
            engine,
            engine_placement,
        }
    }
}

/// Factory that creates a new processor and engine for a selected model directory.
pub trait ModelFactory<
    Processor,
    Engine,
    ImageEngine = ImageGenerationUnavailableEngine,
    EmbeddingsEngine = crate::EmbeddingUnavailableEngine,
    DecisionsEngine = crate::DecisionUnavailableEngine,
    TextToSpeechEngine = crate::SpeechUnavailableEngine,
>: Send + Sync + 'static
{
    /// Creates one unloaded runtime for the exact tagged model configuration.
    ///
    /// A failure reason is delivered to the local API caller, so it must be
    /// bounded and must not expose local filesystem paths or native errors.
    fn create(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> impl std::future::Future<
        Output = Result<
            ModelFactoryRuntime<
                Processor,
                Engine,
                ImageEngine,
                EmbeddingsEngine,
                DecisionsEngine,
                TextToSpeechEngine,
            >,
            String,
        >,
    > + Send;

    /// Updates the complete process-global limit pair used by a future lazy model load.
    fn update_mlx_memory_limits(
        &mut self,
        _effective_mlx_memory_ceiling_bytes: u64,
        _mlx_buffer_cache_limit_bytes: u64,
    ) {
    }

    /// Records how far the next model may be raised to serve at its floor
    /// (#1902). Factories whose engines refuse a too-low ceiling at load use it
    /// to plan that load at the floor instead; the default ignores it.
    fn update_floor_raise_limit(&mut self, _floor_raise_limit_bytes: u64) {}

    /// Returns the global astro SSD prompt cache root directory, if the factory manages one.
    ///
    /// `None` means the factory does not own an `astro_ssd_prompt_cache` (e.g. the `()` unit factory).
    /// The daemon uses this to compute the target path for `astro_ssd_prompt_cache` clear
    /// operations when no model is currently loaded.
    fn global_astro_ssd_prompt_cache_root_directory(&self) -> Option<&Path> {
        None
    }

    /// Whether worker control operations should emit timing attribution.
    fn performance_attribution_enabled(&self) -> bool {
        false
    }
}

impl<Processor, Engine, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    ModelFactory<
        Processor,
        Engine,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    > for ()
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<
        ModelFactoryRuntime<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        >,
        String,
    > {
        Err("model swapping is unavailable because no model factory was configured".to_owned())
    }
}

pub(crate) enum ActiveWorkerRequest<RequestOutput> {
    Autoregressive(Box<ActiveEngineGeneration<RequestOutput>>),
    Image(ActiveImageGeneration),
    Speech(super::speech::ActiveSpeechGeneration),
    DecisionTraining(super::decision_training::ActiveDecisionTraining),
}

pub(crate) struct ActiveImageGeneration {
    pub(crate) request_id: RequestId,
    pub(crate) total_steps: u16,
    started_at: Instant,
    performance_attribution_enabled: bool,
}

impl ActiveImageGeneration {
    pub(crate) fn new(
        request_id: RequestId,
        total_steps: u16,
        performance_attribution_enabled: bool,
    ) -> Self {
        if performance_attribution_enabled {
            tracing::info!(
                operation = "image_generation",
                phase = "start",
                request_id = request_id.value(),
                "performance attribution operation started"
            );
        }
        Self {
            request_id,
            total_steps,
            started_at: Instant::now(),
            performance_attribution_enabled,
        }
    }

    pub(crate) fn elapsed_millis(&self) -> u64 {
        elapsed_millis_since(self.started_at)
    }
}

impl Drop for ActiveImageGeneration {
    fn drop(&mut self) {
        if self.performance_attribution_enabled {
            tracing::info!(
                operation = "image_generation",
                phase = "end",
                request_id = self.request_id.value(),
                elapsed_millis = self.elapsed_millis(),
                "performance attribution operation completed"
            );
        }
    }
}

/// Errors that prevent the inference worker from serving commands.
#[derive(Debug, Error)]
pub enum WorkerRuntimeError {
    #[error("worker inference engine initialization failed: {reason}")]
    InferenceEngineInitializationFailed { reason: String },
    #[error("worker inference engine generation failed: {reason}")]
    InferenceEngineGenerationFailed { reason: String },
    #[error("worker IPC operation failed")]
    Ipc(#[from] ProtocolError),
    #[error("model swap failed: {model_load_failure_reason}")]
    ModelSwapFailed { model_load_failure_reason: String },
    #[error("persistent astro SSD prompt cache clear failed: {reason}")]
    AstroSsdPromptCacheClearFailed { reason: String },
}

pub(crate) struct ActiveEngineGeneration<RequestOutput> {
    pub(crate) request_output: RequestOutput,
    pub(crate) generated_token_count: u16,
    pub(crate) reasoning_token_count: u16,
    pub(crate) max_output_tokens: u16,
    pub(crate) next_sequence_number: u16,
    pub(crate) next_tool_call_index: u16,
    pub(crate) prompt_token_count: u32,
    pub(crate) astro_ssd_prompt_cache_restored_token_count: u32,
    pub(crate) required_prompt_processing_token_count: u32,
    pub(crate) prompt_work_reuse: WorkerPromptWorkReuse,
    /// What drafting did for this reply, from the engine's finalization;
    /// reported just before completion.
    pub(crate) drafting_statistics: Option<ipc_protocol::RequestDraftingStatistics>,
    pub(crate) prefill_processed_tokens: u32,
    pub(crate) prefill_elapsed_millis: u64,
    pub(crate) generation_started_at: Option<Instant>,
    pub(crate) request_id: RequestId,
    pub(crate) engine_has_finalized_generation: bool,
    pub(crate) has_emitted_tool_call: bool,
    pub(crate) last_reported_expert_memory_mode: Option<ExpertMemoryMode>,
    pub(crate) astro_ssd_prompt_cache_diagnostics:
        Option<WorkerAstroSsdPromptCacheRequestDiagnostics>,
}

impl<RequestOutput> ActiveEngineGeneration<RequestOutput> {
    pub(crate) fn new(
        generation_command: &ChatGenerationCommand,
        prompt_token_count: u32,
        astro_ssd_prompt_cache_restored_token_count: u32,
        required_prompt_processing_token_count: u32,
        request_output: RequestOutput,
    ) -> Self {
        Self {
            request_output,
            generated_token_count: 0,
            reasoning_token_count: 0,
            max_output_tokens: generation_command.settings.max_output_tokens,
            next_sequence_number: 0,
            next_tool_call_index: 0,
            prompt_token_count,
            astro_ssd_prompt_cache_restored_token_count,
            required_prompt_processing_token_count,
            prompt_work_reuse: WorkerPromptWorkReuse::default(),
            drafting_statistics: None,
            prefill_processed_tokens: 0,
            prefill_elapsed_millis: 0,
            generation_started_at: None,
            request_id: generation_command.request_id,
            engine_has_finalized_generation: false,
            has_emitted_tool_call: false,
            last_reported_expert_memory_mode: None,
            astro_ssd_prompt_cache_diagnostics: None,
        }
    }
}

pub(crate) fn engine_generation_error(engine_error: InferenceEngineError) -> WorkerRuntimeError {
    WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: engine_error.to_string(),
    }
}

/// Whole milliseconds since `started_at`, saturating rather than wrapping.
pub(crate) fn elapsed_millis_since(started_at: std::time::Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}
