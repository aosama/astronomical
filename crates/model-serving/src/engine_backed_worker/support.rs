//! Shared factory, generation state, and typed failures for the engine-backed worker.

use std::time::Instant;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ExpertMemoryMode, ProtocolError, RequestId, WorkerPromptWorkReuse,
};
use thiserror::Error;

use crate::InferenceEngineError;

/// Factory that creates a new processor and engine for a selected model directory.
pub trait ModelFactory<Processor, Engine>: Send + Sync + 'static {
    /// Creates a processor and unloaded engine with the requested output ceiling.
    ///
    /// A failure reason is delivered to the local API caller, so it must be
    /// bounded and must not expose local filesystem paths or native errors.
    fn create(
        &self,
        model_directory: &str,
        max_output_tokens: u32,
    ) -> impl std::future::Future<Output = Result<(Processor, Engine), String>> + Send;

    /// Updates the ceiling used by a future lazy model load.
    fn update_mlx_memory_ceiling_bytes(&mut self, _effective_mlx_memory_ceiling_bytes: u64) {}
}

impl<Processor, Engine> ModelFactory<Processor, Engine> for () {
    async fn create(
        &self,
        _model_directory: &str,
        _max_output_tokens: u32,
    ) -> Result<(Processor, Engine), String> {
        Err("model swapping is unavailable because no model factory was configured".to_owned())
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
}

pub(crate) struct ActiveEngineGeneration<RequestOutput> {
    pub(crate) request_output: RequestOutput,
    pub(crate) generated_token_count: u16,
    pub(crate) reasoning_token_count: u16,
    pub(crate) max_output_tokens: u16,
    pub(crate) next_sequence_number: u16,
    pub(crate) next_tool_call_index: u16,
    pub(crate) prompt_token_count: u32,
    pub(crate) cached_token_count: u32,
    pub(crate) prompt_work_reuse: WorkerPromptWorkReuse,
    pub(crate) prefill_processed_tokens: u32,
    pub(crate) prefill_elapsed_millis: u64,
    pub(crate) generation_started_at: Option<Instant>,
    pub(crate) request_id: RequestId,
    pub(crate) engine_has_finalized_generation: bool,
    pub(crate) has_emitted_tool_call: bool,
    pub(crate) last_reported_expert_memory_mode: Option<ExpertMemoryMode>,
}

impl<RequestOutput> ActiveEngineGeneration<RequestOutput> {
    pub(crate) fn new(
        generation_command: &ChatGenerationCommand,
        prompt_token_count: u32,
        cached_token_count: u32,
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
            cached_token_count,
            prompt_work_reuse: WorkerPromptWorkReuse::default(),
            prefill_processed_tokens: 0,
            prefill_elapsed_millis: 0,
            generation_started_at: None,
            request_id: generation_command.request_id,
            engine_has_finalized_generation: false,
            has_emitted_tool_call: false,
            last_reported_expert_memory_mode: None,
        }
    }
}

pub(crate) fn engine_generation_error(engine_error: InferenceEngineError) -> WorkerRuntimeError {
    WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: engine_error.to_string(),
    }
}
