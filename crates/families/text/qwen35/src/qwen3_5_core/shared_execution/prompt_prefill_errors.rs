//! Why reading one chunk of a prompt failed, for either Qwen3.5 engine
//! (issue #2473): an MLX memory limit or exhaustion the engine can recover
//! from by reading smaller chunks or more layers at a time, or an engine
//! error. The resident engine also refuses a chunk its memory governor
//! will not admit; the streaming engine has no such refusal, so it names
//! `Infallible` and its matches leave that case out.

use crate::{InferenceEngineError, Qwen3_5ExecutionError};

pub enum PromptPrefillChunkAttemptError<AdmissionRejection> {
    /// The memory governor found that this chunk does not fit. The typed
    /// rejection says which shortfall moves the request to the streaming
    /// engine.
    ForwardAdmissionRejected {
        rejection: AdmissionRejection,
        reason: String,
    },
    ActiveMemoryLimitExceeded {
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
    },
    GraphicsProcessorMemoryExhausted {
        reason: String,
    },
    Engine(InferenceEngineError),
}

impl<AdmissionRejection> From<InferenceEngineError>
    for PromptPrefillChunkAttemptError<AdmissionRejection>
{
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

/// Classifies an execution error from reading a chunk: the recoverable MLX
/// memory cases by their evidence, everything else as an engine error.
pub fn prefill_execution_error<AdmissionRejection>(
    qwen3_5_execution_error: Qwen3_5ExecutionError,
) -> PromptPrefillChunkAttemptError<AdmissionRejection> {
    if let Some((active_memory_bytes, attempted_allocation_bytes, allowed_active_memory_bytes)) =
        qwen3_5_execution_error.active_memory_limit_exceeded_evidence()
    {
        return PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        };
    }
    if qwen3_5_execution_error.is_recoverable_graphics_processor_out_of_memory() {
        return PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: qwen3_5_execution_error.to_string(),
        };
    }
    // Eager Rust expert streaming should resolve every layer before constructing
    // expert computation. Preserve this defensive translation so a violated
    // route-stability invariant still reaches the typed capacity classification
    // instead of changing established request-level error handling.
    if matches!(
        &qwen3_5_execution_error,
        Qwen3_5ExecutionError::InvalidInput { description }
            if *description == "paged route replay exceeded the sparse-layer safety bound"
    ) {
        return PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: "paged expert routes could not stabilize under the active memory ceiling"
                .to_owned(),
        };
    }
    PromptPrefillChunkAttemptError::Engine(qwen3_5_execution_error.into())
}
