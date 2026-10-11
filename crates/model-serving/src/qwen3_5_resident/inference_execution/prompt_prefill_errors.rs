use crate::{InferenceEngineError, Qwen3_5ExecutionError};

use super::engine_request::Qwen3_5PrefillRequestCheckpoint;
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;

pub(super) enum PromptPrefillChunkAttemptError {
    AdaptiveMemoryLimitExceeded {
        reason: String,
    },
    ActiveMemoryLimitExceeded {
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    },
    GraphicsProcessorMemoryExhausted {
        reason: String,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    },
    Engine(InferenceEngineError),
}

impl From<InferenceEngineError> for PromptPrefillChunkAttemptError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for PromptPrefillChunkAttemptError {
    fn from(admission_error: AdaptiveRamGrowthMemoryAdmissionError) -> Self {
        match admission_error {
            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason } => {
                Self::AdaptiveMemoryLimitExceeded { reason }
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                Self::Engine(inference_engine_error)
            }
        }
    }
}

pub(super) fn prefill_execution_error(
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunkAttemptError {
    if let Some((active_memory_bytes, attempted_allocation_bytes, allowed_active_memory_bytes)) =
        qwen3_5_execution_error.active_memory_limit_exceeded_evidence()
    {
        return PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            prefill_request_checkpoint,
        };
    }
    if qwen3_5_execution_error.is_recoverable_graphics_processor_out_of_memory() {
        return PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: qwen3_5_execution_error.to_string(),
            prefill_request_checkpoint,
        };
    }
    // Eager Rust expert streaming should resolve every layer before constructing
    // expert computation. Preserve this defensive translation so a violated
    // route-stability invariant still reaches the bounded checkpoint/reclamation
    // path instead of changing established request-level error classification.
    if matches!(
        &qwen3_5_execution_error,
        Qwen3_5ExecutionError::InvalidInput { description }
            if *description == "paged route replay exceeded the sparse-layer safety bound"
    ) {
        return PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: "paged expert routes could not stabilize under the active memory ceiling"
                .to_owned(),
            prefill_request_checkpoint,
        };
    }
    PromptPrefillChunkAttemptError::Engine(qwen3_5_execution_error.into())
}
