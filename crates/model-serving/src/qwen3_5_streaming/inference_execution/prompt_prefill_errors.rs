use crate::{InferenceEngineError, Qwen3_5ExecutionError};

use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;

pub(super) enum PromptPrefillChunkAttemptError {
    AdaptiveMemoryLimitExceeded {
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
) -> PromptPrefillChunkAttemptError {
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
