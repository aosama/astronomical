use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxRuntimeError;

use crate::{InferenceEngineError, Qwen3_5ExecutionError};

use super::engine_request::Qwen3_5PrefillRequestCheckpoint;
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::speculative_prefill::configured_speculative_prefill_failure;

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

pub(super) fn terminal_optional_prefill_error_is_fallback(
    qwen3_5_execution_error: &Qwen3_5ExecutionError,
) -> bool {
    if qwen3_5_execution_error.is_recoverable_graphics_processor_out_of_memory() {
        return false;
    }
    match qwen3_5_execution_error {
        Qwen3_5ExecutionError::Runtime(mlx_runtime_error) => {
            matches!(mlx_runtime_error, MlxRuntimeError::RuntimeOperation { .. })
                && !mlx_runtime_error.is_recoverable_graphics_processor_out_of_memory()
        }
        Qwen3_5ExecutionError::ExpertPaging(_) => true,
        Qwen3_5ExecutionError::Artifact(_)
        | Qwen3_5ExecutionError::MissingTensor { .. }
        | Qwen3_5ExecutionError::InvalidTensor { .. }
        | Qwen3_5ExecutionError::MissingQuantization { .. }
        | Qwen3_5ExecutionError::UnassignedTensor { .. }
        | Qwen3_5ExecutionError::TypedTensorCountMismatch { .. }
        | Qwen3_5ExecutionError::MissingDecoderLayerWeights { .. }
        | Qwen3_5ExecutionError::TensorPayloadMismatch { .. }
        | Qwen3_5ExecutionError::InvalidInput { .. }
        | Qwen3_5ExecutionError::InvalidDecoderCacheLayout { .. }
        | Qwen3_5ExecutionError::DecoderLayerCountMismatch { .. }
        | Qwen3_5ExecutionError::InvalidRequestDecoderState { .. }
        | Qwen3_5ExecutionError::PersistentPromptCache(_)
        | Qwen3_5ExecutionError::PersistentPromptCacheStateBridge(_)
        | Qwen3_5ExecutionError::SampledDecoding(_) => false,
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

pub(super) fn configured_speculative_prefill_execution_error(
    request_id: RequestId,
    failure_stage: &'static str,
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunkAttemptError {
    // Preserve recoverable capacity errors so the outer prefill loop can restore
    // its checkpoint, reclaim experts, and retry the unchanged chunk once. Non-capacity
    // execution failures become fail-closed configured SpecPrefill errors.
    match prefill_execution_error(qwen3_5_execution_error, prefill_request_checkpoint) {
        PromptPrefillChunkAttemptError::Engine(inference_engine_error) => {
            PromptPrefillChunkAttemptError::Engine(configured_speculative_prefill_failure(
                request_id,
                failure_stage,
                inference_engine_error,
            ))
        }
        active_memory_limit_error @ PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            ..
        } => active_memory_limit_error,
        graphics_processor_memory_error @ PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            ..
        } => graphics_processor_memory_error,
        PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded { reason } => {
            PromptPrefillChunkAttemptError::Engine(configured_speculative_prefill_failure(
                request_id,
                failure_stage,
                reason,
            ))
        }
    }
}
