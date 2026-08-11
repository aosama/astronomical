use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxRuntimeError;

use crate::{InferenceEngineError, Qwen3_5ExecutionError};

use super::engine_request::Qwen3_5PrefillRequestCheckpoint;
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::speculative_prefill::configured_speculative_prefill_failure;

pub(super) enum PromptPrefillChunckAttemptError {
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

impl From<InferenceEngineError> for PromptPrefillChunckAttemptError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for PromptPrefillChunckAttemptError {
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
        | Qwen3_5ExecutionError::PersistentPromptCacheStateBridge(_) => false,
    }
}

pub(super) fn prefill_execution_error(
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunckAttemptError {
    match qwen3_5_execution_error {
        Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        }) => PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            prefill_request_checkpoint,
        },
        Qwen3_5ExecutionError::Runtime(mlx_runtime_error)
            if mlx_runtime_error.is_recoverable_graphics_processor_out_of_memory() =>
        {
            PromptPrefillChunckAttemptError::GraphicsProcessorMemoryExhausted {
                reason: mlx_runtime_error.to_string(),
                prefill_request_checkpoint,
            }
        }
        other_qwen3_5_execution_error => {
            PromptPrefillChunckAttemptError::Engine(other_qwen3_5_execution_error.into())
        }
    }
}

pub(super) fn configured_speculative_prefill_execution_error(
    request_id: RequestId,
    failure_stage: &'static str,
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunckAttemptError {
    // Preserve recoverable capacity errors so the outer prefill loop can restore
    // its checkpoint, reclaim experts, and reduce chunk size. All non-capacity
    // execution failures become fail-closed configured SpecPrefill errors.
    match prefill_execution_error(qwen3_5_execution_error, prefill_request_checkpoint) {
        PromptPrefillChunckAttemptError::Engine(inference_engine_error) => {
            PromptPrefillChunckAttemptError::Engine(configured_speculative_prefill_failure(
                request_id,
                failure_stage,
                inference_engine_error,
            ))
        }
        active_memory_limit_error @ PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
            ..
        } => active_memory_limit_error,
        graphics_processor_memory_error @ PromptPrefillChunckAttemptError::GraphicsProcessorMemoryExhausted {
            ..
        } => graphics_processor_memory_error,
        PromptPrefillChunckAttemptError::AdaptiveMemoryLimitExceeded { reason } => {
            PromptPrefillChunckAttemptError::Engine(configured_speculative_prefill_failure(
                request_id,
                failure_stage,
                reason,
            ))
        }
    }
}
