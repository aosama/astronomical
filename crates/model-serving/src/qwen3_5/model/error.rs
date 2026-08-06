use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

use crate::{ArtifactValidationError, InferenceEngineError};

/// A cause-preserving failure while loading or executing the native Qwen3.5 model.
#[derive(Debug, Error)]
pub enum Qwen3_5ExecutionError {
    #[error("validated Qwen3.5 artifact ownership transfer failed")]
    Artifact(#[from] ArtifactValidationError),
    #[error("direct MLX execution failed: {0}")]
    Runtime(#[from] MlxRuntimeError),
    #[error("required Qwen3.5 tensor '{tensor_name}' is not bound")]
    MissingTensor { tensor_name: String },
    #[error("Qwen3.5 tensor '{tensor_name}' is invalid: {description}")]
    InvalidTensor {
        tensor_name: String,
        description: &'static str,
    },
    #[error("Qwen3.5 quantized module '{module_name}' is not in the validated bit map")]
    MissingQuantization { module_name: String },
    #[error("validated Qwen3.5 tensor '{tensor_name}' was not assigned to a typed model weight")]
    UnassignedTensor { tensor_name: String },
    #[error(
        "typed Qwen3.5 weight traversal produced {actual_tensor_count} tensors, expected {expected_tensor_count}"
    )]
    TypedTensorCountMismatch {
        actual_tensor_count: usize,
        expected_tensor_count: usize,
    },
    #[error("typed Qwen3.5 weights are missing decoder layer {layer_index}")]
    MissingDecoderLayerWeights { layer_index: usize },
    #[error(
        "bound Qwen3.5 tensor payload is {actual_payload_bytes} bytes, expected {expected_payload_bytes} bytes"
    )]
    TensorPayloadMismatch {
        actual_payload_bytes: u64,
        expected_payload_bytes: u64,
    },
    #[error("invalid Qwen3.5 model input: {description}")]
    InvalidInput { description: &'static str },
    #[error("invalid Qwen3.5 decoder-cache layout: {description}")]
    InvalidDecoderCacheLayout { description: String },
    #[error(
        "request decoder state has {actual_decoder_layer_count} decoder layers, expected {expected_decoder_layer_count}"
    )]
    DecoderLayerCountMismatch {
        actual_decoder_layer_count: usize,
        expected_decoder_layer_count: usize,
    },
    #[error("invalid Qwen3.5 request decoder state for decoder layer {layer_index}: {description}")]
    InvalidRequestDecoderState {
        layer_index: usize,
        description: &'static str,
    },
    #[error("expert paging error: {0}")]
    ExpertPaging(#[from] crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError),
}

pub(super) fn invalid_request_decoder_state(
    layer_index: usize,
    description: &'static str,
) -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidRequestDecoderState {
        layer_index,
        description,
    }
}

impl From<Qwen3_5ExecutionError> for InferenceEngineError {
    fn from(qwen3_5_execution_error: Qwen3_5ExecutionError) -> Self {
        match qwen3_5_execution_error {
            Qwen3_5ExecutionError::Runtime(
                MlxRuntimeError::ActiveMemoryLimitExceeded { .. },
            ) => Self::InvalidRequest {
                reason: "generation cannot fit under the configured MLX memory ceiling".to_owned(),
            },
            Qwen3_5ExecutionError::ExpertPaging(
                crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError::MemoryBudget(
                    memory_budget_error
                    @ crate::expert_paging::memory_budget::MemoryBudgetError::BudgetExceeded {
                        ..
                    },
                ),
            ) => Self::InvalidRequest {
                reason: format!(
                    "expert paging memory budget rejected the generation request: {memory_budget_error}"
                ),
            },
            fatal_qwen3_5_execution_error => Self::Fatal {
                reason: fatal_qwen3_5_execution_error.to_string(),
            },
        }
    }
}
