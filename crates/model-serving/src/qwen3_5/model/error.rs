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
    #[error("persistent prompt-cache disk-store error: {0}")]
    PersistentPromptCache(#[from] crate::PersistentPromptCacheDiskStoreError),
    #[error("persistent prompt-cache state bridge error: {0}")]
    PersistentPromptCacheStateBridge(
        #[from] crate::qwen3_5::decoder::PersistentPromptCacheStateBridgeError,
    ),
    #[error("Qwen3.5 sampled decoding error: {0}")]
    SampledDecoding(#[from] crate::InferenceEngineError),
}

impl Qwen3_5ExecutionError {
    /// Returns allocator-capacity evidence regardless of whether MLX failed in
    /// ordinary model execution or Rust bounded expert loading.
    #[must_use]
    pub fn active_memory_limit_exceeded_evidence(&self) -> Option<(usize, usize, usize)> {
        if let Self::ExpertPaging(
            crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError::MemoryBudget(
                crate::MlxAllocationBudgetError::Rejected {
                    active_memory_bytes,
                    pending_allocation_bytes,
                    active_memory_ceiling_bytes,
                    ..
                },
            ),
        ) = self
        {
            return Some((
                usize::try_from(*active_memory_bytes).unwrap_or(usize::MAX),
                usize::try_from(*pending_allocation_bytes).unwrap_or(usize::MAX),
                usize::try_from(*active_memory_ceiling_bytes).unwrap_or(usize::MAX),
            ));
        }
        match self.underlying_mlx_runtime_error()? {
            MlxRuntimeError::ActiveMemoryLimitExceeded {
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            } => Some((
                *active_memory_bytes,
                *attempted_allocation_bytes,
                *allowed_active_memory_bytes,
            )),
            _ => None,
        }
    }

    #[must_use]
    pub(crate) fn is_recoverable_graphics_processor_out_of_memory(&self) -> bool {
        self.underlying_mlx_runtime_error()
            .is_some_and(MlxRuntimeError::is_recoverable_graphics_processor_out_of_memory)
    }

    fn underlying_mlx_runtime_error(&self) -> Option<&MlxRuntimeError> {
        match self {
            Self::Runtime(mlx_runtime_error)
            | Self::ExpertPaging(
                crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError::NativeRuntime(
                    mlx_runtime_error,
                ),
            ) => Some(mlx_runtime_error),
            _ => None,
        }
    }
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
                crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError::NativeRuntime(
                    MlxRuntimeError::ActiveMemoryLimitExceeded { .. },
                ),
            ) => Self::InvalidRequest {
                reason: "generation cannot fit under the configured MLX memory ceiling".to_owned(),
            },
            Qwen3_5ExecutionError::ExpertPaging(
                crate::qwen3_5_moe::expert_paging::expert_pager::ExpertPagingError::MemoryBudget(
                    memory_budget_error
                    @ crate::MlxAllocationBudgetError::Rejected {
                        ..
                    },
                ),
            ) => Self::InvalidRequest {
                reason: format!(
                    "expert paging memory budget rejected the generation request: {memory_budget_error}"
                ),
            },
            Qwen3_5ExecutionError::InvalidInput { description }
                if description == "paged route replay exceeded the sparse-layer safety bound" =>
            {
                Self::InvalidRequest {
                    reason: "paged expert routes could not stabilize under the configured MLX memory ceiling; raise the memory ceiling or shorten the prompt".to_owned(),
                }
            }
            fatal_qwen3_5_execution_error => Self::Fatal {
                reason: fatal_qwen3_5_execution_error.to_string(),
            },
        }
    }
}
