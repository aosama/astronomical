use runtime_integration::MlxRuntimeError;
use thiserror::Error;

use crate::{ArtifactValidationError, InferenceEngineError};

/// A cause-preserving failure while loading or executing a text model.
#[derive(Debug, Error)]
pub enum TextExecutionError {
    #[error("validated artifact ownership transfer failed")]
    Artifact(#[from] ArtifactValidationError),
    #[error("direct MLX execution failed: {0}")]
    Runtime(#[from] MlxRuntimeError),
    #[error("required tensor '{tensor_name}' is not bound")]
    MissingTensor { tensor_name: String },
    #[error("tensor '{tensor_name}' is invalid: {description}")]
    InvalidTensor {
        tensor_name: String,
        description: &'static str,
    },
    #[error("quantized module '{module_name}' is not in the validated bit map")]
    MissingQuantization { module_name: String },
    #[error("validated tensor '{tensor_name}' was not assigned to a typed model weight")]
    UnassignedTensor { tensor_name: String },
    #[error(
        "typed weight traversal produced {actual_tensor_count} tensors, expected {expected_tensor_count}"
    )]
    TypedTensorCountMismatch {
        actual_tensor_count: usize,
        expected_tensor_count: usize,
    },
    #[error("typed weights are missing decoder layer {layer_index}")]
    MissingDecoderLayerWeights { layer_index: usize },
    #[error(
        "bound tensor payload is {actual_payload_bytes} bytes, expected {expected_payload_bytes} bytes"
    )]
    TensorPayloadMismatch {
        actual_payload_bytes: u64,
        expected_payload_bytes: u64,
    },
    #[error("invalid model input: {description}")]
    InvalidInput { description: &'static str },
    #[error("invalid astro decoder state cache layout: {description}")]
    InvalidAstroDecoderStateCacheLayout { description: String },
    #[error(
        "request decoder state has {actual_decoder_layer_count} decoder layers, expected {expected_decoder_layer_count}"
    )]
    DecoderLayerCountMismatch {
        actual_decoder_layer_count: usize,
        expected_decoder_layer_count: usize,
    },
    #[error("invalid request decoder state for decoder layer {layer_index}: {description}")]
    InvalidRequestDecoderState {
        layer_index: usize,
        description: &'static str,
    },
    #[error("expert payload memory admission rejected the operation: {0}")]
    MemoryAdmissionRejected(#[from] crate::MlxAllocationAdmissionError),
    #[error("expert source failure: {description}")]
    ExpertSourceFailure { description: String },
    #[error("validated expert layer plan construction failed: {description}")]
    ExpertLayerPlan { description: String },
    #[error("persistent astro SSD prompt cache disk-store error: {0}")]
    AstroSsdPromptCache(#[from] crate::AstroSsdPromptCacheDiskStoreError),
    #[error("persistent astro SSD prompt cache state bridge error: {0}")]
    AstroSsdPromptCacheStateBridge(
        #[from] crate::decoder_state::AstroSsdPromptCacheStateBridgeError,
    ),
    #[error("sampled decoding error: {0}")]
    SampledDecoding(#[from] crate::InferenceEngineError),
    /// A family reads embedding rows from disk its own way, so its failure
    /// arrives by its rendering.
    #[error("on-disk embedding row read failed: {description}")]
    EmbeddingRowRead { description: String },
    /// A forward failure of a text model whose own error has no counterpart
    /// here, carried by its rendering so the engine reports it as is.
    #[error("text model forward failed: {description}")]
    TextModelForward { description: String },
    #[error("{description}")]
    WeightSourceDescriptor {
        description: &'static str,
        #[source]
        source: std::io::Error,
    },
}

impl TextExecutionError {
    /// Returns allocator-capacity evidence regardless of whether MLX failed in
    /// ordinary model execution or Rust bounded expert loading.
    #[must_use]
    pub fn active_memory_limit_exceeded_evidence(&self) -> Option<(usize, usize, usize)> {
        if let Self::MemoryAdmissionRejected(crate::MlxAllocationAdmissionError::Rejected {
            active_memory_bytes,
            pending_allocation_bytes,
            active_memory_ceiling_bytes,
            ..
        }) = self
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
    pub fn is_recoverable_graphics_processor_out_of_memory(&self) -> bool {
        self.underlying_mlx_runtime_error()
            .is_some_and(MlxRuntimeError::is_recoverable_graphics_processor_out_of_memory)
    }

    fn underlying_mlx_runtime_error(&self) -> Option<&MlxRuntimeError> {
        match self {
            Self::Runtime(mlx_runtime_error) => Some(mlx_runtime_error),
            _ => None,
        }
    }
}

pub fn invalid_request_decoder_state(
    layer_index: usize,
    description: &'static str,
) -> TextExecutionError {
    TextExecutionError::InvalidRequestDecoderState {
        layer_index,
        description,
    }
}

impl From<TextExecutionError> for InferenceEngineError {
    fn from(text_execution_error: TextExecutionError) -> Self {
        match text_execution_error {
            TextExecutionError::Runtime(
                MlxRuntimeError::ActiveMemoryLimitExceeded { .. },
            ) => Self::InvalidRequest {
                reason: "generation cannot fit under the configured MLX memory ceiling".to_owned(),
            },
            TextExecutionError::MemoryAdmissionRejected(
                memory_admission_error @ crate::MlxAllocationAdmissionError::Rejected {
                    ..
                },
            ) => Self::InvalidRequest {
                reason: format!(
                    "expert payload memory admission rejected the generation request: {memory_admission_error}"
                ),
            },
            TextExecutionError::InvalidInput {
                description: "paged route replay exceeded the sparse-layer safety bound",
            } => {
                Self::InvalidRequest {
                    reason: "paged expert routes could not stabilize under the configured MLX memory ceiling; raise the memory ceiling or shorten the prompt".to_owned(),
                }
            }
            fatal_text_execution_error => Self::Fatal {
                reason: fatal_text_execution_error.to_string(),
            },
        }
    }
}

#[cfg(feature = "direct-mlx")]
impl From<mlx_bridge::MlxCError> for TextExecutionError {
    fn from(captured_error: mlx_bridge::MlxCError) -> Self {
        MlxRuntimeError::from(captured_error).into()
    }
}
