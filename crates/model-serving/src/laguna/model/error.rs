use crate::decoder_cache::DecoderCacheLayoutError;
#[cfg(feature = "direct-mlx")]
use crate::laguna::paging::LagunaPagingError;
#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

/// Typed failure from Laguna model construction or a single forward.
#[derive(Debug, Error)]
pub enum LagunaExecutionError {
    #[error("a required Laguna weight is missing: {description}")]
    MissingWeight { description: &'static str },
    #[error("Laguna activation geometry is invalid: {description}")]
    InvalidActivationGeometry { description: &'static str },
    #[error("Laguna runtime operation failed: {description}")]
    RuntimeOperation { description: String },
    #[cfg(feature = "direct-mlx")]
    #[error(
        "Laguna expert allocation requires {pending_allocation_bytes} bytes and exceeds the active ceiling by {shortfall_bytes} bytes"
    )]
    ExpertAllocationRejected {
        pending_allocation_bytes: u64,
        shortfall_bytes: u64,
    },
    #[error("Laguna decoder cache failed: {0}")]
    DecoderCache(#[from] DecoderCacheLayoutError),
    #[cfg(feature = "direct-mlx")]
    #[error("Laguna expert paging failed: {0}")]
    Paging(LagunaPagingError),
    #[cfg(feature = "direct-mlx")]
    #[error("Laguna reached an MLX runtime boundary: {0}")]
    Runtime(MlxRuntimeError),
}

impl LagunaExecutionError {
    #[cfg(feature = "direct-mlx")]
    pub(super) fn missing_weight(description: &'static str) -> Self {
        Self::MissingWeight { description }
    }

    pub(crate) fn invalid_geometry(description: &'static str) -> Self {
        Self::InvalidActivationGeometry { description }
    }

    #[cfg(feature = "direct-mlx")]
    /// Capacity recovery must never classify structural model failures as memory pressure.
    #[must_use]
    pub fn is_recoverable_memory_pressure(&self) -> bool {
        match self {
            Self::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded { .. }) => true,
            Self::Runtime(runtime_error) => {
                runtime_error.is_recoverable_graphics_processor_out_of_memory()
            }
            Self::ExpertAllocationRejected { .. } => true,
            _ => false,
        }
    }
}

#[cfg(feature = "direct-mlx")]
impl From<crate::expert_paging::RetainedExpertLayerCommitError> for LagunaExecutionError {
    fn from(_error: crate::expert_paging::RetainedExpertLayerCommitError) -> Self {
        Self::invalid_geometry("Laguna retained-expert commit failed")
    }
}

#[cfg(feature = "direct-mlx")]
impl From<LagunaPagingError> for LagunaExecutionError {
    fn from(error: LagunaPagingError) -> Self {
        match error {
            LagunaPagingError::Runtime(runtime_error) => Self::Runtime(runtime_error),
            paging_error => Self::Paging(paging_error),
        }
    }
}

#[cfg(feature = "direct-mlx")]
impl From<MlxRuntimeError> for LagunaExecutionError {
    fn from(error: MlxRuntimeError) -> Self {
        Self::Runtime(error)
    }
}
