use crate::decoder_cache::DecoderCacheLayoutError;

/// Typed failure from Laguna model construction or a single forward.
#[derive(Debug, PartialEq, Eq)]
pub enum LagunaExecutionError {
    MissingWeight { description: &'static str },
    InvalidActivationGeometry { description: &'static str },
    RuntimeOperation { description: String },
    DecoderCache(DecoderCacheLayoutError),
}

impl LagunaExecutionError {
    #[cfg(feature = "direct-mlx")]
    pub(super) fn missing_weight(description: &'static str) -> Self {
        Self::MissingWeight { description }
    }

    pub(crate) fn invalid_geometry(description: &'static str) -> Self {
        Self::InvalidActivationGeometry { description }
    }
}

#[cfg(feature = "direct-mlx")]
impl From<crate::expert_paging::RetainedExpertLayerCommitError> for LagunaExecutionError {
    fn from(_error: crate::expert_paging::RetainedExpertLayerCommitError) -> Self {
        Self::invalid_geometry("Laguna retained-expert commit failed")
    }
}

#[cfg(feature = "direct-mlx")]
impl From<super::super::paging::LagunaPagingError> for LagunaExecutionError {
    fn from(_error: super::super::paging::LagunaPagingError) -> Self {
        Self::invalid_geometry("Laguna expert paging failed during model execution")
    }
}

impl From<DecoderCacheLayoutError> for LagunaExecutionError {
    fn from(error: DecoderCacheLayoutError) -> Self {
        Self::DecoderCache(error)
    }
}

#[cfg(feature = "direct-mlx")]
impl From<astronomical_runtime_integration::MlxRuntimeError> for LagunaExecutionError {
    fn from(error: astronomical_runtime_integration::MlxRuntimeError) -> Self {
        Self::RuntimeOperation {
            description: format!(
                "an MLX runtime operation failed during Laguna execution: {error}"
            ),
        }
    }
}
