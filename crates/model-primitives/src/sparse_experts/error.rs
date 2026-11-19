/// Typed failure from family-neutral sparse-expert preparation or reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseExpertError {
    InvalidAssignmentGeometry { description: &'static str },
}

#[cfg(feature = "direct-mlx")]
impl SparseExpertError {
    pub(super) fn into_runtime_error(
        self,
        operation: &'static str,
    ) -> astronomical_runtime_integration::MlxRuntimeError {
        match self {
            Self::InvalidAssignmentGeometry { description } => {
                astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
                    operation,
                    description: description.to_owned(),
                }
            }
        }
    }
}
