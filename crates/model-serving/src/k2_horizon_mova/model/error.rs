use thiserror::Error;

use astronomical_runtime_integration::MlxRuntimeError;

use crate::k2_horizon_mova::K2HorizonMoVAArtifactValidationError;

/// Why K2 Horizon MoVA execution cannot continue.
#[derive(Debug, Error)]
pub enum K2HorizonMoVAExecutionError {
    #[error(transparent)]
    Artifact(#[from] K2HorizonMoVAArtifactValidationError),
    #[error(transparent)]
    Runtime(#[from] MlxRuntimeError),
    #[error("{description}")]
    InvalidExecution { description: String },
}

#[cfg(feature = "direct-mlx")]
impl From<astronomical_mlx_c_rust::MlxCError> for K2HorizonMoVAExecutionError {
    fn from(captured_error: astronomical_mlx_c_rust::MlxCError) -> Self {
        MlxRuntimeError::from(captured_error).into()
    }
}
