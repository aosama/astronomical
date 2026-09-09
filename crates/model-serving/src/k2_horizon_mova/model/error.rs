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
