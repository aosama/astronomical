use thiserror::Error;

use crate::artifact_validation::ArtifactValidationError;
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfigError;

/// Why a K2 Horizon MoVA directory cannot be served.
#[derive(Debug, Error)]
pub enum K2HorizonMoVAArtifactValidationError {
    #[error(transparent)]
    Artifact(#[from] ArtifactValidationError),
    #[error(transparent)]
    Config(#[from] K2HorizonMoVAConfigError),
    #[error("{description}")]
    InvalidArtifact { description: String },
}
