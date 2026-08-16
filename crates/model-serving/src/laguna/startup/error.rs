use thiserror::Error;

use crate::InferenceEngineError;
use crate::laguna::{
    LagunaArtifactValidationError, LagunaPromptProcessingChunkSizerError, LagunaTokenizerError,
};

/// Failure while constructing a Laguna processor and engine from a validated artifact.
#[derive(Debug, Error)]
pub enum LagunaStartupError {
    #[error("Laguna artifact validation failed")]
    ArtifactValidation(#[source] LagunaArtifactValidationError),
    #[error("Laguna artifact requires one immutable revision")]
    ImmutableRevisionRequired,
    #[error("Laguna processor initialization failed")]
    ProcessorInitialization(#[source] LagunaTokenizerError),
    #[error("Laguna weight loading failed")]
    WeightLoading,
    #[error("Laguna paging-plan construction failed")]
    PagingPlan,
    #[error("Laguna model construction failed")]
    ModelConstruction,
    #[error("Laguna runtime initialization failed")]
    RuntimeInitialization,
    #[error("Laguna engine owner thread failed")]
    EngineOwner(#[source] InferenceEngineError),
    #[error("Laguna prompt-processing chunk sizer initialization failed")]
    ChunkSizer(#[source] LagunaPromptProcessingChunkSizerError),
    #[error("Laguna performance-attribution log initialization failed")]
    PerformanceAttributionLog(#[source] std::io::Error),
}

impl LagunaStartupError {
    /// Describes a load failure without local paths or native details.
    #[must_use]
    pub fn public_model_load_failure_reason(&self) -> String {
        match self {
            Self::ArtifactValidation(_) => "Laguna artifact validation failed".to_owned(),
            Self::ImmutableRevisionRequired => {
                "Laguna artifact requires one immutable revision".to_owned()
            }
            Self::ProcessorInitialization(_) => "Laguna processor initialization failed".to_owned(),
            Self::WeightLoading => "Laguna weight loading failed".to_owned(),
            Self::PagingPlan => "Laguna paging-plan construction failed".to_owned(),
            Self::ModelConstruction => "Laguna model construction failed".to_owned(),
            Self::RuntimeInitialization => "Laguna runtime initialization failed".to_owned(),
            Self::EngineOwner(_) => "Laguna engine initialization failed".to_owned(),
            Self::ChunkSizer(_) => {
                "Laguna prompt-processing chunk sizer initialization failed".to_owned()
            }
            Self::PerformanceAttributionLog(_) => {
                "Laguna performance attribution could not initialize".to_owned()
            }
        }
    }
}
