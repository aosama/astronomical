use std::io;
use std::path::PathBuf;

use astronomical_model_serving::{
    ExpertMemoryAdmissionError, InferenceEngineError, Qwen3_5ArtifactValidationError,
    Qwen3_5RamBudgetGeometryError, Qwen3_5ResidentPromptProcessingChunkSizerError,
    Qwen3_5StreamingPromptProcessingChunkSizerError, Qwen3_5TokenizerError,
};
use thiserror::Error;

use crate::worker_startup_error;

/// Failure produced while constructing the Qwen family processor and engine.
#[derive(Debug, Error)]
pub enum Qwen3_5ModelStartupError {
    #[error("failed to validate Qwen3.5 artifact at {model_directory:?}")]
    ArtifactValidation {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5ArtifactValidationError,
    },
    #[error("failed to derive Qwen3.5 artifact RAM geometry at {model_directory:?}")]
    ArtifactRamGeometry {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5RamBudgetGeometryError,
    },
    #[error("failed to decide Qwen3.5 complete residency at {model_directory:?}")]
    CompleteResidencyDecision {
        model_directory: PathBuf,
        #[source]
        source: ExpertMemoryAdmissionError,
    },
    #[error("failed to initialize Qwen3.5 processor at {model_directory:?}")]
    ProcessorInitialization {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5TokenizerError,
    },
    #[error("failed to open Qwen3.5 performance-attribution log at {log_path:?}")]
    OpenPerformanceAttributionLog {
        log_path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to configure Qwen3.5 prompt-processing chunks")]
    StreamingPromptProcessingChunkSizing(#[source] Qwen3_5StreamingPromptProcessingChunkSizerError),
    #[error("failed to configure resident Qwen3.5 prompt-processing chunks")]
    ResidentPromptProcessingChunkSizing(#[source] Qwen3_5ResidentPromptProcessingChunkSizerError),
    #[error("failed to start Qwen3.5 engine at {model_directory:?}")]
    EngineInitialization {
        model_directory: PathBuf,
        #[source]
        source: InferenceEngineError,
    },
}

impl Qwen3_5ModelStartupError {
    /// Describes a Qwen model-load failure with safe detail but no local paths.
    #[must_use]
    pub fn public_model_load_failure_reason(&self) -> String {
        let unbounded_public_model_load_failure_reason = match self {
            Self::ArtifactValidation { source, .. } => source.public_failure_reason(),
            Self::ArtifactRamGeometry { .. } => {
                "Qwen3.5 artifact RAM geometry could not be derived".to_owned()
            }
            Self::CompleteResidencyDecision { .. } => {
                "Qwen3.5 complete-residency admission failed".to_owned()
            }
            Self::ProcessorInitialization { .. } => {
                "Qwen3.5 processor initialization failed".to_owned()
            }
            Self::EngineInitialization { .. } => "Qwen3.5 engine initialization failed".to_owned(),
            Self::OpenPerformanceAttributionLog { .. }
            | Self::StreamingPromptProcessingChunkSizing(_) => {
                "model initialization failed".to_owned()
            }
            Self::ResidentPromptProcessingChunkSizing(_) => {
                "model initialization failed".to_owned()
            }
        };
        worker_startup_error::bound_public_model_load_failure_reason(
            unbounded_public_model_load_failure_reason,
        )
    }
}
