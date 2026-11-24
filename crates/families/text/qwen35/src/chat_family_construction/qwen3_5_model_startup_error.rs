//! Failure vocabulary for the Qwen3.5 startup pipeline. The public reason
//! carries no local paths; the worker bounds its length at its own boundary
//! before it reaches daemon-facing diagnostics.

use std::io;
use std::path::PathBuf;

use crate::{
    ExpertMemoryAdmissionError, InferenceEngineError, Qwen3_5ArtifactValidationError,
    Qwen3_5MemoryGeometryError, Qwen3_5PromptProcessingChunkSizerError, Qwen3_5TokenizerError,
};
use thiserror::Error;

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
        source: Qwen3_5MemoryGeometryError,
    },
    #[error("failed to decide Qwen3.5 complete residency at {model_directory:?}")]
    CompleteResidencyDecision {
        model_directory: PathBuf,
        #[source]
        source: ExpertMemoryAdmissionError,
    },
    #[error("failed to total the Qwen3.5 text weights at {model_directory:?}")]
    TextWeightPayloadUnreadable { model_directory: PathBuf },
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
    StreamingPromptProcessingChunkSizing(#[source] Qwen3_5PromptProcessingChunkSizerError),
    #[error("failed to configure resident Qwen3.5 prompt-processing chunks")]
    ResidentPromptProcessingChunkSizing(#[source] Qwen3_5PromptProcessingChunkSizerError),
    #[error("failed to start Qwen3.5 engine at {model_directory:?}")]
    EngineInitialization {
        model_directory: PathBuf,
        #[source]
        source: InferenceEngineError,
    },
}

impl Qwen3_5ModelStartupError {
    /// Describes a Qwen model-load failure with safe detail but no local
    /// paths. The worker bounds its length.
    #[must_use]
    pub fn public_model_load_failure_reason(&self) -> String {
        match self {
            Self::ArtifactValidation { source, .. } => source.public_failure_reason(),
            Self::ArtifactRamGeometry { .. } | Self::TextWeightPayloadUnreadable { .. } => {
                ipc_protocol::MEMORY_GEOMETRY_FAILED_REASON.to_owned()
            }
            Self::CompleteResidencyDecision { .. } => {
                ipc_protocol::RESIDENCY_ADMISSION_FAILED_REASON.to_owned()
            }
            Self::ProcessorInitialization { .. } => {
                ipc_protocol::PROCESSOR_INITIALIZATION_FAILED_REASON.to_owned()
            }
            Self::EngineInitialization { .. } => {
                ipc_protocol::QWEN_ENGINE_INITIALIZATION_FAILED_REASON.to_owned()
            }
            Self::OpenPerformanceAttributionLog { .. }
            | Self::StreamingPromptProcessingChunkSizing(_) => {
                "model initialization failed".to_owned()
            }
            Self::ResidentPromptProcessingChunkSizing(_) => {
                "model initialization failed".to_owned()
            }
        }
    }
}
