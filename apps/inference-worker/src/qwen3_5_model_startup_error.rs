use std::io;
use std::path::PathBuf;

use astronomical_model_serving::{
    InferenceEngineError, Qwen3_5ArtifactValidationError, Qwen3_5PromptProcessingChunkSizerError,
    Qwen3_5TokenizerError,
};
use thiserror::Error;

use crate::worker_startup_error::bound_public_model_load_failure_reason;

/// Failure produced while constructing the Qwen family processor and engine.
#[derive(Debug, Error)]
pub enum Qwen3_5ModelStartupError {
    #[error("failed to validate Qwen3.5 artifact at {model_directory:?}")]
    ArtifactValidation {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5ArtifactValidationError,
    },
    #[error(
        "requested Qwen model '{requested_model_id}' does not match validated model '{validated_model_id}'"
    )]
    RequestedModelIdentityMismatch {
        requested_model_id: String,
        validated_model_id: String,
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
    PromptProcessingChunkSizing(#[source] Qwen3_5PromptProcessingChunkSizerError),
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
            Self::ArtifactValidation {
                source: Qwen3_5ArtifactValidationError::OptiQMetadata(metadata_error),
                ..
            } => format!("Qwen3.5 OptiQ metadata validation failed: {metadata_error}"),
            Self::ArtifactValidation {
                source: Qwen3_5ArtifactValidationError::Config(config_error),
                ..
            } => format!("Qwen3.5 config validation failed: {config_error}"),
            Self::ArtifactValidation {
                source: Qwen3_5ArtifactValidationError::Qwen3_5ShardIndex(shard_index_error),
                ..
            } => format!("Qwen3.5 shard-index validation failed: {shard_index_error}"),
            Self::ArtifactValidation { .. } => "Qwen3.5 artifact validation failed".to_owned(),
            Self::RequestedModelIdentityMismatch {
                requested_model_id,
                validated_model_id,
            } => format!(
                "requested model '{requested_model_id}' does not match validated model '{validated_model_id}'"
            ),
            Self::ProcessorInitialization { .. } => {
                "Qwen3.5 processor initialization failed".to_owned()
            }
            Self::EngineInitialization { .. } => "Qwen3.5 engine initialization failed".to_owned(),
            Self::OpenPerformanceAttributionLog { .. } | Self::PromptProcessingChunkSizing(_) => {
                "model initialization failed".to_owned()
            }
        };
        bound_public_model_load_failure_reason(unbounded_public_model_load_failure_reason)
    }
}
