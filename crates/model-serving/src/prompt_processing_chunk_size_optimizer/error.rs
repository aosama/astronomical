//! Typed setup, measurement, and persistence failures at the optimizer boundary.

use std::path::PathBuf;
use thiserror::Error;

/// Invalid setup or measurement for prompt-processing chunk-size optimization.
#[derive(Debug, Error)]
pub enum PromptProcessingChunkSizeOptimizerError {
    #[error("at least one candidate prompt_processing_chunk_size_tokens value is required")]
    NoCandidateChunkSizeTokens,
    #[error("candidate prompt_processing_chunk_size_tokens values must be positive")]
    CandidateChunkSizeTokensMustBePositive,
    #[error(
        "candidate prompt_processing_chunk_size_tokens value {candidate_chunk_size_tokens} was not registered"
    )]
    UnregisteredCandidateChunkSizeTokens { candidate_chunk_size_tokens: usize },
    #[error("measured prompt-processing chunk forward elapsed milliseconds must be positive")]
    MeasurementForwardElapsedMillisMustBePositive,
    #[error("measured prompt-processing chunk processed token count must be positive")]
    MeasurementProcessedTokenCountMustBePositive,
    #[error("failed to create optimizer state directory {directory}")]
    OptimizerStateDirectoryCreationFailed {
        directory: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize optimizer state")]
    OptimizerStateSerializationFailed { source: serde_json::Error },
    #[error("failed to write optimizer state to {path}")]
    OptimizerStateWriteFailed {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to rename optimizer state from {from} to {to}")]
    OptimizerStateRenameFailed {
        from: PathBuf,
        to: PathBuf,
        source: std::io::Error,
    },
}
