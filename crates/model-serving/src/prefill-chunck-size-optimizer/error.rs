use std::path::PathBuf;
use thiserror::Error;

/// Invalid setup or observation for prompt pre-processing chunk-size optimization.
#[derive(Debug, Error)]
pub enum PrefillChunckSizeOptimizerError {
    #[error("at least one candidate prefill_chunck_tokens value is required")]
    NoCandidatePrefillChunckTokens,
    #[error("candidate prefill_chunck_tokens values must be positive")]
    CandidatePrefillChunckTokensMustBePositive,
    #[error("drift trigger factor must be at least two")]
    DriftTriggerFactorMustBeAtLeastTwo,
    #[error(
        "candidate prefill_chunck_tokens value {candidate_prefill_chunck_tokens} was not registered"
    )]
    UnregisteredCandidatePrefillChunckTokens {
        candidate_prefill_chunck_tokens: usize,
    },
    #[error("observed prefill chunk elapsed milliseconds must be positive")]
    ObservationElapsedMillisMustBePositive,
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
