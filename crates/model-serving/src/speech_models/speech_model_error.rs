//! Why a speech model could not load or speak, shared by every family.
//!
//! The engine boundary turns these into the request-facing failures: an
//! `Input` error is the caller's to fix, the others are the server's.

use astronomical_runtime_integration::MlxRuntimeError;
use astronomical_speech_contract::SpeechFailure;
use mlx_bridge::MlxCError;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum SpeechModelError {
    #[error("{0}")]
    Runtime(#[from] MlxRuntimeError),
    #[error("speech model artifact is not usable: {0}")]
    Artifact(String),
    #[error("speech model cannot speak this input: {0}")]
    Input(String),
}

impl From<MlxCError> for SpeechModelError {
    fn from(binding_error: MlxCError) -> Self {
        Self::Runtime(MlxRuntimeError::from(binding_error))
    }
}

/// An input error is the caller's to fix; anything else is the server's.
impl From<SpeechModelError> for SpeechFailure {
    fn from(model_error: SpeechModelError) -> Self {
        match model_error {
            SpeechModelError::Input(reason) => SpeechFailure::invalid_request(reason),
            other_error => SpeechFailure::fatal_execution(other_error.to_string()),
        }
    }
}

#[cfg(test)]
#[path = "speech_model_error_tests.rs"]
mod tests;
