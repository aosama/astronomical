//! Why a speech request did not produce audio.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The longest reason a failure carries; a longer one is cut, so an error
/// from deep in a model cannot flood a response or a log line.
const MAXIMUM_REASON_CHARACTERS: usize = 512;

#[derive(Clone, Debug, Deserialize, Error, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFailure {
    /// The caller's to fix: empty text, an unknown voice, a speed the model
    /// does not support.
    #[error("invalid speech request: {reason}")]
    InvalidRequest { reason: String },
    /// The server's: the model failed while loading or speaking.
    #[error("speech failed: {reason}")]
    FatalExecution { reason: String },
    /// Another request is being spoken.
    #[error("the speech engine is busy with another request")]
    EngineBusy,
    /// The caller withdrew the request; the segments already spoken are dropped.
    #[error("the speech request was cancelled")]
    Cancelled,
}

impl SpeechFailure {
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: bounded_reason(reason.into()),
        }
    }

    pub fn fatal_execution(reason: impl Into<String>) -> Self {
        Self::FatalExecution {
            reason: bounded_reason(reason.into()),
        }
    }
}

fn bounded_reason(reason: String) -> String {
    if reason.chars().count() <= MAXIMUM_REASON_CHARACTERS {
        return reason;
    }
    let mut bounded: String = reason.chars().take(MAXIMUM_REASON_CHARACTERS).collect();
    bounded.push('…');
    bounded
}

#[cfg(test)]
#[path = "speech_failure_tests.rs"]
mod tests;
