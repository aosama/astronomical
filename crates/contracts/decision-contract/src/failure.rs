//! Why a decision request could not be answered.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A decision request's failure, in the two classes a caller acts on differently.
#[derive(Clone, Debug, Deserialize, Eq, Error, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionFailure {
    /// The request itself cannot be answered by this model; sending it again
    /// unchanged fails the same way. The reason names what to change.
    #[error("{reason}")]
    InvalidRequest { reason: String },
    /// The model could not run. The request may be fine.
    #[error("{reason}")]
    FatalExecution { reason: String },
    /// Another request already owns the worker; this one may be sent again.
    #[error("another request is already running")]
    EngineBusy,
}

/// The longest failure reason kept, in characters.
///
/// Reasons embed request text and library error messages of unbounded
/// length, and a failure must still fit one inter-process message.
pub const MAXIMUM_FAILURE_REASON_CHARACTERS: usize = 512;

impl DecisionFailure {
    /// An invalid-request failure whose reason is cut to the kept length.
    #[must_use]
    pub fn invalid_request(reason: &str) -> Self {
        Self::InvalidRequest {
            reason: bounded_reason(reason),
        }
    }

    /// A fatal-execution failure whose reason is cut to the kept length.
    #[must_use]
    pub fn fatal_execution(reason: &str) -> Self {
        Self::FatalExecution {
            reason: bounded_reason(reason),
        }
    }
}

fn bounded_reason(reason: &str) -> String {
    reason
        .chars()
        .take(MAXIMUM_FAILURE_REASON_CHARACTERS)
        .collect()
}

#[cfg(test)]
#[path = "failure_tests.rs"]
mod tests;
