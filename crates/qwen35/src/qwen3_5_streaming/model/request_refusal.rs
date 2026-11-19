//! The error a request is refused with before any forward runs.

use crate::InferenceEngineError;

pub fn invalid_request_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.into(),
    }
}
