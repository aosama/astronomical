//! The two ways this engine turns a request away: as invalid, or as one the
//! streaming engine has to serve.

use crate::{InferenceEngineError, StreamingEngineRequirement};

pub use text_generation::engine_errors::invalid_request_error;

/// The request does not fit beside every weight, so it moves to the
/// streaming engine, which keeps what fits and reads the rest from the SSD.
///
/// That holds for a dense model as much as for one with experts: the
/// streaming engine keeps a dense model's leading layers and streams the
/// others (issue #1985). A dense model used to get an invalid-request error
/// here, which a person saw as a failed chat at a limit just above the
/// model's size.
pub(in crate::qwen3_5_resident) fn resident_capacity_error(
    cause: StreamingEngineRequirement,
    reason: impl Into<String>,
) -> InferenceEngineError {
    InferenceEngineError::ResidentForkRequired {
        cause,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "request_errors_tests.rs"]
mod tests;
