//! The two ways this engine turns a request away.

use crate::{InferenceEngineError, StreamingEngineRequirement};

use super::Qwen3_5ResidentModel;

pub(crate) fn invalid_request_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.into(),
    }
}

/// The request does not fit beside every weight. A model whose experts are
/// all in memory can be served by the streaming engine instead, so the
/// error says so; any other model has no such engine to move to.
pub(in crate::qwen3_5_resident) fn resident_capacity_error(
    model: &Qwen3_5ResidentModel,
    cause: StreamingEngineRequirement,
    reason: impl Into<String>,
) -> InferenceEngineError {
    let reason = reason.into();
    if model.resident_expert_weights.is_some() {
        InferenceEngineError::ResidentForkRequired { cause, reason }
    } else {
        InferenceEngineError::InvalidRequest { reason }
    }
}
