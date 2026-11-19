//! Fatal-error constructors shared by both Qwen3.5 inference engines.

use crate::InferenceEngineError;

pub fn qwen3_5_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    fatal_engine_error(runtime_error.to_string())
}

pub fn fatal_engine_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}
