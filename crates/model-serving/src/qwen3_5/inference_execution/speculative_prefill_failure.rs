use astronomical_ipc_protocol::RequestId;

use super::InferenceEngineError;

/// Converts a configured SpecPrefill execution problem into a bounded
/// user-visible request failure without changing execution mode.
pub(super) fn configured_speculative_prefill_failure(
    request_id: RequestId,
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    tracing::error!(
        request_id = request_id.value(),
        failure_stage,
        error = %internal_error,
        "configured SpecPrefill execution stopped the request"
    );
    InferenceEngineError::InvalidRequest {
        reason: format!(
            "configured SpecPrefill failed during {failure_stage}; the request was stopped without a target-only retry"
        ),
    }
}

/// Converts a configured SpecPrefill activation problem into a bounded
/// user-visible model-loading failure without activating a partial policy.
pub(super) fn configured_speculative_prefill_activation_failure(
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    tracing::error!(
        failure_stage,
        error = %internal_error,
        "configured SpecPrefill activation failed"
    );
    InferenceEngineError::Fatal {
        reason: format!(
            "configured SpecPrefill failed during {failure_stage}; model use was stopped"
        ),
    }
}
