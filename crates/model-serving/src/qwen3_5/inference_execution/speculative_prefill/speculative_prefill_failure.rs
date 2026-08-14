//! Central fail-closed translation for configured SpecPrefill failures.
//!
//! Internal errors may contain native details or local paths. These helpers log
//! that evidence locally and return a bounded, stable explanation to the worker
//! protocol. Keeping translation in one place also makes the no-target-only-
//! retry guarantee explicit for every failure stage.

use astronomical_ipc_protocol::RequestId;

// The production parent and the hermetic include parent both import this public
// error type, keeping this fail-closed translator testable without crate aliases.
use super::InferenceEngineError;

/// Converts a configured SpecPrefill execution problem into a bounded
/// user-visible request failure without changing execution mode.
///
/// Request-scoped failures preserve the loaded worker. They are invalid-request
/// outcomes because only this request's partially prepared state is abandoned.
pub(crate) fn configured_speculative_prefill_failure(
    request_id: RequestId,
    failure_stage: &'static str,
    internal_error: impl std::fmt::Display,
) -> InferenceEngineError {
    // Log rich internal evidence before replacing it with a bounded protocol-safe
    // message. This is local-only diagnostics, not public error text.
    tracing::error!(
        request_id = request_id.value(),
        failure_stage,
        error = %internal_error,
        "configured SpecPrefill execution stopped the request"
    );
    InferenceEngineError::InvalidRequest {
        // Never claim target-only recovery: no fallback is attempted after
        // configured SpecPrefill has started mutating request state.
        reason: format!(
            "configured SpecPrefill failed during {failure_stage}; the request was stopped without a target-only retry"
        ),
    }
}

/// Converts a configured SpecPrefill activation problem into a bounded
/// user-visible model-loading failure without activating a partial policy.
///
/// Activation failures are fatal to this model load because configuration
/// explicitly requested SpecPrefill and startup could not establish its complete
/// compatibility/storage contract.
pub(crate) fn configured_speculative_prefill_activation_failure(
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
