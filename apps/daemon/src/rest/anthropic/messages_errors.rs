//! Every Anthropic error the Messages endpoint answers with: pure mappings
//! from contract rejections, generation-start refusals, and worker failures
//! onto Anthropic error values, plus the one conversion to an HTTP response,
//! whose status always comes from the error's own type.

use astronomical_ipc_protocol::ChatGenerationFailureReason;
use astronomical_rest_contract::{AnthropicError, AnthropicMessagesValidationError};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::messages_translation::AnthropicMessagesTranslationError;
use crate::GenerationStartError;

/// Answers with `error` under the HTTP status its Anthropic type owns.
pub(crate) fn anthropic_error_response(error: AnthropicError) -> Response {
    let status_code = match StatusCode::from_u16(error.http_status_code()) {
        Ok(status_code) => status_code,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status_code, Json(error.into_response_body())).into_response()
}

pub(crate) fn json_rejection_error(json_error: &serde_json::Error) -> AnthropicError {
    AnthropicError::invalid_request(format!(
        "request body is not a valid Messages request: {json_error}"
    ))
}

pub(crate) fn validation_rejection_error(
    validation_error: &AnthropicMessagesValidationError,
) -> AnthropicError {
    if validation_error.is_request_too_large() {
        return AnthropicError::request_too_large(validation_error.to_string());
    }
    AnthropicError::invalid_request(validation_error.to_string())
}

pub(crate) fn translation_rejection_error(
    translation_error: &AnthropicMessagesTranslationError,
) -> AnthropicError {
    AnthropicError::invalid_request(translation_error.to_string())
}

pub(crate) fn unknown_model_error(requested_model: &str) -> AnthropicError {
    AnthropicError::not_found(format!("model: {requested_model}"))
}

pub(crate) fn model_capability_mismatch_error() -> AnthropicError {
    AnthropicError::invalid_request("the requested model does not support Messages generation")
}

/// The worker is restarting or not yet ready: a temporary condition Anthropic
/// clients retry on, so it is answered as overloaded.
pub(crate) fn worker_unavailable_error() -> AnthropicError {
    AnthropicError::overloaded("the local worker is unavailable")
}

pub(crate) fn request_identifier_exhausted_error() -> AnthropicError {
    AnthropicError::api_error("the local request identifier space is exhausted")
}

pub(crate) fn generation_start_error(
    generation_start_error: GenerationStartError,
) -> AnthropicError {
    match generation_start_error {
        GenerationStartError::CapacityUnavailable => {
            AnthropicError::rate_limit("the generation queue is full")
        }
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => AnthropicError::api_error(format!(
            "the model failed to load: {model_load_failure_reason}"
        )),
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes,
        } => AnthropicError::request_too_large(format!(
            "the request expands to {actual_ipc_message_bytes} bytes for local processing, exceeding the {maximum_ipc_message_bytes}-byte limit; reduce image sizes or conversation history"
        )),
        GenerationStartError::WorkerUnavailable => worker_unavailable_error(),
    }
}

/// Maps a worker failure onto an Anthropic error. Context overflow uses
/// Anthropic's own "prompt is too long" wording because Anthropic clients
/// match that text to trigger conversation compaction.
pub(crate) fn failure_error(failure_reason: &ChatGenerationFailureReason) -> AnthropicError {
    match failure_reason {
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => AnthropicError::invalid_request(format!(
            "prompt is too long: {actual_total_context_tokens} tokens > {maximum_context_tokens} maximum"
        )),
        ChatGenerationFailureReason::InvalidRequest { reason } => {
            AnthropicError::invalid_request(reason.clone())
        }
        ChatGenerationFailureReason::EngineBusy => {
            AnthropicError::overloaded("the local worker is busy with another request")
        }
        other_failure_reason => AnthropicError::api_error(format!(
            "the local worker could not generate the message: {other_failure_reason:?}"
        )),
    }
}
