//! The refusals shared by the embeddings, decisions and speech endpoints.
//!
//! Each refuses a request for the same reasons before it reaches
//! the worker (a malformed body, an unavailable worker, a full queue, a
//! model that failed to load), and a client should see one error shape for
//! each reason whichever endpoint it called.

use axum::{
    Json,
    response::{IntoResponse, Response},
};

use crate::GenerationStartError;

pub(crate) fn invalid_request_response(
    message: impl Into<String>,
    parameter: Option<&'static str>,
    code: &'static str,
) -> Response {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::invalid_request(
                message,
                parameter,
                Some(code),
            ),
        ),
    )
        .into_response()
}

pub(crate) fn worker_unavailable_response() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "the local worker is unavailable",
                Some("worker_unavailable"),
            ),
        ),
    )
        .into_response()
}

pub(crate) fn request_identifier_exhausted_response() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "the local request identifier space is exhausted",
                Some("request_id_exhausted"),
            ),
        ),
    )
        .into_response()
}

pub(crate) fn generation_start_error_response(start_error: GenerationStartError) -> Response {
    match start_error {
        GenerationStartError::CapacityUnavailable => (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::capacity_unavailable(
                    "the generation queue is full",
                ),
            ),
        )
            .into_response(),
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::model_load_failed(
                    model_load_failure_reason,
                ),
            ),
        )
            .into_response(),
        GenerationStartError::RequestTooLarge { .. } => request_too_large_response(),
        GenerationStartError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn request_too_large_response() -> Response {
    (
        axum::http::StatusCode::PAYLOAD_TOO_LARGE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::invalid_request(
                "the request exceeded the local IPC transport limit",
                None,
                Some("request_too_large"),
            ),
        ),
    )
        .into_response()
}
