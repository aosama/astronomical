//! Beside-source coverage for the pure chat-endpoint rejection mappings:
//! the generation-start error to HTTP response mapping and the validation
//! error to HTTP status split. The endpoint orchestration itself runs under
//! the daemon integration journeys.

use axum::http::StatusCode;

use super::{chat_validation_rejection_response, generation_start_rejection_response};
use crate::GenerationStartError;
use astronomical_rest_contract::OpenAiChatCompletionValidationError;

#[test]
fn should_map_capacity_unavailability_to_http_429() {
    let response = generation_start_rejection_response(GenerationStartError::CapacityUnavailable);

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn should_map_model_load_failure_to_http_503() {
    let response = generation_start_rejection_response(GenerationStartError::ModelLoadFailed {
        model_load_failure_reason: "the quantized pack is unreadable".to_owned(),
    });

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn should_map_oversized_requests_to_http_413() {
    let response = generation_start_rejection_response(GenerationStartError::RequestTooLarge {
        actual_ipc_message_bytes: 1_048_576,
        maximum_ipc_message_bytes: 262_144,
    });

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[test]
fn should_map_worker_unavailability_to_http_503() {
    let response = generation_start_rejection_response(GenerationStartError::WorkerUnavailable);

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn should_map_portable_budget_rejections_to_http_413() {
    let response =
        chat_validation_rejection_response(OpenAiChatCompletionValidationError::ImageTooLarge {
            actual_bytes: 3000,
            maximum_bytes: 2000,
            excess_bytes: 1000,
        });

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[test]
fn should_map_standard_validation_rejections_to_http_400() {
    let response =
        chat_validation_rejection_response(OpenAiChatCompletionValidationError::EmptyMessages);

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
