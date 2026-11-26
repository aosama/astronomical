//! Beside-source coverage for the pure chat-endpoint rejection mappings:
//! the generation-start error to HTTP response mapping and the validation
//! error to HTTP status split. The endpoint orchestration itself runs under
//! the daemon integration journeys.

use axum::http::StatusCode;

use super::{chat_validation_rejection_response, split_chat_request};
use crate::GenerationStartError;
use crate::rest::openai::chat_generation_admission::generation_start_rejection_response;
use rest_contract::{OpenAiChatCompletionRequest, OpenAiChatCompletionValidationError};

fn shakespeare_chat_request(extra_fields: &str) -> OpenAiChatCompletionRequest {
    let request_json = format!(
        r#"{{"model":"local-model","messages":[{{"role":"user","content":"Shall I compare thee to a summer's day?"}}]{extra_fields}}}"#
    );
    serde_json::from_str(&request_json).expect("the Shakespeare chat request should parse")
}

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

#[test]
fn should_split_a_streamed_request_into_parts_and_usage_preferences() {
    let chat_completion_request =
        shakespeare_chat_request(r#","stream":true,"stream_options":{"include_usage":true}"#);

    let (request_parts, reply_plan) =
        split_chat_request(chat_completion_request).expect("the request should be valid");

    assert_eq!(request_parts.model, "local-model");
    assert!(reply_plan.should_stream_response);
    assert!(reply_plan.includes_usage);
}

#[test]
fn should_split_a_plain_request_into_a_non_streaming_reply_without_usage() {
    let chat_completion_request = shakespeare_chat_request("");

    let (_request_parts, reply_plan) =
        split_chat_request(chat_completion_request).expect("the request should be valid");

    assert!(!reply_plan.should_stream_response);
    assert!(!reply_plan.includes_usage);
    assert!(!reply_plan.reasoning_excluded);
    assert!(reply_plan.structured_output.is_none());
}

#[test]
fn should_refuse_to_split_a_request_without_messages() {
    let chat_completion_request: OpenAiChatCompletionRequest =
        serde_json::from_str(r#"{"model":"local-model","messages":[]}"#)
            .expect("an empty message list should still parse");

    let split_outcome = split_chat_request(chat_completion_request);

    assert!(matches!(
        split_outcome,
        Err(OpenAiChatCompletionValidationError::EmptyMessages)
    ));
}
