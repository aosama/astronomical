use astronomical_ipc_protocol::ChatGenerationFailureReason;
use astronomical_rest_contract::{AnthropicError, AnthropicErrorType};

use super::messages_errors::{
    failure_error, generation_start_error, unknown_model_error, worker_unavailable_error,
};
use crate::GenerationStartError;

fn error_type_and_status(error: &AnthropicError) -> (AnthropicErrorType, u16) {
    (error.error_type, error.http_status_code())
}

#[test]
fn should_pair_each_generation_start_refusal_with_its_anthropic_type_and_status() {
    for (start_error, expected_type, expected_status) in [
        (
            GenerationStartError::CapacityUnavailable,
            AnthropicErrorType::RateLimitError,
            429,
        ),
        (
            GenerationStartError::ModelLoadFailed {
                model_load_failure_reason: "weights missing".to_owned(),
            },
            AnthropicErrorType::ApiError,
            500,
        ),
        (
            GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes: 2,
                maximum_ipc_message_bytes: 1,
            },
            AnthropicErrorType::RequestTooLarge,
            413,
        ),
        (
            GenerationStartError::WorkerUnavailable,
            AnthropicErrorType::OverloadedError,
            529,
        ),
    ] {
        let error = generation_start_error(start_error);

        assert_eq!(
            error_type_and_status(&error),
            (expected_type, expected_status)
        );
    }
}

#[test]
fn should_map_worker_failures_onto_retryable_or_terminal_errors() {
    let busy_error = failure_error(&ChatGenerationFailureReason::EngineBusy);
    assert_eq!(
        error_type_and_status(&busy_error),
        (AnthropicErrorType::OverloadedError, 529)
    );

    let invalid_error = failure_error(&ChatGenerationFailureReason::invalid_request(
        "unknown tool call id",
    ));
    assert_eq!(
        error_type_and_status(&invalid_error),
        (AnthropicErrorType::InvalidRequestError, 400)
    );
    assert_eq!(invalid_error.message, "unknown tool call id");

    let malformed_error = failure_error(&ChatGenerationFailureReason::MalformedModelOutput);
    assert_eq!(
        error_type_and_status(&malformed_error),
        (AnthropicErrorType::ApiError, 500)
    );
}

/// Running out of graphics memory must not look retryable: an Anthropic
/// client that retries would reload the model and fail the same way.
#[test]
fn should_report_out_of_graphics_memory_as_terminal_with_advice() {
    let out_of_memory_error = failure_error(&ChatGenerationFailureReason::OutOfGraphicsMemory);

    assert_eq!(
        error_type_and_status(&out_of_memory_error),
        (AnthropicErrorType::ApiError, 500)
    );
    assert!(
        out_of_memory_error
            .message
            .contains("ran out of graphics memory")
    );
    assert!(out_of_memory_error.message.contains("use a smaller model"));
}

#[test]
fn should_word_context_overflow_the_way_anthropic_clients_match_it() {
    let overflow_error = failure_error(&ChatGenerationFailureReason::ContextLengthExceeded {
        actual_total_context_tokens: 1,
        maximum_context_tokens: 0,
    });

    assert_eq!(
        overflow_error.message,
        "prompt is too long: 1 tokens > 0 maximum"
    );
    assert_eq!(overflow_error.http_status_code(), 400);
}

#[test]
fn should_answer_an_unknown_model_with_not_found_and_an_unready_worker_as_overloaded() {
    assert_eq!(
        error_type_and_status(&unknown_model_error("no-such-model")),
        (AnthropicErrorType::NotFoundError, 404)
    );
    assert_eq!(
        error_type_and_status(&worker_unavailable_error()),
        (AnthropicErrorType::OverloadedError, 529)
    );
}
