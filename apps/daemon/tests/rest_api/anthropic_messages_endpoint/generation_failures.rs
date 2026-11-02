use super::*;

async fn assert_error_status_and_type(
    http_response: axum::response::Response,
    expected_status: u16,
    expected_error_type: &str,
) {
    assert_eq!(http_response.status().as_u16(), expected_status);
    let error_document = response_json(http_response).await;
    assert_eq!(error_document["error"]["type"], expected_error_type);
}

#[tokio::test]
async fn should_answer_an_unavailable_worker_as_retryable_overload() {
    let application = build_application(ScriptedChatExecutor::with_start_error(
        GenerationStartError::WorkerUnavailable,
    ));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 529, "overloaded_error").await;
}

#[tokio::test]
async fn should_answer_a_model_load_failure_as_an_api_error() {
    let application = build_application(ScriptedChatExecutor::with_start_error(
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason: "weights missing".to_owned(),
        },
    ));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 500, "api_error").await;
}

#[tokio::test]
async fn should_answer_an_oversized_worker_command_as_request_too_large() {
    let application = build_application(ScriptedChatExecutor::with_start_error(
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes: 2,
            maximum_ipc_message_bytes: 1,
        },
    ));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 413, "request_too_large").await;
}

#[tokio::test]
async fn should_answer_a_worker_that_closes_before_completing_as_overloaded() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Partial".to_owned()),
    ]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 529, "overloaded_error").await;
}

#[tokio::test]
async fn should_answer_a_cancelled_non_streaming_request_as_an_api_error() {
    let application = build_application(ScriptedChatExecutor::new(vec![completed_event(
        ChatGenerationCompletionReason::Cancelled,
    )]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 500, "api_error").await;
}

#[tokio::test]
async fn should_answer_a_busy_engine_as_retryable_overload() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::EngineBusy,
        },
    ]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_error_status_and_type(http_response, 529, "overloaded_error").await;
}

#[tokio::test]
async fn should_return_an_empty_thinking_block_when_the_display_is_omitted() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::ReasoningFragment("private reasoning".to_owned()),
        ChatGenerationStreamEvent::TextFragment("Thou art more lovely".to_owned()),
        completed_event(ChatGenerationCompletionReason::EndOfSequence),
    ]));
    let request_body = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":256,"thinking":{{"type":"adaptive","display":"omitted"}},"messages":[{{"role":"user","content":"Compare thee."}}]}}"#
    );

    let http_response = post_messages_body(application, &request_body).await;

    let message_document = response_json(http_response).await;
    assert_eq!(message_document["content"][0]["type"], "thinking");
    assert_eq!(message_document["content"][0]["thinking"], "");
    assert_eq!(
        message_document["content"][1]["text"],
        "Thou art more lovely"
    );
}
