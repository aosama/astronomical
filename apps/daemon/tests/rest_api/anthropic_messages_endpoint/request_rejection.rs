use super::*;

async fn assert_anthropic_error(
    http_response: axum::response::Response,
    expected_status: StatusCode,
    expected_error_type: &str,
) -> Value {
    assert_eq!(http_response.status(), expected_status);
    let error_document = response_json(http_response).await;
    assert_eq!(error_document["type"], "error");
    assert_eq!(error_document["error"]["type"], expected_error_type);
    error_document
}

#[tokio::test]
async fn should_reject_malformed_json_with_an_invalid_request_error() {
    let application = build_application(ScriptedChatExecutor::new(Vec::new()));

    let http_response = post_messages_body(application, "{not json").await;

    assert_anthropic_error(
        http_response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    )
    .await;
}

#[tokio::test]
async fn should_reject_a_request_without_max_tokens() {
    let application = build_application(ScriptedChatExecutor::new(Vec::new()));
    let request_body =
        format!(r#"{{"model":"{MODEL_ID}","messages":[{{"role":"user","content":"hello"}}]}}"#);

    let http_response = post_messages_body(application, &request_body).await;

    assert_anthropic_error(
        http_response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    )
    .await;
}

#[tokio::test]
async fn should_reject_stop_sequences_the_worker_cannot_apply() {
    let application = build_application(ScriptedChatExecutor::new(Vec::new()));
    let request_body = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":64,"stop_sequences":["THE END"],"messages":[{{"role":"user","content":"hello"}}]}}"#
    );

    let http_response = post_messages_body(application, &request_body).await;

    let error_document = assert_anthropic_error(
        http_response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    )
    .await;
    assert!(
        error_document["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("stop_sequences")
    );
}

#[tokio::test]
async fn should_answer_an_unknown_model_with_not_found() {
    let application = build_application(ScriptedChatExecutor::new(Vec::new()));
    let request_body = r#"{"model":"no-such-model","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}"#;

    let http_response = post_messages_body(application, request_body).await;

    assert_anthropic_error(http_response, StatusCode::NOT_FOUND, "not_found_error").await;
}

#[tokio::test]
async fn should_answer_a_full_queue_with_a_rate_limit_error() {
    let application = build_application(ScriptedChatExecutor::with_start_error(
        GenerationStartError::CapacityUnavailable,
    ));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    assert_anthropic_error(
        http_response,
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_error",
    )
    .await;
}

#[tokio::test]
async fn should_answer_non_streaming_context_overflow_as_prompt_too_long() {
    let application = build_application(ScriptedChatExecutor::new(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let http_response = post_messages_body(application, &messages_request_body(false)).await;

    let error_document = assert_anthropic_error(
        http_response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    )
    .await;
    assert_eq!(
        error_document["error"]["message"],
        "prompt is too long: 262145 tokens > 262144 maximum"
    );
}
