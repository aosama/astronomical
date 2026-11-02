use super::*;

fn count_request_body() -> String {
    format!(
        r#"{{"model":"{MODEL_ID}","system":"Speak as Prospero.","messages":[{{"role":"user","content":"Our revels now are ended."}}]}}"#
    )
}

async fn post_count_body(
    application: axum::Router,
    request_body: &str,
) -> axum::response::Response {
    application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens?beta=true")
                .header(header::CONTENT_TYPE, "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(request_body.to_owned()))
                .expect("the count request should be valid"),
        )
        .await
        .expect("the application should answer the count request")
}

fn counted_event(prompt_token_count: u32) -> ChatGenerationStreamEvent {
    ChatGenerationStreamEvent::Completed {
        prompt_token_count,
        generated_token_count: 0,
        reasoning_token_count: 0,
        cached_token_count: 0,
        reason: ChatGenerationCompletionReason::EndOfSequence,
    }
}

#[tokio::test]
async fn should_answer_with_the_worker_measured_prompt_token_count() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(vec![
        counted_event(1_234),
    ]));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    assert_eq!(
        response_json(http_response).await,
        serde_json::json!({"input_tokens": 1_234})
    );
}

#[tokio::test]
async fn should_pass_a_zero_count_boundary_through_unchanged() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(vec![
        counted_event(0),
    ]));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(response_json(http_response).await["input_tokens"], 0);
}

#[tokio::test]
async fn should_pass_a_preparation_rejection_through_as_invalid_request() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::invalid_request(
                "image input needs a vision model",
            ),
        },
    ]));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    let error_document = response_json(http_response).await;
    assert_eq!(error_document["error"]["type"], "invalid_request_error");
    assert_eq!(
        error_document["error"]["message"],
        "image input needs a vision model"
    );
}

#[tokio::test]
async fn should_answer_an_overflowing_prompt_as_prompt_too_long() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(
        response_json(http_response).await["error"]["message"],
        "prompt is too long: 262145 tokens > 262144 maximum"
    );
}

#[tokio::test]
async fn should_refuse_to_estimate_when_the_worker_cannot_count() {
    let application = build_application(ScriptedChatExecutor::new(Vec::new()));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(http_response.status().as_u16(), 529);
}

#[tokio::test]
async fn should_reject_generation_controls_and_malformed_json() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(Vec::new()));
    let request_with_max_tokens = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":64,"messages":[{{"role":"user","content":"hello"}}]}}"#
    );

    let http_response = post_count_body(application.clone(), &request_with_max_tokens).await;
    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response_json(http_response).await["error"]["type"],
        "invalid_request_error"
    );

    let http_response = post_count_body(application, "{not json").await;
    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn should_answer_a_count_for_an_unknown_model_with_not_found() {
    let application = build_application(ScriptedChatExecutor::answering_token_counts(Vec::new()));
    let request_body =
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"hello"}]}"#;

    let http_response = post_count_body(application, request_body).await;

    assert_eq!(http_response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        response_json(http_response).await["error"]["type"],
        "not_found_error"
    );
}

#[tokio::test]
async fn should_answer_a_count_against_a_full_queue_with_a_rate_limit_error() {
    let application = build_application(ScriptedChatExecutor::with_start_error(
        GenerationStartError::CapacityUnavailable,
    ));

    let http_response = post_count_body(application, &count_request_body()).await;

    assert_eq!(http_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response_json(http_response).await["error"]["type"],
        "rate_limit_error"
    );
}
