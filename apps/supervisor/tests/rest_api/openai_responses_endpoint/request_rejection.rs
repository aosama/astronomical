use super::*;

#[tokio::test]
async fn should_reject_invalid_json_before_worker_admission() {
    let application = build_application(ScriptedResponsesExecutor::new(Vec::new()));

    let http_response = post_response_body(application, r#"{"model":"broken""#).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    assert_error_code(http_response, "invalid_json").await;
}

#[tokio::test]
async fn should_reject_a_model_that_is_not_loaded() {
    let application = build_application(ScriptedResponsesExecutor::new(Vec::new()));

    let http_response =
        post_response_body(application, r#"{"model":"another/model","input":"hello"}"#).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    assert_error_code(http_response, "model_not_found").await;
}

#[tokio::test]
async fn should_reject_stateful_response_storage_before_worker_admission() {
    let application = build_application(ScriptedResponsesExecutor::new(Vec::new()));

    let http_response = post_response_body(
        application,
        &format!(r#"{{"model":"{MODEL_ID}","input":"hello","store":true}}"#),
    )
    .await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    assert_error_code(http_response, "invalid_request").await;
}

#[tokio::test]
async fn should_return_too_many_requests_when_generation_capacity_is_active() {
    let application = build_application(ScriptedResponsesExecutor::with_start_error(
        GenerationStartError::CapacityUnavailable,
    ));

    let http_response = post_response(application, false).await;

    assert_eq!(http_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_error_code(http_response, "server_capacity").await;
}

#[tokio::test]
async fn should_return_payload_too_large_when_a_responses_request_exceeds_the_ipc_frame() {
    let application = build_application(ScriptedResponsesExecutor::with_start_error(
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes: 33_554_433,
            maximum_ipc_message_bytes: 33_554_432,
        },
    ));

    let http_response = post_response(application, false).await;

    assert_eq!(http_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_error_code(http_response, "request_too_large").await;
}

#[tokio::test]
async fn should_explain_why_the_requested_responses_model_could_not_be_loaded() {
    let application = build_application(ScriptedResponsesExecutor::with_start_error(
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason: "OptiQ metadata uses unsupported 2-bit quantization"
                .to_owned(),
        },
    ));

    let http_response = post_response(application, false).await;

    assert_eq!(http_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let response_body = to_bytes(http_response.into_body(), 8 * 1024)
        .await
        .expect("the model-load failure response should be readable");
    let response_text = String::from_utf8(response_body.to_vec())
        .expect("the model-load failure response should contain UTF-8");
    assert!(response_text.contains(r#""code":"model_load_failed""#));
    assert!(response_text.contains(
        r#""message":"the requested model could not be loaded: OptiQ metadata uses unsupported 2-bit quantization""#
    ));
}

#[tokio::test]
async fn should_canonicalize_a_provider_prefixed_model_for_an_idle_worker() {
    let application = build_application_with_config_warning_and_discovered_models(
        ScriptedResponsesExecutor::idle_with_expected_model("requested-model"),
        None,
        vec![DiscoveredModel {
            model_id: "requested-model".to_owned(),
            model_family: astronomical_config::ModelFamily::Qwen3_5,
            revision: "test-revision".to_owned(),
            model_directory: PathBuf::from("/models/requested-model"),
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            has_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
            model_size_bytes: 0,
        }],
    );

    let http_response = post_response_body(
        application,
        r#"{"model":"mlx-community/requested-model","input":"hello"}"#,
    )
    .await;

    assert_eq!(http_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_error_code(http_response, "server_capacity").await;
}

#[tokio::test]
async fn should_return_context_length_exceeded_for_a_non_streaming_response() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 262_145,
                maximum_context_tokens: 262_144,
            },
        },
    ]));

    let http_response = post_response(application, false).await;

    assert_eq!(http_response.status(), StatusCode::BAD_REQUEST);
    let response_body = to_bytes(http_response.into_body(), 16 * 1024)
        .await
        .expect("the context error body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the context error should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "context_length_exceeded"
    );
    assert_eq!(response_document["error"]["param"], "input");
}
