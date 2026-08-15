use astronomical_config::DiscoveredModel;
use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, MtpRuntimeState,
};
use astronomical_supervisor::{
    ActiveRequestProgress, ChatGenerationStreamEvent, WorkerActivity, build_application,
    build_application_with_config_warning_and_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tower::ServiceExt;

use super::support::{extract_tool_call_id, get_status, post_chat, post_chat_with_message};
use crate::common::{MODEL_ID, ScriptedExecutor};

#[tokio::test]
async fn should_report_health_readiness_and_the_loaded_model() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    assert_eq!(get_status(&application, "/health").await, StatusCode::OK);
    assert_eq!(get_status(&application, "/ready").await, StatusCode::OK);

    let models_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be valid"),
        )
        .await
        .expect("the application should return a model list");
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    let models_text = String::from_utf8_lossy(&models_body);
    assert!(models_text.contains(MODEL_ID));
    assert!(models_text.contains(r#""context_window":262144"#));
    assert!(models_text.contains(r#""max_input_tokens":241664"#));
    assert!(models_text.contains(r#""max_output_tokens":20480"#));
}

#[tokio::test]
async fn should_stream_openai_chat_outputs_and_done() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ReasoningFragment("inspect first".to_owned()),
        ChatGenerationStreamEvent::TextFragment("done".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3,
            generated_token_count: 2,
            reasoning_token_count: 1,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let response_body = post_chat(application).await;

    assert!(response_body.contains(r#""reasoning_content":"inspect first""#));
    assert!(response_body.contains(r#""content":"done""#));
    assert!(response_body.contains(r#""finish_reason":"stop""#));
    assert!(response_body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn should_keep_the_openai_stream_open_after_internal_prefill_progress() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::PrefillProgress {
            processed_tokens: 2_048,
            total_tokens: 19_485,
            elapsed_millis: 1_300,
            forward_prefill_chunk_elapsed_millis: Some(1_200),
            completed_prefill_chunk_tokens: Some(2_048),
            mlx_active_memory_bytes: Some(22_164_699_392),
            mlx_allocator_cache_memory_bytes: Some(0),
            mlx_peak_memory_bytes: Some(24_754_436_684),
        },
        ChatGenerationStreamEvent::TextFragment("still connected".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 19_485,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 8_192,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let response_body = post_chat(application).await;

    assert!(response_body.contains(r#""content":"still connected""#));
    assert!(response_body.contains(r#""finish_reason":"stop""#));
    assert!(response_body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn should_accept_a_streaming_chat_request_with_one_large_user_message() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let large_user_message = "x".repeat(128 * 1024);

    let response_body = post_chat_with_message(application, &large_user_message).await;

    assert!(response_body.contains(r#""finish_reason":"stop""#));
    assert!(response_body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn should_timestamp_openai_chat_chunks_with_current_unix_time() {
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let response_body = post_chat(application).await;

    assert!(!response_body.contains(r#""created":0"#));
}

#[tokio::test]
async fn should_emit_meaningful_openai_errors_for_each_stream_failure() {
    let failure_cases = [
        (
            ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens: 262_145,
                    maximum_context_tokens: 262_144,
                },
            },
            "context_length_exceeded",
            "requested context uses 262145 tokens, exceeding the 262144-token model context window",
        ),
        (
            ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::InvalidRequest {
                    reason: "rendered prompt exceeds the 262144-byte worker limit".to_owned(),
                },
            },
            "chat_invalid_request",
            "the local worker rejected the chat request: rendered prompt exceeds the 262144-byte worker limit",
        ),
        (
            ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::EngineBusy,
            },
            "chat_engine_busy",
            "the local inference engine is already processing another request",
        ),
        (
            ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::MalformedModelOutput,
            },
            "chat_malformed_model_output",
            "the model produced malformed structured output",
        ),
        (
            ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::FatalExecution {
                    reason: "GPU allocation exceeded the platform buffer limit while evaluating the model; reduce the prompt size or configured prefill chunk size".to_owned(),
                },
            },
            "chat_worker_unavailable",
            "the local worker stopped after a fatal model execution error: GPU allocation exceeded the platform buffer limit while evaluating the model; reduce the prompt size or configured prefill chunk size",
        ),
        (
            ChatGenerationStreamEvent::Error(
                astronomical_supervisor::ChatGenerationStreamErrorCode::WorkerUnavailable,
            ),
            "chat_worker_unavailable",
            "the local worker became unavailable while processing the chat request",
        ),
    ];

    for (stream_failure, expected_code, expected_message) in failure_cases {
        let application = build_application(ScriptedExecutor::ready(vec![stream_failure]));
        let response_body = post_chat(application).await;

        assert!(response_body.contains(&format!(r#""code":"{expected_code}""#)));
        assert!(response_body.contains(&format!(r#""message":"{expected_message}""#)));
        assert!(!response_body.contains("[DONE]"));
    }
}

#[tokio::test]
async fn should_not_reuse_tool_call_ids_across_application_restarts() {
    let tool_call_stream = || {
        vec![ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        }]
    };

    let first_response = post_chat(build_application(ScriptedExecutor::ready(
        tool_call_stream(),
    )))
    .await;
    let second_response = post_chat(build_application(ScriptedExecutor::ready(
        tool_call_stream(),
    )))
    .await;

    let first_tool_call_id = extract_tool_call_id(&first_response);
    let second_tool_call_id = extract_tool_call_id(&second_response);
    assert_ne!(first_tool_call_id, second_tool_call_id);
}

#[tokio::test]
async fn should_send_tool_call_arguments_with_secret_bearing_lines_unredacted_on_the_wire() {
    let arguments_with_secret_line =
        r#"{"command":"export api_key=sk-secret-123\nls -la"}"#.to_owned();
    let application = build_application(ScriptedExecutor::ready(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "bash".to_owned(),
            arguments_json: arguments_with_secret_line.clone(),
        },
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 1,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        },
    ]));
    let response_body = post_chat(application).await;

    assert!(
        response_body.contains("api_key=sk-secret-123"),
        "the actual secret-bearing command line must reach the client unredacted, got: {response_body}"
    );
    assert!(
        !response_body.contains("[REDACTED"),
        "no redaction marker must leak onto the wire, got: {response_body}"
    );
}

#[tokio::test]
async fn should_not_expose_the_removed_legacy_text_route() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let legacy_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/text/generations")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"prompt":"hello"}"#))
                .expect("the legacy-route probe should be valid"),
        )
        .await
        .expect("the application should return a response");
    assert_eq!(legacy_response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_report_ready_status_idle_activity_and_model_id_for_a_ready_worker() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mlx_memory_ceiling_bytes = 40_000_000_000;
    let application = build_application_with_config_warning_and_discovered_models(
        scripted_executor,
        None,
        vec![DiscoveredModel {
            model_id: MODEL_ID.to_owned(),
            model_family: astronomical_config::ModelFamily::Qwen3_5,
            revision: "test".to_owned(),
            model_directory: "/models/test".into(),
            context_window: 262_144,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            has_vision: true,
            model_size_bytes: 18_420_000_000,
        }],
    );
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    assert_eq!(response.status(), StatusCode::OK);
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status body should contain JSON");
    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["activity"], "idle");
    assert_eq!(status_document["mtp_enabled"], false);
    assert_eq!(status_document["ready_model_id"], MODEL_ID);
    assert!(status_document.get("expert_storage_format").is_none());
    assert_eq!(status_document["mtp_runtime_state"], "disabled");
    assert_eq!(
        status_document["mtp_unavailable_reason"],
        serde_json::Value::Null
    );
    assert_eq!(
        status_document["ready_model_size_bytes"],
        18_420_000_000_u64
    );
    assert_eq!(
        status_document["mlx_memory_snapshot"],
        serde_json::Value::Null
    );
    assert!(
        status_document
            .get("last_request_mlx_active_memory_bytes")
            .is_none()
    );
    assert_eq!(
        status_document["mlx_memory_ceiling_bytes"],
        40_000_000_000_u64
    );
    assert_eq!(
        status_document["serving_session"]["completed_request_count"],
        0
    );
    assert_eq!(
        status_document["serving_session"]["total_prompt_token_count"],
        0
    );
    assert_eq!(
        status_document["serving_session"]["total_reused_prompt_token_count"],
        0
    );
    assert_eq!(
        status_document["serving_session"]["average_prefill_tok_per_second"],
        0.0
    );
    assert_eq!(
        status_document["serving_session"]["average_generation_tok_per_second"],
        0.0
    );
    assert_eq!(status_document["persistent_prompt_cache"]["hits"], 0);
    assert_eq!(status_document["persistent_prompt_cache"]["misses"], 0);
    assert_eq!(
        status_document["persistent_prompt_cache"]["tokens_saved"],
        0
    );
    assert_eq!(status_document["persistent_prompt_cache"]["hit_rate"], 0.0);
    assert_eq!(
        status_document["config_warning"],
        serde_json::Value::Null,
        "a status response without a config warning must explicitly serialize the null field so the menu poller can distinguish absent from unset"
    );
}

#[tokio::test]
async fn should_report_target_only_mtp_runtime_state_in_status() {
    let status_document = mtp_status_document(MtpRuntimeState::TargetOnly, None).await;

    assert_eq!(status_document["mtp_runtime_state"], "target_only");
    assert_eq!(
        status_document["mtp_unavailable_reason"],
        serde_json::Value::Null
    );
}

#[tokio::test]
async fn should_report_active_mtp_runtime_state_without_an_unavailable_reason() {
    let status_document = mtp_status_document(MtpRuntimeState::Active, None).await;

    assert_eq!(status_document["mtp_runtime_state"], "active");
    assert_eq!(
        status_document["mtp_unavailable_reason"],
        serde_json::Value::Null
    );
}

#[tokio::test]
async fn should_report_unavailable_mtp_runtime_state_with_its_reason() {
    let status_document = mtp_status_document(
        MtpRuntimeState::Unavailable,
        Some("MTP sidecar tensor inventory is incomplete".to_owned()),
    )
    .await;

    assert_eq!(status_document["mtp_runtime_state"], "unavailable");
    assert_eq!(
        status_document["mtp_unavailable_reason"],
        "MTP sidecar tensor inventory is incomplete"
    );
}

async fn mtp_status_document(
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
) -> serde_json::Value {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mtp_runtime_state = mtp_runtime_state;
    scripted_executor.health_snapshot.mtp_unavailable_reason = mtp_unavailable_reason;
    let response = build_application(scripted_executor)
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    serde_json::from_slice(&status_body).expect("the status body should contain JSON")
}

#[tokio::test]
async fn should_report_the_ignored_fixed_prompt_processing_chunk_size_warning_in_status() {
    let application = astronomical_supervisor::build_application_with_config_warning(
        ScriptedExecutor::ready(Vec::new()),
        Some("Adaptive prompt-processing chunk-size selection is active. The configured fixed prompt-processing chunk size of 4096 tokens is ignored.".to_owned()),
    );
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status body should contain JSON");
    assert_eq!(status_document["status"], "ready");
    assert_eq!(
        status_document["config_warning"],
        "Adaptive prompt-processing chunk-size selection is active. The configured fixed prompt-processing chunk size of 4096 tokens is ignored."
    );
}

#[tokio::test]
async fn should_report_generating_activity_for_an_active_worker() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.health_snapshot.activity = WorkerActivity::Generating;
    let application = build_application(executor);
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status body should contain JSON");

    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["activity"], "generating");
}

#[tokio::test]
async fn should_report_completed_prefill_chunk_tokens_for_prompt_processing_progress() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.health_snapshot.activity = WorkerActivity::PromptProcessing;
    executor.health_snapshot.active_request_progress = Some(ActiveRequestProgress::Prefill {
        prompt_processing_phase: astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target,
        processed_tokens: 0,
        total_tokens: 2_200,
        elapsed_millis: 0,
        request_started_at: tokio::time::Instant::now(),
        completed_prefill_chunk_tokens: Some(2_048),
    });
    let application = build_application(executor);
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status body should contain JSON");

    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["activity"], "prompt_processing");
    assert_eq!(status_document["progress"]["phase"], "target");
    assert_eq!(
        status_document["progress"]["completed_prefill_chunk_tokens"],
        2_048
    );
}

#[tokio::test]
async fn should_omit_completed_prefill_chunk_tokens_before_the_first_measurement() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.health_snapshot.activity = WorkerActivity::PromptProcessing;
    executor.health_snapshot.active_request_progress = Some(ActiveRequestProgress::Prefill {
        prompt_processing_phase: astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target,
        processed_tokens: 0,
        total_tokens: 2_200,
        elapsed_millis: 0,
        request_started_at: tokio::time::Instant::now(),
        completed_prefill_chunk_tokens: None,
    });
    let application = build_application(executor);
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_body).expect("the status body should contain JSON");

    assert_eq!(status_document["progress"]["phase"], "target");
    assert!(
        status_document["progress"]
            .get("completed_prefill_chunk_tokens")
            .is_none(),
        "initial progress must not claim a chunk size before the engine measures one"
    );
}
