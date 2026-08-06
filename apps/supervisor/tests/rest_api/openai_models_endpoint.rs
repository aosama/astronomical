use std::path::PathBuf;

use astronomical_config::DiscoveredModel;
use astronomical_supervisor::{
    build_application, build_application_with_config_warning_and_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use crate::common::{MODEL_ID, ScriptedExecutor};

const DISCOVERED_VISION_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";

#[tokio::test]
async fn should_list_complete_capabilities_for_a_ready_worker_model() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let models_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be valid"),
        )
        .await
        .expect("the application should return a model list");

    assert_eq!(models_response.status(), StatusCode::OK);
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    let model_list_document: serde_json::Value =
        serde_json::from_slice(&models_body).expect("the model-list body should contain JSON");
    let advertised_model = &model_list_document["data"][0];

    assert_eq!(advertised_model["id"], MODEL_ID);
    assert_eq!(
        advertised_model["input_modalities"],
        serde_json::json!(["text", "image"])
    );
    assert_eq!(
        advertised_model["output_modalities"],
        serde_json::json!(["text"])
    );
    assert_eq!(advertised_model["supports_streaming"], true);
    assert_eq!(advertised_model["supports_reasoning"], true);
    assert_eq!(
        advertised_model["reasoning_format"],
        "openai_chat_reasoning_content_and_responses_reasoning_summary_text"
    );
    assert_eq!(advertised_model["supports_tool_calls"], true);
    assert_eq!(advertised_model["tool_call_format"], "openai_function_call");
    assert_eq!(
        advertised_model["supported_endpoints"],
        serde_json::json!(["/v1/chat/completions", "/v1/responses"])
    );
}

#[tokio::test]
async fn should_get_a_discovered_model_by_provider_prefixed_id() {
    let application = build_application_with_config_warning_and_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        None,
        vec![discovered_model_with_vision_support(true)],
    );
    let model_response = application
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/models/mlx-community/{DISCOVERED_VISION_MODEL_ID}"
                ))
                .body(Body::empty())
                .expect("the model-detail request should be valid"),
        )
        .await
        .expect("the application should return a model-detail response");

    assert_eq!(model_response.status(), StatusCode::OK);
    let model_body = to_bytes(model_response.into_body(), 8 * 1024)
        .await
        .expect("the model-detail body should be readable");
    let advertised_model: serde_json::Value =
        serde_json::from_slice(&model_body).expect("the model-detail body should contain JSON");

    assert_eq!(advertised_model["id"], DISCOVERED_VISION_MODEL_ID);
    assert_eq!(
        advertised_model["input_modalities"],
        serde_json::json!(["text", "image"])
    );
    assert_eq!(
        advertised_model["output_modalities"],
        serde_json::json!(["text"])
    );
    assert_eq!(
        advertised_model["supported_endpoints"],
        serde_json::json!(["/v1/chat/completions", "/v1/responses"])
    );
}

#[tokio::test]
async fn should_list_text_only_input_modality_for_a_discovered_model_without_vision_support() {
    let application = build_application_with_config_warning_and_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        None,
        vec![discovered_model_with_vision_support(false)],
    );
    let models_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be valid"),
        )
        .await
        .expect("the application should return a model list");

    assert_eq!(models_response.status(), StatusCode::OK);
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    let model_list_document: serde_json::Value =
        serde_json::from_slice(&models_body).expect("the model-list body should contain JSON");

    assert_eq!(
        model_list_document["data"][0]["input_modalities"],
        serde_json::json!(["text"])
    );
    assert_eq!(
        model_list_document["data"][0]["output_modalities"],
        serde_json::json!(["text"])
    );
}

#[tokio::test]
async fn should_return_an_openai_model_not_found_error_for_an_unknown_model() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let model_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models/unknown-local-model")
                .body(Body::empty())
                .expect("the model-detail request should be valid"),
        )
        .await
        .expect("the application should return a model-detail response");

    assert_eq!(model_response.status(), StatusCode::NOT_FOUND);
    let model_response_body = to_bytes(model_response.into_body(), 8 * 1024)
        .await
        .expect("the model-detail body should be readable");
    assert_eq!(
        model_response_body.as_ref(),
        br#"{"error":{"message":"model is not loaded by the local worker","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#
    );
}

#[tokio::test]
async fn should_not_advertise_stale_model_metadata_when_the_worker_is_unavailable() {
    let application = build_application(ScriptedExecutor::unavailable());
    let models_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be valid"),
        )
        .await
        .expect("the application should return a model list");

    assert_eq!(models_response.status(), StatusCode::OK);
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    assert_eq!(models_body.as_ref(), br#"{"object":"list","data":[]}"#);
}

#[tokio::test]
async fn should_fail_closed_when_discovered_model_capabilities_are_invalid() {
    let mut invalid_discovered_model = discovered_model_with_vision_support(true);
    invalid_discovered_model.max_input_tokens = invalid_discovered_model.context_window;
    invalid_discovered_model.max_output_tokens = 1;
    let application = build_application_with_config_warning_and_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        None,
        vec![invalid_discovered_model],
    );
    let models_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("the model-list request should be valid"),
        )
        .await
        .expect("the application should return a model list");

    assert_eq!(models_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    assert_eq!(
        models_body.as_ref(),
        br#"{"error":{"message":"model capability metadata is unavailable","type":"server_error","code":"model_metadata_unavailable"}}"#
    );
}

fn discovered_model_with_vision_support(has_vision: bool) -> DiscoveredModel {
    DiscoveredModel {
        model_id: DISCOVERED_VISION_MODEL_ID.to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from("/tmp/astronomical-discovered-vision-model"),
        context_window: 262_144,
        max_input_tokens: 241_664,
        max_output_tokens: 20_480,
        has_vision,
        model_size_bytes: 0,
    }
}
