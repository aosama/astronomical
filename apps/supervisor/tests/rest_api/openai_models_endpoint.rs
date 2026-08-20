use std::{fs, path::PathBuf};

use astronomical_config::{
    ChatModelCapabilities as DiscoveredChatModelCapabilities, DiscoveredModel,
    ImageGenerationCapabilities, ModelCapabilities,
};
use astronomical_ipc_protocol::{ChatModelCapabilities, WorkerModelCapabilities};
use astronomical_supervisor::{build_application, build_application_with_discovered_models};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use crate::common::{MODEL_ID, ScriptedExecutor};

const DISCOVERED_VISION_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";

#[tokio::test]
async fn should_list_the_independent_token_limits_produced_by_model_discovery() {
    let model_root = tempfile::tempdir().expect("a temporary model root should be created");
    let model_directory = model_root.path().join("Discovered-Qwen-Model");
    fs::create_dir(&model_directory).expect("the model directory should be created");
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#,
    )
    .expect("the model configuration should be written");
    fs::write(model_directory.join("tokenizer.json"), r#"{"version":1}"#)
        .expect("the tokenizer metadata should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{}}"#,
    )
    .expect("the model index should be written");
    let discovered_models =
        astronomical_config::discover_models(&[model_root.path().to_path_buf()])
            .expect("production model discovery should complete")
            .into_iter()
            .flat_map(|directory_scan| directory_scan.discovered_models)
            .collect();
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        discovered_models,
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
    let advertised_model = &model_list_document["data"][0];
    assert_eq!(advertised_model["id"], "Discovered-Qwen-Model");
    assert_eq!(advertised_model["context_window"], 262_144);
    assert_eq!(advertised_model["max_input_tokens"], 262_143);
    assert_eq!(advertised_model["max_output_tokens"], u16::MAX);
}

#[tokio::test]
async fn should_list_complete_capabilities_for_a_ready_worker_model() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.ready_model_capabilities =
        Some(WorkerModelCapabilities::from(ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 262_143,
            max_output_tokens: 20_480,
            context_window: 262_144,
        }));
    let application = build_application(scripted_executor);
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
    assert_eq!(advertised_model["context_window"], 262_144);
    assert_eq!(advertised_model["max_input_tokens"], 262_143);
    assert_eq!(advertised_model["max_output_tokens"], 20_480);
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
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
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
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
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
    assert_eq!(model_list_document["data"][0]["context_window"], 262_144);
    assert_eq!(model_list_document["data"][0]["max_input_tokens"], 262_143);
    assert_eq!(
        model_list_document["data"][0]["max_output_tokens"],
        u16::MAX
    );
}

#[tokio::test]
async fn should_list_image_model_metadata_without_autoregressive_token_limits() {
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        vec![discovered_image_model()],
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
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    let model_list_document: serde_json::Value =
        serde_json::from_slice(&models_body).expect("the model-list body should contain JSON");
    let advertised_model = &model_list_document["data"][0];

    assert_eq!(advertised_model["id"], "FLUX.2-klein-4B");
    assert_eq!(
        advertised_model["input_modalities"],
        serde_json::json!(["text"])
    );
    assert_eq!(
        advertised_model["output_modalities"],
        serde_json::json!(["image"])
    );
    assert_eq!(
        advertised_model["supported_endpoints"],
        serde_json::json!(["/v1/images/generations"])
    );
    assert!(advertised_model.get("context_window").is_none());
    assert!(advertised_model.get("max_input_tokens").is_none());
    assert!(advertised_model.get("max_output_tokens").is_none());
}

#[tokio::test]
async fn should_project_family_derived_reasoning_and_tool_capabilities() {
    let mut discovered_model = discovered_model_with_vision_support(false);
    let ModelCapabilities::Chat(capabilities) = &mut discovered_model.capabilities else {
        panic!("the fixture should be a chat model");
    };
    capabilities.supports_reasoning = false;
    capabilities.supports_tool_calls = false;
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        vec![discovered_model],
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
    let models_body = to_bytes(models_response.into_body(), 8 * 1024)
        .await
        .expect("the model-list body should be readable");
    let model_list_document: serde_json::Value =
        serde_json::from_slice(&models_body).expect("the model-list body should contain JSON");
    let advertised_model = &model_list_document["data"][0];

    assert_eq!(advertised_model["supports_reasoning"], false);
    assert!(advertised_model["reasoning_format"].is_null());
    assert_eq!(advertised_model["supports_tool_calls"], false);
    assert!(advertised_model["tool_call_format"].is_null());
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
    let ModelCapabilities::Chat(capabilities) = &mut invalid_discovered_model.capabilities else {
        panic!("the fixture should be a chat model");
    };
    capabilities.max_input_tokens = capabilities.context_window;
    capabilities.max_output_tokens = 1;
    let application = build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
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
    const CONTEXT_WINDOW: u32 = 262_144;
    DiscoveredModel {
        model_id: DISCOVERED_VISION_MODEL_ID.to_owned(),
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from("/tmp/astronomical-discovered-vision-model"),
        capabilities: ModelCapabilities::Chat(DiscoveredChatModelCapabilities {
            context_window: CONTEXT_WINDOW,
            max_input_tokens: CONTEXT_WINDOW - 1,
            max_output_tokens: u32::from(u16::MAX),
            supports_vision: has_vision,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 0,
    }
}

fn discovered_image_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "FLUX.2-klein-4B".to_owned(),
        provider_model_id: Some("black-forest-labs/FLUX.2-klein-4B".to_owned()),
        model_family: astronomical_config::ModelFamily::Flux2Klein,
        revision: "reviewed-revision".to_owned(),
        model_directory: PathBuf::from("/tmp/fictional-flux-model"),
        capabilities: ModelCapabilities::ImageGeneration(ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing: false,
            supports_multiple_reference_images: false,
        }),
        license: Some(astronomical_config::ModelLicense::Apache20),
        model_size_bytes: 0,
    }
}
