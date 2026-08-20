//! Acceptance coverage for the non-streaming OpenAI image-generation journey.

use std::{io::Cursor, path::PathBuf};

use astronomical_config::{
    DiscoveredModel, ImageGenerationCapabilities, ModelCapabilities, ModelFamily, ModelLicense,
};
use astronomical_supervisor::{
    ImageGenerationExecutionError, build_application_with_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{ColorType, ImageDecoder, codecs::png::PngDecoder};
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const IMAGE_MODEL_ID: &str = "FLUX.2-klein-4B";

#[tokio::test]
async fn should_generate_one_base64_png_through_the_public_http_journey() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_image_generation_commands();
    let application = build_application_with_discovered_models(executor, vec![image_model()]);

    let response = application
        .oneshot(image_request(valid_request_document()))
        .await
        .expect("the application should return an image response");

    assert_eq!(response.status(), StatusCode::OK);
    let response_document = response_json(response).await;
    let png_bytes = STANDARD
        .decode(
            response_document["data"][0]["b64_json"]
                .as_str()
                .expect("the image payload should be base64 text"),
        )
        .expect("the image payload should be valid base64");
    let png_decoder = PngDecoder::new(Cursor::new(png_bytes))
        .expect("the public image payload should be a decodable PNG");
    assert_eq!(png_decoder.dimensions(), (1_024, 1_024));
    assert_eq!(png_decoder.color_type(), ColorType::Rgb8);
    assert_eq!(response_document["data"][0]["mime_type"], "image/png");
    assert_eq!(
        response_document["data"][0]["model_revision"],
        "fixture-revision"
    );
    assert_eq!(response_document["data"][0]["seed"], 7);
    assert_eq!(response_document["data"][0]["width"], 1_024);
    assert_eq!(response_document["data"][0]["height"], 1_024);

    let commands = received_commands
        .lock()
        .expect("the image command log should remain available");
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].model, IMAGE_MODEL_ID);
    assert_eq!(commands[0].settings.seed, 7);
}

#[tokio::test]
async fn should_reject_every_unsupported_image_field_before_dispatch() {
    let invalid_requests = [
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": " ", "width": 1024, "height": 1024, "steps": 4, "guidance": 1.0, "response_format": "b64_json"}),
            "prompt",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 63, "height": 1024, "steps": 4, "guidance": 1.0, "response_format": "b64_json"}),
            "width",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 80, "height": 1023, "steps": 4, "guidance": 1.0, "response_format": "b64_json"}),
            "height",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 1024, "height": 1024, "steps": 5, "guidance": 1.0, "response_format": "b64_json"}),
            "steps",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 1024, "height": 1024, "steps": 4, "guidance": 1.1, "response_format": "b64_json"}),
            "guidance",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 1024, "height": 1024, "steps": 4, "guidance": 1.0, "response_format": "url"}),
            "response_format",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 1024, "height": 1024, "steps": 4, "guidance": 1.0, "response_format": "b64_json", "n": 2}),
            "n",
        ),
        (
            serde_json::json!({"model": IMAGE_MODEL_ID, "prompt": "Romeo", "width": 1024, "height": 1024, "steps": 4, "guidance": 1.0, "response_format": "b64_json", "quality": "hd"}),
            "request",
        ),
    ];

    for (request_document, expected_parameter) in invalid_requests {
        let executor = ScriptedExecutor::ready(Vec::new());
        let received_commands = executor.received_image_generation_commands();
        let response = build_application_with_discovered_models(executor, vec![image_model()])
            .oneshot(image_request(request_document))
            .await
            .expect("the application should reject invalid image input");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let response_document = response_json(response).await;
        assert_eq!(response_document["error"]["param"], expected_parameter);
        assert!(
            received_commands
                .lock()
                .expect("command log should remain available")
                .is_empty()
        );
    }
}

#[tokio::test]
async fn should_reject_a_chat_model_before_image_queue_admission() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_image_generation_commands();
    let application = build_application_with_discovered_models(executor, vec![chat_model()]);
    let mut request_document = valid_request_document();
    request_document["model"] = serde_json::json!("chat-only-model");

    let response = application
        .oneshot(image_request(request_document))
        .await
        .expect("the application should reject the wrong modality");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "model_capability_mismatch"
    );
    assert_eq!(response_document["error"]["param"], "model");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_reject_an_image_only_model_from_chat_before_queue_admission() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_generation_commands();
    let application = build_application_with_discovered_models(executor, vec![image_model()]);

    let response = application
        .oneshot(text_generation_request(
            "/v1/chat/completions",
            serde_json::json!({
                "model": IMAGE_MODEL_ID,
                "messages": [{"role": "user", "content": "Wherefore art thou Romeo?"}]
            }),
        ))
        .await
        .expect("the chat endpoint should reject the wrong modality");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "model_capability_mismatch"
    );
    assert_eq!(response_document["error"]["param"], "model");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_reject_an_image_only_model_from_responses_before_queue_admission() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_generation_commands();
    let application = build_application_with_discovered_models(executor, vec![image_model()]);

    let response = application
        .oneshot(text_generation_request(
            "/v1/responses",
            serde_json::json!({
                "model": IMAGE_MODEL_ID,
                "input": "Wherefore art thou Romeo?"
            }),
        ))
        .await
        .expect("the Responses endpoint should reject the wrong modality");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "model_capability_mismatch"
    );
    assert_eq!(response_document["error"]["param"], "model");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_return_no_partial_image_when_worker_execution_fails() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.image_generation_outcome = Err(ImageGenerationExecutionError::WorkerFailure(
        astronomical_ipc_protocol::ImageGenerationFailureReason::EncodingFailed {
            reason: "PNG encoder rejected the completed RGB payload".to_owned(),
        },
    ));
    let application = build_application_with_discovered_models(executor, vec![image_model()]);

    let response = application
        .oneshot(image_request(valid_request_document()))
        .await
        .expect("the application should map the worker failure");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "image_generation_failed"
    );
    assert!(response_document["data"].is_null());
}

#[tokio::test]
async fn should_not_expose_worker_image_failure_reasons_or_local_paths() {
    let fictional_private_path = "/Users/fictional-person/private-model/decoder.safetensors";
    let failure_cases = [
        (
            astronomical_ipc_protocol::ImageGenerationFailureReason::InvalidRequest {
                reason: format!("could not read {fictional_private_path}"),
            },
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "the image request was rejected by the local worker",
        ),
        (
            astronomical_ipc_protocol::ImageGenerationFailureReason::EncodingFailed {
                reason: format!("PNG encoding failed near {fictional_private_path}"),
            },
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_generation_failed",
            "image generation failed in the local worker",
        ),
        (
            astronomical_ipc_protocol::ImageGenerationFailureReason::FatalExecution {
                reason: format!("native execution failed while mapping {fictional_private_path}"),
            },
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_generation_failed",
            "image generation failed in the local worker",
        ),
    ];

    for (worker_failure, expected_status, expected_code, expected_message) in failure_cases {
        let mut executor = ScriptedExecutor::ready(Vec::new());
        executor.image_generation_outcome =
            Err(ImageGenerationExecutionError::WorkerFailure(worker_failure));
        let response = build_application_with_discovered_models(executor, vec![image_model()])
            .oneshot(image_request(valid_request_document()))
            .await
            .expect("the application should sanitize the worker failure");

        assert_eq!(response.status(), expected_status);
        let response_document = response_json(response).await;
        assert_eq!(response_document["error"]["code"], expected_code);
        assert_eq!(response_document["error"]["message"], expected_message);
        let response_text = response_document.to_string();
        assert!(!response_text.contains(fictional_private_path));
        assert!(response_text.len() < 1_024);
    }
}

#[tokio::test]
async fn should_return_gateway_timeout_without_partial_image_when_the_execution_deadline_expires() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.image_generation_outcome = Err(ImageGenerationExecutionError::DeadlineExceeded);
    let application = build_application_with_discovered_models(executor, vec![image_model()]);

    let response = application
        .oneshot(image_request(valid_request_document()))
        .await
        .expect("the application should map the bounded deadline");

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let response_document = response_json(response).await;
    assert_eq!(
        response_document["error"]["code"],
        "image_generation_timeout"
    );
    assert!(response_document["data"].is_null());
}

#[tokio::test]
async fn should_reject_malformed_json_and_transport_overflow_before_dispatch() {
    let malformed_executor = ScriptedExecutor::ready(Vec::new());
    let malformed_commands = malformed_executor.received_image_generation_commands();
    let malformed_response =
        build_application_with_discovered_models(malformed_executor, vec![image_model()])
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"model\":"))
                    .expect("the malformed HTTP request should still be constructible"),
            )
            .await
            .expect("the application should reject malformed JSON");
    assert_eq!(malformed_response.status(), StatusCode::BAD_REQUEST);
    assert!(
        malformed_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );

    let oversized_executor = ScriptedExecutor::ready(Vec::new());
    let oversized_commands = oversized_executor.received_image_generation_commands();
    let oversized_prompt = "R".repeat(32 * 1024 * 1024);
    let oversized_response =
        build_application_with_discovered_models(oversized_executor, vec![image_model()])
            .oneshot(image_request(serde_json::json!({
                "model": IMAGE_MODEL_ID,
                "prompt": oversized_prompt,
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 1.0,
                "response_format": "b64_json"
            })))
            .await
            .expect("the application should reject the transport overflow");
    assert_eq!(oversized_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        oversized_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

fn valid_request_document() -> serde_json::Value {
    serde_json::json!({
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "prompt": "A moonlit balcony scene from Romeo and Juliet",
        "seed": 7,
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance": 1.0,
        "response_format": "b64_json"
    })
}

fn image_request(request_document: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the image request should be valid")
}

fn text_generation_request(uri: &str, request_document: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the text generation request should be valid")
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the bounded response body should be readable");
    serde_json::from_slice(&response_body).expect("the response should contain JSON")
}

fn image_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: IMAGE_MODEL_ID.to_owned(),
        provider_model_id: Some("black-forest-labs/FLUX.2-klein-4B".to_owned()),
        model_family: ModelFamily::Flux2Klein,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/flux2-klein"),
        capabilities: ModelCapabilities::ImageGeneration(ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing: false,
            supports_multiple_reference_images: false,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: 1,
    }
}

fn chat_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "chat-only-model".to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat-only"),
        capabilities: ModelCapabilities::Chat(astronomical_config::ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1,
    }
}
