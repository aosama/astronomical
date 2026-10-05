//! Public status contracts for image activity stay separate from text Observatory contracts so
//! each cross-language progress shape remains small and directly reviewable.

use astronomical_ipc_protocol::ImageGenerationPhase;
use astronomical_supervisor::{ActiveRequestProgress, WorkerActivity};
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use tower::ServiceExt;

use super::observatory_contracts;
use super::observatory_contracts::ContractScriptedExecutor;

#[tokio::test]
async fn should_expose_image_generation_progress_with_completed_and_total_steps_when_active() {
    let mut health_snapshot = observatory_contracts::ready_health_snapshot_with_model();
    health_snapshot.activity = WorkerActivity::ImageGeneration;
    health_snapshot.active_request_progress = Some(ActiveRequestProgress::ImageGeneration {
        phase: ImageGenerationPhase::Denoising,
        completed_steps: 2,
        total_steps: 4,
        elapsed_millis: 1_000,
    });
    let application = astronomical_supervisor::build_application(ContractScriptedExecutor::ready(
        health_snapshot,
    ));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the image status request should be valid"),
        )
        .await
        .expect("the application should return image progress status");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the image progress status body should be readable");
    let status_document: serde_json::Value = serde_json::from_slice(&response_body)
        .expect("the image progress status body should contain JSON");

    assert_eq!(status_document["activity"], "image_generation");
    assert_eq!(status_document["progress"]["phase"], "denoising");
    assert_eq!(status_document["progress"]["completed_steps"], 2);
    assert_eq!(status_document["progress"]["total_steps"], 4);
    assert_eq!(status_document["progress"]["elapsed_ms"], 1_000);
}
