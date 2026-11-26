//! Public status contracts for image activity stay separate from text Observatory contracts so
//! each cross-language progress shape remains small and directly reviewable.

use crate::rest_api::http_exchange_fixtures::{get_response, read_json_body};
use astronomical_daemon::{ActiveRequestProgress, WorkerActivity};
use ipc_protocol::ImageGenerationPhase;

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
    let application =
        astronomical_daemon::build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = get_response(&application, "/v1/status").await;
    let status_document = read_json_body(response, 16 * 1024).await;

    assert_eq!(status_document["activity"], "image_generation");
    assert_eq!(status_document["progress"]["phase"], "denoising");
    assert_eq!(status_document["progress"]["completed_steps"], 2);
    assert_eq!(status_document["progress"]["total_steps"], 4);
    assert_eq!(status_document["progress"]["elapsed_ms"], 1_000);
}
