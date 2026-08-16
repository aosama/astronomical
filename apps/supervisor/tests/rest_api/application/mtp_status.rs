use astronomical_ipc_protocol::{
    MtpDepthStatus, MtpRuntimeState, WorkerRuntimeFeatureConfiguration,
};
use astronomical_supervisor::build_application;
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

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

#[tokio::test]
async fn should_report_distinct_requested_and_effective_mtp_depths() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mtp_runtime_state = MtpRuntimeState::Active;
    scripted_executor.health_snapshot.mtp_depth_status = MtpDepthStatus {
        configured_draft_depth: Some(3),
        artifact_maximum_draft_depth: Some(3),
        artifact_default_draft_depth: Some(2),
        resolved_requested_draft_depth: Some(3),
        effective_execution_draft_depth: Some(1),
    };

    let status_document = status_document(scripted_executor).await;

    assert_eq!(status_document["mtp_configured_draft_depth"], 3);
    assert_eq!(status_document["mtp_artifact_maximum_draft_depth"], 3);
    assert_eq!(status_document["mtp_artifact_default_draft_depth"], 2);
    assert_eq!(status_document["mtp_resolved_requested_draft_depth"], 3);
    assert_eq!(status_document["mtp_effective_execution_draft_depth"], 1);
}

#[tokio::test]
async fn should_report_worker_acknowledged_mtp_enablement_without_reloadable_config() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor
        .health_snapshot
        .worker_runtime_feature_configuration = Some(WorkerRuntimeFeatureConfiguration {
        persistent_prompt_cache_enabled: false,
        mtp_enabled: true,
        mtp_draft_depth: Some(2),
        speculative_prefill_enabled: false,
    });

    let status_document = status_document(scripted_executor).await;

    assert_eq!(status_document["mtp_enabled"], true);
    assert_eq!(status_document["mtp_configured_draft_depth"], 2);
}

async fn mtp_status_document(
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
) -> serde_json::Value {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mtp_runtime_state = mtp_runtime_state;
    scripted_executor.health_snapshot.mtp_unavailable_reason = mtp_unavailable_reason;
    status_document(scripted_executor).await
}

async fn status_document(scripted_executor: ScriptedExecutor) -> serde_json::Value {
    // Exercise the public HTTP boundary because status combines worker acknowledgement with the
    // optional reloadable configuration. Directly inspecting WorkerHealthSnapshot would miss the
    // exact regression where an active worker was serialized as MTP-disabled.
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
