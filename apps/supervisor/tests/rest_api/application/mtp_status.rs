use astronomical_ipc_protocol::{
    MtpDepthResolutionReason, MtpDepthStatus, MtpRuntimeState, WorkerChunkingConfiguration,
    WorkerLoadedAutoregressiveModelRuntimeConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerRuntimeFeatureConfiguration,
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
        artifact_maximum_draft_depth: Some(1),
        artifact_default_draft_depth: Some(1),
        resolved_requested_draft_depth: Some(3),
        capped_draft_depth: Some(1),
        effective_execution_draft_depth: Some(1),
        resolution_reason: Some(MtpDepthResolutionReason::ConfiguredDepthClampedToArtifactMaximum),
    };

    let status_document = status_document(scripted_executor).await;

    assert_eq!(status_document["mtp_configured_draft_depth"], 3);
    assert_eq!(status_document["mtp_artifact_maximum_draft_depth"], 1);
    assert_eq!(status_document["mtp_artifact_default_draft_depth"], 1);
    assert_eq!(status_document["mtp_resolved_requested_draft_depth"], 3);
    assert_eq!(status_document["mtp_capped_draft_depth"], 1);
    assert_eq!(status_document["mtp_effective_execution_draft_depth"], 1);
    assert_eq!(
        status_document["mtp_depth_resolution_reason"],
        "configured MTP draft depth was clamped to the declared artifact maximum"
    );
}

#[tokio::test]
async fn should_report_guidance_without_capping_an_explicit_depth_for_a_silent_artifact() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mtp_runtime_state = MtpRuntimeState::Active;
    scripted_executor.health_snapshot.mtp_depth_status = MtpDepthStatus {
        configured_draft_depth: Some(3),
        artifact_maximum_draft_depth: None,
        artifact_default_draft_depth: None,
        resolved_requested_draft_depth: Some(3),
        capped_draft_depth: Some(3),
        effective_execution_draft_depth: Some(3),
        resolution_reason: Some(MtpDepthResolutionReason::ConfiguredDepthExceedsAutomaticGuidance),
    };

    let status_document = status_document(scripted_executor).await;

    assert_eq!(
        status_document["mtp_artifact_maximum_draft_depth"],
        serde_json::Value::Null
    );
    assert_eq!(status_document["mtp_resolved_requested_draft_depth"], 3);
    assert_eq!(status_document["mtp_capped_draft_depth"], 3);
    assert_eq!(status_document["mtp_effective_execution_draft_depth"], 3);
    assert_eq!(
        status_document["mtp_depth_resolution_reason"],
        "configured MTP draft depth exceeds the automatic depth-one guidance"
    );
}

#[tokio::test]
async fn should_report_loaded_model_mtp_policy_without_reloadable_config() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor
        .health_snapshot
        .worker_runtime_feature_configuration = Some(WorkerRuntimeFeatureConfiguration {
        configuration_generation:
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 50_000_000_000,
        loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                model_id: "astronomical/test-worker".to_owned(),
                maximum_context_tokens: 2_048,
                maximum_output_tokens: 128,
                chunking: WorkerChunkingConfiguration {
                    fixed_prompt_processing_chunk_size_tokens: 256,
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                    full_attention_key_value_growth_tokens: 256,
                    speculative_prefill_draft_forward_tokens: 256,
                    prefill_graph_submission_layer_interval: 1,
                    experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                    prompt_cache_block_tokens: None,
                    prompt_cache_common_prefix_stride_blocks: 4,
                },
                mtp_enabled: true,
                mtp_draft_depth: Some(2),
                speculative_prefill_enabled: false,
                speculative_prefill: None,
            },
        )),
    });
    scripted_executor.health_snapshot.mtp_runtime_state = MtpRuntimeState::Active;

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
    let status_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    serde_json::from_slice(&status_body).expect("the status body should contain JSON")
}
