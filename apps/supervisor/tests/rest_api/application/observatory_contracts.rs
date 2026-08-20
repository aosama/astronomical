use astronomical_ipc_protocol::{
    ChatModelCapabilities, MlxMemorySnapshotSource, MtpRuntimeState,
    SpeculativePrefillRuntimeState, WorkerMlxMemorySnapshot,
};
use astronomical_supervisor::{
    ActiveRequestProgress, ExpertResidencySnapshot, WorkerActivity, WorkerHealthSnapshot,
    build_application, parse_macos_memory_pressure_level,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tokio::time::{Duration, Instant};
use tower::ServiceExt;

const OBSERVATORY_CONTRACT_MODEL_ID: &str = "astronomical/observatory-contract-model";

pub(super) fn ready_health_snapshot_with_model() -> WorkerHealthSnapshot {
    WorkerHealthSnapshot::ready_with_model(
        OBSERVATORY_CONTRACT_MODEL_ID.to_owned(),
        ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            context_window: 262_144,
        },
        MtpRuntimeState::Disabled,
        None,
    )
}

#[tokio::test]
async fn should_expose_ready_model_id_and_serving_session_in_status_when_ready_and_idle() {
    let application = build_application(ContractScriptedExecutor::ready(
        ready_health_snapshot_with_model(),
    ));
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
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["activity"], "idle");
    assert_eq!(
        status_document["ready_model_id"],
        OBSERVATORY_CONTRACT_MODEL_ID
    );
    assert_eq!(status_document["speculative_prefill_enabled"], false);
    assert_eq!(
        status_document["speculative_prefill_runtime_state"],
        "disabled"
    );
    // The Observatory UI must never assume progress is present while idle.
    assert!(
        status_document.get("progress").is_none() || status_document["progress"].is_null(),
        "idle status should not carry a progress object"
    );
    assert!(status_document["serving_session"].is_object());
    assert!(status_document["persistent_prompt_cache"].is_object());
}

#[tokio::test]
async fn should_expose_active_speculative_prefill_identity_in_status() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.speculative_prefill_runtime_state = SpeculativePrefillRuntimeState::Active;
    health_snapshot.speculative_prefill_draft_model_id =
        Some("astronomical/speculative-draft".to_owned());
    health_snapshot.speculative_prefill_draft_model_revision = Some("draft-revision-1".to_owned());
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(
        status_document["speculative_prefill_runtime_state"],
        "active"
    );
    assert_eq!(
        status_document["speculative_prefill_draft_model_id"],
        "astronomical/speculative-draft"
    );
    assert_eq!(
        status_document["speculative_prefill_draft_model_revision"],
        "draft-revision-1"
    );
}

#[tokio::test]
async fn should_expose_unavailable_speculative_prefill_reason_in_status() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.speculative_prefill_runtime_state = SpeculativePrefillRuntimeState::Unavailable;
    health_snapshot.speculative_prefill_unavailable_reason =
        Some("draft model could not be materialized".to_owned());
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(
        status_document["speculative_prefill_runtime_state"],
        "unavailable"
    );
    assert_eq!(
        status_document["speculative_prefill_unavailable_reason"],
        "draft model could not be materialized"
    );
}

#[tokio::test]
async fn should_expose_prefill_progress_with_processed_and_total_tokens_when_active() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.activity = WorkerActivity::PromptProcessing;
    health_snapshot.active_request_progress = Some(ActiveRequestProgress::Prefill {
        prompt_processing_phase: astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target,
        processed_tokens: 1_024,
        total_tokens: 8_192,
        elapsed_millis: 800,
        request_started_at: Instant::now(),
        completed_prefill_chunk_tokens: Some(1_024),
    });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(status_document["activity"], "prompt_processing");
    assert_eq!(status_document["progress"]["phase"], "target");
    assert_eq!(status_document["progress"]["processed_tokens"], 1_024);
    assert_eq!(status_document["progress"]["total_tokens"], 8_192);
    assert_eq!(status_document["progress"]["elapsed_ms"], 800);
    assert_eq!(
        status_document["progress"]["completed_prefill_chunk_tokens"],
        1_024
    );
}

#[tokio::test]
async fn should_advance_live_prefill_elapsed_time_before_the_first_completed_forward() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.activity = WorkerActivity::PromptProcessing;
    health_snapshot.active_request_progress = Some(ActiveRequestProgress::Prefill {
        prompt_processing_phase: astronomical_ipc_protocol::WorkerPromptProcessingPhase::Drafter,
        processed_tokens: 0,
        total_tokens: 8_192,
        elapsed_millis: 0,
        request_started_at: Instant::now() - Duration::from_millis(250),
        completed_prefill_chunk_tokens: None,
    });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));

    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(status_document["progress"]["phase"], "drafter");
    assert_eq!(status_document["progress"]["processed_tokens"], 0);
    assert!(
        status_document["progress"]["elapsed_ms"]
            .as_u64()
            .unwrap_or(0)
            >= 250
    );
}

#[tokio::test]
async fn should_expose_generation_progress_with_processed_and_total_tokens_when_active() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.activity = WorkerActivity::Generating;
    health_snapshot.active_request_progress = Some(ActiveRequestProgress::Generation {
        generated_token_count: 42,
        maximum_output_tokens: 512,
        elapsed_millis: 3_200,
    });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    assert_eq!(status_document["activity"], "generating");
    assert_eq!(status_document["progress"]["phase"], "generation");
    assert_eq!(status_document["progress"]["processed_tokens"], 42);
    assert_eq!(status_document["progress"]["total_tokens"], 512);
    assert_eq!(status_document["progress"]["elapsed_ms"], 3_200);
    // Generation progress never carries completed_prefill_chunk_tokens.
    assert!(
        status_document["progress"]
            .get("completed_prefill_chunk_tokens")
            .is_none()
            || status_document["progress"]["completed_prefill_chunk_tokens"].is_null()
    );
}

#[tokio::test]
async fn should_expose_generation_preparation_without_inventing_token_progress() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    let request_started_at = Instant::now() - Duration::from_millis(500);
    let preparation_started_at = Instant::now() - Duration::from_millis(125);
    health_snapshot.activity = WorkerActivity::GenerationPreparation;
    health_snapshot.expert_residency = Some(ExpertResidencySnapshot {
        total_layer_count: 40,
        complete_layer_count: 24,
        complete_layer_payload_bytes: 12_000_000_000,
        partial_layer_count: 8,
        partial_layer_payload_bytes: 1_000_000_000,
    });
    health_snapshot.active_request_progress = Some(ActiveRequestProgress::GenerationPreparation {
        request_started_at,
        preparation_started_at,
        total_layer_count: 40,
        complete_layer_count: 24,
        partial_layer_count: 8,
    });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the preparation status request should be valid"),
        )
        .await
        .expect("the application should return preparation status");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the preparation status body should be readable");
    let status_document: serde_json::Value = serde_json::from_slice(&response_body)
        .expect("the preparation status body should contain JSON");

    assert_eq!(status_document["activity"], "generation_preparation");
    assert_eq!(
        status_document["progress"]["phase"],
        "generation_preparation"
    );
    assert_eq!(status_document["progress"]["processed_tokens"], 0);
    assert_eq!(status_document["progress"]["total_tokens"], 1);
    assert_eq!(
        status_document["expert_residency"]["complete_layer_count"],
        24
    );
    assert_eq!(
        status_document["expert_residency"]["partial_layer_count"],
        8
    );
}

#[tokio::test]
async fn should_zero_fill_cache_stats_when_no_worker_data_so_the_ui_never_sees_missing_fields() {
    let application = build_application(ContractScriptedExecutor::ready(
        ready_health_snapshot_with_model(),
    ));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/cache/stats")
                .body(Body::empty())
                .expect("the cache-stats request should be valid"),
        )
        .await
        .expect("the application should return a cache-stats response");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body = to_bytes(response.into_body(), 8 * 1024)
        .await
        .expect("the cache-stats body should be readable");
    let cache_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the cache-stats body should contain JSON");

    assert_eq!(cache_document["persistent_prompt_cache_hits"], 0);
    assert_eq!(cache_document["persistent_prompt_cache_misses"], 0);
    assert_eq!(cache_document["persistent_prompt_cache_tokens_saved"], 0);
    assert_eq!(cache_document["persistent_prompt_cache_hit_rate"], 0.0);
    assert_eq!(
        cache_document["persistent_prompt_cache_total_size_bytes"],
        0
    );
    assert_eq!(
        cache_document["persistent_prompt_cache_maximum_size_bytes"],
        0
    );
    // Every field the Observatory cache panel reads must be present.
    assert!(cache_document["persistent_prompt_cache_sequence_state_block_count"].is_u64());
    assert!(cache_document["persistent_prompt_cache_boundary_state_snapshot_count"].is_u64());
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["target"]["eligible_token_count"],
        0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["target"]["restored_token_count"],
        0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["target"]["reuse_rate"],
        0.0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["drafter"]["eligible_token_count"],
        0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["drafter"]["restored_token_count"],
        0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["drafter"]["reuse_rate"],
        0.0
    );
    assert_eq!(
        cache_document["speculative_prefill_cache_efficacy"]["combined"]["reuse_rate"],
        0.0
    );
}

#[tokio::test]
async fn should_expose_gpu_utilization_and_memory_pressure_through_system_telemetry() {
    let application = build_application(ContractScriptedExecutor::ready(
        ready_health_snapshot_with_model(),
    ));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/system/telemetry")
                .body(Body::empty())
                .expect("the telemetry request should be valid"),
        )
        .await
        .expect("the application should return a telemetry response");
    assert_eq!(response.status(), StatusCode::OK);
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the telemetry body should be readable");
    let telemetry_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the telemetry body should contain JSON");
    // The field must always be present. On Apple Silicon with a GPU it is a
    // number in 0–100; on machines without an AGX accelerator it is null.
    assert!(
        telemetry_document["gpu_utilization_percentage"].is_number()
            || telemetry_document["gpu_utilization_percentage"].is_null(),
        "gpu_utilization_percentage must be a number or null"
    );
    if let Some(gpu_percentage) = telemetry_document["gpu_utilization_percentage"].as_f64() {
        assert!(
            (0.0..=100.0).contains(&gpu_percentage),
            "gpu_utilization_percentage must be in 0–100 range"
        );
    }
    assert!(
        telemetry_document["memory_pressure"].is_null()
            || telemetry_document["memory_pressure"]
                .as_str()
                .is_some_and(|memory_pressure| {
                    matches!(memory_pressure, "normal" | "warning" | "critical")
                }),
        "memory_pressure must be null, normal, warning, or critical"
    );
}

#[test]
fn should_parse_macos_memory_pressure_bitmasks_without_treating_unknown_values_as_normal() {
    let memory_pressure_cases = [
        ("1", Some("normal")),
        ("2", Some("warning")),
        ("4", Some("critical")),
        ("6", Some("critical")),
        ("3", Some("warning")),
        ("0", None),
        ("8", None),
        ("not-a-pressure-level", None),
    ];

    for (sysctl_value_text, expected_memory_pressure_level) in memory_pressure_cases {
        assert_eq!(
            parse_macos_memory_pressure_level(sysctl_value_text),
            expected_memory_pressure_level,
            "unexpected pressure mapping for {sysctl_value_text}"
        );
    }
}

#[tokio::test]
async fn should_expose_mlx_memory_snapshot_and_serving_session_in_status_for_the_memory_panel() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot.latest_mlx_memory_snapshot = Some(WorkerMlxMemorySnapshot {
        source: MlxMemorySnapshotSource::SpeculativePrefillDraftScoring,
        active_memory_bytes: 12_000,
        allocator_cache_memory_bytes: 500,
        peak_memory_bytes: 13_000,
        expert_payload_bytes: 2_000,
        model_core_payload_bytes: 4_000,
        context_state_payload_bytes: 1_000,
        speculative_prefill_draft_memory_bytes: 5_000,
    });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");
    // MLX memory snapshot is optional (None when no worker observation yet)
    // but the ceiling must always be present.
    assert!(status_document["mlx_memory_ceiling_bytes"].is_u64());
    assert_eq!(
        status_document["mlx_memory_snapshot"]["speculative_prefill_draft_memory_bytes"],
        5_000
    );
    // serving_session must always be present for the session panel.
    assert!(status_document["serving_session"].is_object());
    assert!(status_document["serving_session"]["completed_request_count"].is_u64());
    assert!(status_document["serving_session"]["total_prompt_token_count"].is_u64());
    assert!(status_document["serving_session"]["total_reused_prompt_token_count"].is_u64());
}

pub(super) struct ContractScriptedExecutor {
    health_snapshot: WorkerHealthSnapshot,
}

impl ContractScriptedExecutor {
    pub(super) fn ready(health_snapshot: WorkerHealthSnapshot) -> Self {
        Self { health_snapshot }
    }
}

impl astronomical_supervisor::ChatGenerationExecutor for ContractScriptedExecutor {
    fn start_chat_generation(
        &self,
        _generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        tokio::sync::mpsc::Receiver<
                            astronomical_supervisor::ChatGenerationStreamEvent,
                        >,
                        astronomical_supervisor::GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(
            async move { Err(astronomical_supervisor::GenerationStartError::WorkerUnavailable) },
        )
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        self.health_snapshot.clone()
    }
}

impl astronomical_supervisor::ImageGenerationExecutor for ContractScriptedExecutor {}
