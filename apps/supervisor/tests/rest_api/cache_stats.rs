use astronomical_ipc_protocol::{ChatModelCapabilities, MtpRuntimeState, WorkerEvent};
use astronomical_supervisor::{WorkerHealthSnapshot, WorkerHealthStatus, build_application};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/cache-stats-test-model";
const TEST_CONFIGURED_PROMPT_CACHE_MAXIMUM_SIZE_BYTES: u64 = 123_456_789;

const POPULATED_PERSISTENT_PROMPT_CACHE_STATS_EVENT: WorkerEvent =
    WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_hits: 12,
        persistent_prompt_cache_misses: 3,
        persistent_prompt_cache_tokens_saved: 95_000,
        persistent_prompt_cache_sequence_state_block_count: 87,
        persistent_prompt_cache_boundary_state_snapshot_count: 1,
        persistent_prompt_cache_visual_embedding_count: 5,
        persistent_prompt_cache_total_size_bytes: 1_073_741_824,
        persistent_prompt_cache_visual_embedding_total_size_bytes: 222_222,
        persistent_prompt_cache_maximum_size_bytes: TEST_CONFIGURED_PROMPT_CACHE_MAXIMUM_SIZE_BYTES,
        persistent_prompt_cache_visual_embedding_hits: 4,
        persistent_prompt_cache_visual_embedding_misses: 2,
        persistent_prompt_cache_visual_embedding_rows_loaded: 256,
    };

#[tokio::test]
async fn should_return_populated_cache_stats_for_a_ready_worker_with_cache() {
    let mut health_snapshot = WorkerHealthSnapshot::ready_with_model(
        MODEL_ID.to_owned(),
        ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            context_window: 262_144,
        },
        astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
        MtpRuntimeState::Disabled,
        None,
    );
    health_snapshot.persistent_prompt_cache_stats =
        Some(POPULATED_PERSISTENT_PROMPT_CACHE_STATS_EVENT.clone());
    let application = build_application(StatsScriptedExecutor::ready(health_snapshot));
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
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the cache-stats body should contain JSON");
    assert_eq!(status_document["persistent_prompt_cache_hits"], 12);
    assert_eq!(status_document["persistent_prompt_cache_misses"], 3);
    assert_eq!(
        status_document["persistent_prompt_cache_tokens_saved"],
        95_000
    );
    assert_eq!(
        status_document["persistent_prompt_cache_sequence_state_block_count"],
        87
    );
    assert_eq!(
        status_document["persistent_prompt_cache_boundary_state_snapshot_count"],
        1
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_count"],
        5
    );
    assert_eq!(
        status_document["persistent_prompt_cache_total_size_bytes"],
        1_073_741_824_u64
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_total_size_bytes"],
        222_222
    );
    assert_eq!(
        status_document["persistent_prompt_cache_maximum_size_bytes"],
        TEST_CONFIGURED_PROMPT_CACHE_MAXIMUM_SIZE_BYTES
    );
    assert_eq!(status_document["persistent_prompt_cache_hit_rate"], 0.8);
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_hits"],
        4
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_misses"],
        2
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_rows_loaded"],
        256
    );
}

#[tokio::test]
async fn should_return_zeroed_cache_stats_when_worker_is_unavailable() {
    let health_snapshot = WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable);
    let application = build_application(StatsScriptedExecutor::ready(health_snapshot));
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
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the cache-stats body should contain JSON");
    assert_eq!(status_document["persistent_prompt_cache_hits"], 0);
    assert_eq!(status_document["persistent_prompt_cache_misses"], 0);
    assert_eq!(status_document["persistent_prompt_cache_tokens_saved"], 0);
    assert_eq!(
        status_document["persistent_prompt_cache_sequence_state_block_count"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_boundary_state_snapshot_count"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_count"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_total_size_bytes"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_total_size_bytes"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_maximum_size_bytes"],
        0
    );
    assert_eq!(status_document["persistent_prompt_cache_hit_rate"], 0.0);
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_hits"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_misses"],
        0
    );
    assert_eq!(
        status_document["persistent_prompt_cache_visual_embedding_rows_loaded"],
        0
    );
}

#[tokio::test]
async fn should_return_200_ok_with_json_content_type_for_cache_stats() {
    let health_snapshot = WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable);
    let application = build_application(StatsScriptedExecutor::ready(health_snapshot));
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
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .expect("the cache-stats response should have a content-type header");
    assert!(
        content_type
            .to_str()
            .expect("the content-type should be valid ASCII")
            .starts_with("application/json"),
        "the cache-stats response should be JSON"
    );
}

#[tokio::test]
async fn should_compute_hit_rate_as_hits_over_total_queries() {
    let mut health_snapshot = WorkerHealthSnapshot::ready_with_model(
        MODEL_ID.to_owned(),
        ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            context_window: 262_144,
        },
        astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
        MtpRuntimeState::Disabled,
        None,
    );
    health_snapshot.persistent_prompt_cache_stats = Some(WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_hits: 2,
        persistent_prompt_cache_misses: 1,
        persistent_prompt_cache_tokens_saved: 6_144,
        persistent_prompt_cache_sequence_state_block_count: 5,
        persistent_prompt_cache_boundary_state_snapshot_count: 1,
        persistent_prompt_cache_visual_embedding_count: 2,
        persistent_prompt_cache_total_size_bytes: 200_000_000,
        persistent_prompt_cache_visual_embedding_total_size_bytes: 50_000,
        persistent_prompt_cache_maximum_size_bytes: TEST_CONFIGURED_PROMPT_CACHE_MAXIMUM_SIZE_BYTES,
        persistent_prompt_cache_visual_embedding_hits: 0,
        persistent_prompt_cache_visual_embedding_misses: 0,
        persistent_prompt_cache_visual_embedding_rows_loaded: 0,
    });
    let application = build_application(StatsScriptedExecutor::ready(health_snapshot));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/cache/stats")
                .body(Body::empty())
                .expect("the cache-stats request should be valid"),
        )
        .await
        .expect("the application should return a cache-stats response");
    let response_body = to_bytes(response.into_body(), 8 * 1024)
        .await
        .expect("the cache-stats body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the cache-stats body should contain JSON");
    // 2 hits / 3 total = 0.6667 rounded to 4 decimals
    assert_eq!(status_document["persistent_prompt_cache_hit_rate"], 0.6667);
}

struct StatsScriptedExecutor {
    health_snapshot: WorkerHealthSnapshot,
}

impl StatsScriptedExecutor {
    fn ready(health_snapshot: WorkerHealthSnapshot) -> Self {
        Self { health_snapshot }
    }
}

impl astronomical_supervisor::ChatGenerationExecutor for StatsScriptedExecutor {
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
