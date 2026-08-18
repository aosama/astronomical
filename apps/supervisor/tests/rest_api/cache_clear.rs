use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, GenerationPerformanceLog, ResolvedRuntimeConfig,
    ResolvedRuntimeConfigResolver, ShutdownController, WorkerHandle, WorkerHealthStatus,
    build_application, build_application_with_full_control,
};
use axum::{
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
};
use tokio::time::{Instant, sleep, timeout};
use tower::ServiceExt;

const DELAYED_MODEL_ID: &str = "astronomical/delayed-completion-model";
const TEST_TIMEOUT: Duration = Duration::from_secs(10);
const ROMEO_AND_JULIET: &str = include_str!(
    "../../../inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
async fn should_clear_the_entire_ssd_cache_while_the_worker_is_idle() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;
        eprintln!("[cache-clear-rest] deleting the global cache while idle");

        let clear_response = delete_cache(&test_context.application, "/v1/cache").await;
        assert_eq!(clear_response.status(), StatusCode::OK);
        let clear_document = response_json(clear_response).await;
        assert_eq!(clear_document["status"], "cleared");
        assert_eq!(clear_document["model_id"], serde_json::Value::Null);
        assert_eq!(clear_document["blocks_removed"], 3);
        assert_eq!(clear_document["bytes_freed"], 4_096);
        wait_for_zeroed_cache_stats(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_clear_one_model_cache_by_its_model_id() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;

        let clear_response = delete_cache(
            &test_context.application,
            "/v1/cache?model=astronomical%2Frequested-model",
        )
        .await;
        assert_eq!(clear_response.status(), StatusCode::OK);
        let clear_document = response_json(clear_response).await;
        assert_eq!(clear_document["model_id"], "astronomical/requested-model");

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_queue_only_the_newest_cache_clear_until_generation_finishes() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;
        eprintln!("[cache-clear-rest] starting the Romeo and Juliet generation journey");
        let mut generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(41))
            .await
            .expect("generation should start");

        let global_clear_response = delete_cache(&test_context.application, "/v1/cache").await;
        assert_eq!(global_clear_response.status(), StatusCode::ACCEPTED);
        let scoped_clear_response = delete_cache(
            &test_context.application,
            "/v1/cache?model=astronomical%2Frequested-model",
        )
        .await;
        assert_eq!(scoped_clear_response.status(), StatusCode::ACCEPTED);

        let queued_stats = get_cache_stats(&test_context.application).await;
        assert_eq!(
            queued_stats["pending_cache_clear"]["model_id"],
            "astronomical/requested-model"
        );

        timeout(Duration::from_secs(2), generation_events.recv())
            .await
            .expect("generation should finish before the timeout")
            .expect("generation should emit completion");
        wait_for_pending_clear_to_finish(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_wait_for_a_queued_generation_before_applying_the_cache_clear() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;
        let mut first_generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(51))
            .await
            .expect("first generation should start");
        let queued_worker_handle = test_context.worker_handle.clone();
        let queued_generation_task = tokio::spawn(async move {
            queued_worker_handle
                .start_chat_generation(delayed_generation_command(52))
                .await
        });

        let clear_response = delete_cache(&test_context.application, "/v1/cache").await;
        assert_eq!(clear_response.status(), StatusCode::ACCEPTED);
        receive_generation_completion(&mut first_generation_events).await;
        let mut queued_generation_events = timeout(Duration::from_secs(2), queued_generation_task)
            .await
            .expect("queued generation should start after the first")
            .expect("queued generation task should not panic")
            .expect("queued generation should be accepted");

        let cache_stats_during_queued_generation = get_cache_stats(&test_context.application).await;
        assert!(
            !cache_stats_during_queued_generation["pending_cache_clear"].is_null(),
            "cache clear must remain pending while the queued generation runs"
        );
        receive_generation_completion(&mut queued_generation_events).await;
        wait_for_pending_clear_to_finish(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_reject_a_model_id_that_can_escape_the_cache_root() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;

        let clear_response =
            delete_cache(&test_context.application, "/v1/cache?model=..%2Foutside").await;
        assert_eq!(clear_response.status(), StatusCode::BAD_REQUEST);
        let empty_model_response =
            delete_cache(&test_context.application, "/v1/cache?model=").await;
        assert_eq!(empty_model_response.status(), StatusCode::BAD_REQUEST);

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_not_offer_cache_clear_without_live_worker_control() {
    run_bounded_test(async {
        let application = build_application(WorkerHandle::unavailable());

        let clear_response = delete_cache(&application, "/v1/cache").await;

        assert_eq!(clear_response.status(), StatusCode::NOT_FOUND);
    })
    .await;
}

#[tokio::test]
async fn should_return_service_unavailable_when_the_worker_has_stopped() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;
        test_context
            .worker_handle
            .clone()
            .shutdown()
            .await
            .expect("worker fixture should stop");

        let clear_response = delete_cache(&test_context.application, "/v1/cache").await;

        assert_eq!(clear_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    })
    .await;
}

#[tokio::test]
async fn should_reject_a_worker_acknowledgement_for_a_different_model_scope() {
    run_bounded_test(async {
        let test_context = launch_cache_clear_application().await;

        let clear_response = delete_cache(
            &test_context.application,
            "/v1/cache?model=astronomical%2Fmismatched-clear-model",
        )
        .await;

        assert_eq!(clear_response.status(), StatusCode::SERVICE_UNAVAILABLE);
        test_context.shutdown().await;
    })
    .await;
}

struct CacheClearTestContext {
    application: axum::Router,
    worker_handle: WorkerHandle,
    _temporary_directory: tempfile::TempDir,
}

impl CacheClearTestContext {
    async fn shutdown(self) {
        self.worker_handle
            .shutdown()
            .await
            .expect("cache-clear fixture should shut down");
    }
}

async fn launch_cache_clear_application() -> CacheClearTestContext {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_directory = tempfile::tempdir().expect("test directory should be created");
    let model_directory = temporary_directory.path().join("delayed-completion-model");
    let model_directories = Arc::new(HashMap::from([(
        DELAYED_MODEL_ID.to_owned(),
        model_directory,
    )]));
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(temporary_directory.path())
            .expect("test performance log should open"),
        Arc::clone(&model_directories),
        20_480,
    )
    .await
    .expect("idle worker fixture should launch");
    wait_for_ready_worker(&worker_handle).await;

    let resolved_runtime_config = ResolvedRuntimeConfig {
        worker_executable_path: worker_executable_path.clone(),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_directories,
        max_output_tokens: 20_480,
        maximum_mlx_memory_bytes: None,
        chunking: astronomical_config::ChunkingConfig::default(),
        persistent_prompt_cache_enabled: true,
        performance_attribution_enabled: false,
        mtp_enabled: false,
        mtp_draft_depth: None,
        mtp_pairings: Vec::new(),
        speculative_prefill: astronomical_config::SpeculativePrefillConfig::disabled(),
        speculative_prefill_draft_model_directory: None,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            temporary_directory.path().join("cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            temporary_directory.path().join("logs"),
            astronomical_config::LogLevel::Warn,
            2,
        ),
    };
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        temporary_directory.path().to_path_buf(),
        worker_executable_path,
    );
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::new(RwLock::new(resolved_runtime_config)),
        runtime_config_resolver,
        ShutdownController::new(),
    );
    CacheClearTestContext {
        application,
        worker_handle,
        _temporary_directory: temporary_directory,
    }
}

fn delayed_generation_command(request_id: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: DELAYED_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: ROMEO_AND_JULIET.to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}

async fn receive_generation_completion(
    generation_events: &mut tokio::sync::mpsc::Receiver<
        astronomical_supervisor::ChatGenerationStreamEvent,
    >,
) {
    timeout(Duration::from_secs(2), generation_events.recv())
        .await
        .expect("generation should finish before the timeout")
        .expect("generation should emit completion");
}

async fn delete_cache(application: &axum::Router, request_uri: &str) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri(request_uri)
                .body(Body::empty())
                .expect("cache-clear request should be valid"),
        )
        .await
        .expect("application should return a cache-clear response")
}

async fn get_cache_stats(application: &axum::Router) -> serde_json::Value {
    let stats_response = application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/cache/stats")
                .body(Body::empty())
                .expect("cache-stats request should be valid"),
        )
        .await
        .expect("application should return cache stats");
    response_json(stats_response).await
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), 8 * 1024)
        .await
        .expect("response body should be readable");
    serde_json::from_slice(&response_body).expect("response should contain JSON")
}

async fn wait_for_ready_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if worker_handle.worker_health_snapshot().status == WorkerHealthStatus::Ready {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "worker should become ready"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_pending_clear_to_finish(application: &axum::Router) {
    let clear_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let cache_stats = get_cache_stats(application).await;
        if cache_stats["pending_cache_clear"].is_null() {
            return;
        }
        assert!(
            Instant::now() < clear_deadline,
            "queued clear should finish"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_zeroed_cache_stats(application: &axum::Router) {
    let stats_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let cache_stats = get_cache_stats(application).await;
        if cache_stats["persistent_prompt_cache_maximum_size_bytes"] == 50_000_000_000_u64 {
            assert_eq!(
                cache_stats["persistent_prompt_cache_sequence_state_block_count"],
                0
            );
            assert_eq!(cache_stats["persistent_prompt_cache_total_size_bytes"], 0);
            return;
        }
        assert!(
            Instant::now() < stats_deadline,
            "cleared stats should arrive"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn run_bounded_test(test_journey: impl std::future::Future<Output = ()>) {
    timeout(TEST_TIMEOUT, test_journey)
        .await
        .expect("cache-clear REST journey should finish within ten seconds");
}
