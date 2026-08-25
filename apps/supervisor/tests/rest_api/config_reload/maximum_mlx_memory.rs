use super::*;

use std::collections::HashMap;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration, WorkerModelConfiguration,
};
use astronomical_supervisor::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};

const DELAYED_MEMORY_MODEL_ID: &str = "astronomical/delayed-completion-model";
const ROMEO_AND_JULIET: &str = include_str!(
    "../../../../inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
async fn should_require_full_reload_when_other_configuration_changes_are_pending() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance-log directory should be created");
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        Arc::new(HashMap::new()),
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        worker_executable_path.clone(),
    );
    let mut initial_resolved_config = runtime_config_resolver
        .load()
        .expect("initial config should resolve");
    initial_resolved_config.worker_executable_path = worker_executable_path;
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        ShutdownController::new(),
    );
    write_config_file(
        &config_home_directory,
        r#"{"diagnostics":{"log_level":"info"}}"#,
    );

    let response = put_maximum_mlx_memory(&application, 32).await;

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(
        std::fs::read_to_string(config_home_directory.join(".astronomical-dev/config.json"))
            .expect("config should remain readable")
            .contains("\"log_level\": \"info\"")
    );
    assert!(
        reloadable_config
            .read()
            .expect("live config should remain readable")
            .maximum_mlx_memory_bytes
            .is_none()
    );
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

#[tokio::test]
async fn should_not_let_a_rejected_update_restore_over_a_newer_memory_setting() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let performance_log_directory = config_home_directory.join("performance");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the performance-log directory should be created");
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        Arc::new(std::collections::HashMap::new()),
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        worker_executable_path.clone(),
    );
    let mut initial_resolved_config = runtime_config_resolver
        .load()
        .expect("initial config should resolve");
    initial_resolved_config.worker_executable_path = worker_executable_path;
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_application_with_full_control(
        worker_handle.clone(),
        reloadable_config,
        runtime_config_resolver,
        ShutdownController::new(),
    );

    let rejected_application = application.clone();
    let rejected_update_task =
        tokio::spawn(async move { put_maximum_mlx_memory(&rejected_application, 31).await });
    wait_for_persisted_maximum(&config_home_directory, 31).await;
    write_config_file(
        &config_home_directory,
        r#"{"runtime":{"model_directories":[],"maximum_mlx_memory_gb":32}}"#,
    );
    let rejected_response = rejected_update_task
        .await
        .expect("the rejected update task should finish");

    assert_eq!(rejected_response.status(), StatusCode::BAD_REQUEST);
    wait_for_persisted_maximum(&config_home_directory, 32).await;
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

#[tokio::test]
async fn should_preserve_persisted_intent_when_queued_application_is_rejected() {
    timeout(Duration::from_secs(10), async {
        let worker_executable_path = PathBuf::from(
            std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
                .expect("Cargo should provide the idle worker fixture path"),
        );
        let config_home_directory = tempfile::tempdir()
            .expect("a config home should be created")
            .keep();
        write_config_file(&config_home_directory, r#"{}"#);
        let model_policy_catalog = Arc::new(HashMap::from([(
            DELAYED_MEMORY_MODEL_ID.to_owned(),
            delayed_memory_model_policy(&config_home_directory),
        )]));
        let performance_log_directory = config_home_directory.join("performance");
        std::fs::create_dir_all(&performance_log_directory)
            .expect("the performance log directory should be created");
        let worker_handle = WorkerHandle::launch(
            &worker_executable_path,
            Duration::from_secs(2),
            GenerationPerformanceLog::open(&performance_log_directory)
                .expect("the performance log should open"),
            Arc::clone(&model_policy_catalog),
        )
        .await
        .expect("the idle worker should launch");
        wait_for_idle_worker(&worker_handle).await;
        let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
            config_home_directory.clone(),
            worker_executable_path.clone(),
        );
        let mut initial_resolved_config = runtime_config_resolver
            .load()
            .expect("initial config should resolve");
        initial_resolved_config.worker_executable_path = worker_executable_path;
        let application = build_application_with_full_control(
            worker_handle.clone(),
            Arc::new(RwLock::new(initial_resolved_config)),
            runtime_config_resolver,
            ShutdownController::new(),
        );
        let mut generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the delayed generation should start");

        let queued_response = put_maximum_mlx_memory(&application, 31).await;
        assert_eq!(queued_response.status(), StatusCode::ACCEPTED);
        let conflicting_response = put_maximum_mlx_memory(&application, 32).await;
        assert_eq!(conflicting_response.status(), StatusCode::CONFLICT);
        generation_events
            .recv()
            .await
            .expect("the delayed generation should complete");
        wait_for_memory_rejection(&worker_handle).await;
        wait_for_persisted_maximum(&config_home_directory, 31).await;

        let mut second_generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the second delayed generation should start");
        assert_eq!(
            put_maximum_mlx_memory(&application, 31).await.status(),
            StatusCode::ACCEPTED
        );
        write_config_file(
            &config_home_directory,
            r#"{"runtime":{"model_directories":[],"maximum_mlx_memory_gb":33}}"#,
        );
        second_generation_events
            .recv()
            .await
            .expect("the second delayed generation should complete");
        wait_for_memory_rejection(&worker_handle).await;
        wait_for_persisted_maximum(&config_home_directory, 33).await;

        worker_handle
            .shutdown()
            .await
            .expect("the worker should shut down");
    })
    .await
    .expect("the queued memory rejection journey should finish");
}

#[tokio::test]
async fn should_rollback_live_state_when_a_reloaded_memory_setting_is_rejected_after_queueing() {
    timeout(Duration::from_secs(10), async {
        let worker_executable_path = PathBuf::from(
            std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
                .expect("Cargo should provide the idle worker fixture path"),
        );
        let config_home_directory = tempfile::tempdir()
            .expect("a config home should be created")
            .keep();
        write_config_file(&config_home_directory, r#"{}"#);
        let model_policy_catalog = Arc::new(HashMap::from([(
            DELAYED_MEMORY_MODEL_ID.to_owned(),
            delayed_memory_model_policy(&config_home_directory),
        )]));
        let performance_log_directory = config_home_directory.join("performance");
        std::fs::create_dir_all(&performance_log_directory)
            .expect("the performance log directory should be created");
        let worker_handle = WorkerHandle::launch(
            &worker_executable_path,
            Duration::from_secs(2),
            GenerationPerformanceLog::open(&performance_log_directory)
                .expect("the performance log should open"),
            Arc::clone(&model_policy_catalog),
        )
        .await
        .expect("the idle worker should launch");
        wait_for_idle_worker(&worker_handle).await;
        let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
            config_home_directory.clone(),
            worker_executable_path.clone(),
        );
        let mut initial_resolved_config = runtime_config_resolver
            .load()
            .expect("initial config should resolve");
        initial_resolved_config.worker_executable_path = worker_executable_path;
        let initial_generation = initial_resolved_config.configuration_generation.clone();
        let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
        let application = build_application_with_full_control(
            worker_handle.clone(),
            Arc::clone(&reloadable_config),
            runtime_config_resolver,
            ShutdownController::new(),
        );
        let mut generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the delayed generation should start");
        write_config_file(
            &config_home_directory,
            r#"{"runtime":{"model_directories":[],"maximum_mlx_memory_gb":31}}"#,
        );

        let reload_response = post_config_reload(&application).await;
        assert_eq!(reload_response.status(), StatusCode::OK);
        assert_eq!(
            put_maximum_mlx_memory(&application, 32).await.status(),
            StatusCode::CONFLICT
        );
        generation_events
            .recv()
            .await
            .expect("the delayed generation should complete");
        wait_for_resolved_generation(&reloadable_config, &initial_generation).await;
        wait_for_persisted_maximum(&config_home_directory, 31).await;

        worker_handle
            .shutdown()
            .await
            .expect("the worker should shut down");
    })
    .await
    .expect("the queued reload rollback journey should finish");
}

fn delayed_memory_model_policy(model_root: &std::path::Path) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: model_root.join("delayed-completion-model"),
        generation_defaults: RuntimeModelGenerationDefaults {
            maximum_output_tokens: 128,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 2_048,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
                model_id: DELAYED_MEMORY_MODEL_ID.to_owned(),
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
                mtp_draft_depth: None,
                speculative_prefill: None,
            },
        ),
    }
}

fn delayed_memory_generation_command() -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(9_001),
        model: DELAYED_MEMORY_MODEL_ID.to_owned(),
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
        qwen_thinking_channel_seed: None,
    }
}

async fn wait_for_resolved_generation(
    reloadable_config: &Arc<RwLock<ResolvedRuntimeConfig>>,
    expected_generation: &str,
) {
    let reconciliation_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let has_restored_live_configuration = {
            let resolved_config = reloadable_config
                .read()
                .expect("resolved configuration should remain readable");
            resolved_config.configuration_generation == expected_generation
                && resolved_config.maximum_mlx_memory_bytes.is_none()
        };
        if has_restored_live_configuration {
            return;
        }
        assert!(
            Instant::now() < reconciliation_deadline,
            "rejected queued reload did not restore live configuration"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_memory_rejection(worker_handle: &WorkerHandle) {
    let rejection_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot
            .pending_mlx_memory_ceiling_bytes
            .is_none()
            && worker_health_snapshot.mlx_memory_limit_error.is_some()
        {
            sleep(Duration::from_millis(50)).await;
            return;
        }
        assert!(
            Instant::now() < rejection_deadline,
            "queued memory rejection was not published"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn put_maximum_mlx_memory(
    application: &axum::Router,
    maximum_mlx_memory_gb: u64,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config/maximum-mlx-memory")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    "{{\"maximum_mlx_memory_gb\":{maximum_mlx_memory_gb}}}"
                )))
                .expect("the memory-limit request should be valid"),
        )
        .await
        .expect("the application should return a memory-limit response")
}

async fn wait_for_persisted_maximum(home_directory: &std::path::Path, expected_gigabytes: u64) {
    let config_file_path = home_directory.join(".astronomical-dev").join("config.json");
    let persistence_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let persisted_gigabytes = std::fs::read(&config_file_path)
            .ok()
            .and_then(|config_bytes| {
                serde_json::from_slice::<serde_json::Value>(&config_bytes).ok()
            })
            .and_then(|config_document| {
                config_document["runtime"]["maximum_mlx_memory_gb"].as_u64()
            });
        if persisted_gigabytes == Some(expected_gigabytes) {
            return;
        }
        assert!(
            Instant::now() < persistence_deadline,
            "maximum_mlx_memory_gb did not become {expected_gigabytes}; observed {persisted_gigabytes:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_idle_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.machine_mlx_memory_ceiling_bytes == 40_000_000_000
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "idle worker did not report its memory limits: {worker_health_snapshot:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}
