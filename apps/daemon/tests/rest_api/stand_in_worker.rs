//! The daemon's stand-in worker process behind a full application, shared by
//! the REST journeys that need a live worker loop: it serves one delayed
//! model, so a test can load a model, hold a request open, and watch the
//! worker answer control commands around it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, GenerationPerformanceLog, ResolvedRuntimeConfig,
    ResolvedRuntimeConfigResolver, RuntimeModelPolicy, ShutdownController, WorkerHandle,
    WorkerHealthStatus, build_application_with_full_control,
};
use axum::body::to_bytes;
use ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerAutoregressiveModelConfiguration, WorkerModelConfiguration,
};
use tokio::time::{Instant, sleep, timeout};

pub(super) const TEST_TIMEOUT: Duration = Duration::from_secs(10);
const ROMEO_AND_JULIET: &str = include_str!(
    "../../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

pub(super) const DELAYED_MODEL_ID: &str = "astronomical/delayed-completion-model";

pub(super) struct StandInWorkerTestContext {
    pub(super) application: axum::Router,
    pub(super) worker_handle: WorkerHandle,
    _temporary_directory: tempfile::TempDir,
}

impl StandInWorkerTestContext {
    pub(super) async fn shutdown(self) {
        self.worker_handle
            .shutdown()
            .await
            .expect("the stand-in worker should shut down");
    }
}

pub(super) async fn launch_stand_in_worker_application() -> StandInWorkerTestContext {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-daemon-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_directory = tempfile::tempdir().expect("test directory should be created");
    let model_directory = temporary_directory.path().join("delayed-completion-model");
    let model_policy_catalog = Arc::new(HashMap::from([(
        DELAYED_MODEL_ID.to_owned(),
        RuntimeModelPolicy {
            model_directory,
            generation_defaults: astronomical_daemon::RuntimeModelGenerationDefaults {
                maximum_output_tokens: 128,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 2_048,
            configured_chunking_fields: Default::default(),
            worker_model_configuration: test_worker_model_configuration(DELAYED_MODEL_ID),
        },
    )]));
    let worker_handle = WorkerHandle::launch(
        &worker_executable_path,
        Duration::from_secs(2),
        GenerationPerformanceLog::open(temporary_directory.path())
            .expect("test performance log should open"),
        Arc::clone(&model_policy_catalog),
    )
    .await
    .expect("idle worker fixture should launch");
    wait_for_ready_worker(&worker_handle).await;

    let resolved_runtime_config = ResolvedRuntimeConfig {
        neural_engine_program_directory: "neural-engine-programs".into(),
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: worker_executable_path.clone(),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog,
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        astro_ssd_prompt_cache_enabled: true,
        configured_astro_ssd_prompt_cache_enabled: None,
        metal_residency_refresh_enabled: false,
        idle_model_unload_seconds: 0,
        menu_bar_shows_progress_and_speed: false,
        menu_bar_shows_technical_details: false,
        skill_settings: Default::default(),
        sharing_settings: Default::default(),
        configured_astro_ssd_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        astro_ssd_prompt_cache_config: config::AstroSsdPromptCacheConfig::new(
            temporary_directory.path().join("cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: config::LoggingConfig::new(
            temporary_directory.path().join("logs"),
            config::LogLevel::Warn,
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
    StandInWorkerTestContext {
        application,
        worker_handle,
        _temporary_directory: temporary_directory,
    }
}

pub(super) fn test_worker_model_configuration(model_id: &str) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
        model_id: model_id.to_owned(),
        maximum_context_tokens: 2_048,
        maximum_output_tokens: 128,
        chunking: crate::support::worker_chunking_with_prompt_chunk_tokens(256, 2_048),
    })
}

pub(super) fn delayed_generation_command(request_id: u64) -> ChatGenerationCommand {
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
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
    }
}

pub(super) async fn receive_generation_completion(
    generation_events: &mut tokio::sync::mpsc::Receiver<
        astronomical_daemon::ChatGenerationStreamEvent,
    >,
) {
    timeout(Duration::from_secs(2), generation_events.recv())
        .await
        .expect("generation should finish before the timeout")
        .expect("generation should emit completion");
}

pub(super) async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), 8 * 1024)
        .await
        .expect("response body should be readable");
    serde_json::from_slice(&response_body).expect("response should contain JSON")
}

pub(super) async fn wait_for_ready_worker(worker_handle: &WorkerHandle) {
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

pub(super) async fn run_bounded_test(test_journey: impl std::future::Future<Output = ()>) {
    timeout(TEST_TIMEOUT, test_journey)
        .await
        .expect("the stand-in worker journey should finish within ten seconds");
}
