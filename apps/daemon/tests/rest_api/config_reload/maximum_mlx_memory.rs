use super::*;

use std::collections::HashMap;

use astronomical_daemon::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};
use ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerAutoregressiveModelConfiguration, WorkerModelConfiguration,
};

const DELAYED_MEMORY_MODEL_ID: &str = "astronomical/delayed-completion-model";
const ROMEO_AND_JULIET: &str = include_str!(
    "../../../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
async fn should_require_full_reload_when_other_configuration_changes_are_pending() {
    let MemoryJourney {
        worker_handle,
        application,
        reloadable_config,
        config_home_directory,
        ..
    } = start_memory_journey(MemoryJourneyModels::None).await;
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
    shut_down_worker(worker_handle).await;
}

#[tokio::test]
async fn should_not_let_a_rejected_update_restore_over_a_newer_memory_setting() {
    let MemoryJourney {
        worker_handle,
        application,
        config_home_directory,
        ..
    } = start_memory_journey(MemoryJourneyModels::None).await;

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
    shut_down_worker(worker_handle).await;
}

#[tokio::test]
async fn should_apply_a_memory_setting_while_a_reply_runs() {
    timeout(Duration::from_secs(10), async {
        let MemoryJourney {
            worker_handle,
            application,
            config_home_directory,
            ..
        } = start_memory_journey(MemoryJourneyModels::DelayedCompletion).await;
        let mut generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the delayed generation should start");

        let applied_response = put_maximum_mlx_memory(&application, 32).await;

        assert_eq!(applied_response.status(), StatusCode::OK);
        assert_eq!(
            worker_handle
                .worker_health_snapshot()
                .mlx_memory_ceiling_bytes,
            32 * 1_073_741_824
        );
        generation_events
            .recv()
            .await
            .expect("the reply should still finish after the change");
        wait_for_persisted_maximum(&config_home_directory, 32).await;
        shut_down_worker(worker_handle).await;
    })
    .await
    .expect("the mid-reply memory journey should finish");
}

#[tokio::test]
async fn should_refuse_a_memory_setting_during_a_reply_at_once_and_keep_the_reply() {
    timeout(Duration::from_secs(10), async {
        let MemoryJourney {
            worker_handle,
            application,
            config_home_directory,
            ..
        } = start_memory_journey(MemoryJourneyModels::DelayedCompletion).await;
        let mut generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the delayed generation should start");

        let refused_response = put_maximum_mlx_memory(&application, 31).await;

        assert_eq!(refused_response.status(), StatusCode::BAD_REQUEST);
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        assert_eq!(
            worker_health_snapshot.mlx_memory_limit_error.as_deref(),
            Some("fixture rejected the requested limit")
        );
        assert_ne!(
            worker_health_snapshot.mlx_memory_ceiling_bytes,
            31 * 1_073_741_824
        );
        generation_events
            .recv()
            .await
            .expect("the reply should still finish after the refusal");
        wait_for_persisted_maximum(&config_home_directory, 31).await;
        shut_down_worker(worker_handle).await;
    })
    .await
    .expect("the mid-reply refusal journey should finish");
}

#[tokio::test]
async fn should_accept_the_next_memory_setting_after_a_reloaded_one_is_refused() {
    timeout(Duration::from_secs(10), async {
        let MemoryJourney {
            worker_handle,
            application,
            reloadable_config,
            config_home_directory,
            initial_generation,
        } = start_memory_journey(MemoryJourneyModels::DelayedCompletion).await;
        let mut generation_events = worker_handle
            .start_chat_generation(delayed_memory_generation_command())
            .await
            .expect("the delayed generation should start");
        write_config_file(
            &config_home_directory,
            r#"{"runtime":{"model_directories":[],"maximum_mlx_memory_gb":31}}"#,
        );

        let reload_response = post_config_reload(&application).await;

        assert_eq!(reload_response.status(), StatusCode::BAD_REQUEST);
        let live_configuration_generation = reloadable_config
            .read()
            .map(|resolved_config| resolved_config.configuration_generation.clone())
            .expect("live config should remain readable");
        assert_eq!(live_configuration_generation, initial_generation);
        generation_events
            .recv()
            .await
            .expect("the reply should still finish after the refusal");
        // A refused reload once left a marker behind that refused every later
        // change with a conflict until the daemon restarted.
        assert_eq!(
            put_maximum_mlx_memory(&application, 32).await.status(),
            StatusCode::OK
        );
        shut_down_worker(worker_handle).await;
    })
    .await
    .expect("the refused reload journey should finish");
}

enum MemoryJourneyModels {
    None,
    DelayedCompletion,
}

struct MemoryJourney {
    worker_handle: WorkerHandle,
    application: axum::Router,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    config_home_directory: PathBuf,
    initial_generation: String,
}

/// Brings up an idle worker and a daemon application over an empty config
/// file, optionally hosting the delayed-completion model so a memory-limit
/// change can arrive while a reply runs.
async fn start_memory_journey(models: MemoryJourneyModels) -> MemoryJourney {
    let worker_executable_path = idle_worker_executable_path();
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let model_policy_catalog = match models {
        MemoryJourneyModels::None => Arc::new(HashMap::new()),
        MemoryJourneyModels::DelayedCompletion => Arc::new(HashMap::from([(
            DELAYED_MEMORY_MODEL_ID.to_owned(),
            delayed_memory_model_policy(&config_home_directory),
        )])),
    };
    let worker_handle = launch_idle_worker(
        &worker_executable_path,
        &config_home_directory,
        model_policy_catalog,
    )
    .await;
    wait_for_idle_worker(&worker_handle).await;
    let (runtime_config_resolver, initial_resolved_config) =
        resolve_initial_config(&config_home_directory, &worker_executable_path);
    let initial_generation = initial_resolved_config.configuration_generation.clone();
    let reloadable_config = Arc::new(RwLock::new(initial_resolved_config));
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        ShutdownController::new(),
    );
    MemoryJourney {
        worker_handle,
        application,
        reloadable_config,
        config_home_directory,
        initial_generation,
    }
}

fn delayed_memory_model_policy(model_root: &std::path::Path) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: model_root.join("delayed-completion-model"),
        generation_defaults: RuntimeModelGenerationDefaults {
            maximum_output_tokens: 128,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 2_048,
        configured_chunking_fields: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
                model_id: DELAYED_MEMORY_MODEL_ID.to_owned(),
                maximum_context_tokens: 2_048,
                maximum_output_tokens: 128,
                chunking: crate::support::worker_chunking_with_prompt_chunk_tokens(256, 2_048),
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
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
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
