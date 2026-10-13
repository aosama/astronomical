//! Shared harness for the stop-first memory-demotion acceptance tests.
//!
//! The demotion tests observe the process boundary from outside the
//! daemon's control flow, so the harness owns the fixture launch, the
//! launch-observation log assertions, and the process-liveness probes those
//! observations rely on.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_daemon::{
    ChatGenerationExecutor, GenerationPerformanceLog, RuntimeModelPolicy, WorkerHandle,
};
use astronomical_ipc_protocol::{WorkerLogLevel, WorkerStartupConfiguration};

use super::worker_model_swap::{
    assert_generation_completed, chat_command, runtime_model_policy, wait_for_ready_worker,
};

/// Also the demotion acknowledgement timeout. Wide enough to absorb a fully
/// loaded continuous-integration runner, where a two-second window once let a
/// starved handshake masquerade as a lifecycle failure; still far inside the
/// repository's 120-second test ceiling.
pub(super) const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(5);
pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(30);
pub(super) const LAUNCH_ORDER_LOG_FILE: &str = "demotion-launch-order.log";
pub(super) const WORKER_PROCESS_IDENTIFIER_PREFIX: &str = "demotion-worker-";

pub(super) struct DemotionTestContext {
    pub(super) temporary_directory: PathBuf,
    pub(super) worker_handle: WorkerHandle,
}

impl DemotionTestContext {
    pub(super) async fn launch(configuration_generation: &str, model_id: &str) -> Self {
        let temporary_directory = tempfile::tempdir()
            .expect("the memory-demotion test directory should be created")
            .keep();
        let worker_handle = WorkerHandle::launch_with_startup_configuration(
            demotion_fixture_path(),
            MODEL_LOAD_TIMEOUT,
            GenerationPerformanceLog::open(&temporary_directory)
                .expect("the performance log should open"),
            model_policy_catalog(model_id),
            startup_configuration(
                configuration_generation,
                &temporary_directory,
                Some(40_000_000_000),
            ),
        )
        .await
        .expect("the trusted worker should launch");
        wait_for_ready_worker(&worker_handle).await;
        Self {
            temporary_directory,
            worker_handle,
        }
    }

    /// Loads a model so the trusted worker holds a selected model, which is the
    /// state a real demotion happens from.
    pub(super) async fn load_selected_model(&self, model_id: &str) {
        let mut generation_events = self
            .worker_handle
            .start_chat_generation(chat_command(model_id, 1))
            .await
            .expect("the trusted model should load for the demotion journey");
        assert_generation_completed(&mut generation_events, "before the memory demotion").await;
        wait_for_ready_worker(&self.worker_handle).await;
        assert_eq!(
            self.worker_handle
                .worker_health_snapshot()
                .ready_model_id
                .as_deref(),
            Some(model_id),
            "the demotion must start from a genuinely selected model"
        );
    }

    pub(super) async fn shutdown(self) {
        let _unused_shutdown_result = self.worker_handle.shutdown().await;
    }
}

/// Asserts what the named worker observed about another worker at the instant it
/// started. `alive` proves overlapping model-bearing processes existed, which is
/// exactly what the stop-first contract forbids.
pub(super) fn assert_launch_observation(
    temporary_directory: &std::path::Path,
    observing_generation: &str,
    observed_generation: &str,
    expected_state: &str,
) {
    let log_contents = std::fs::read_to_string(temporary_directory.join(LAUNCH_ORDER_LOG_FILE))
        .expect("every launched worker fixture should record a launch observation");
    let observation = log_contents
        .lines()
        .find(|line| line.starts_with(&format!("{observing_generation} observed")))
        .unwrap_or_else(|| {
            panic!(
                "the {observing_generation} worker should record a launch observation in {log_contents:?}"
            )
        });
    let expected_observation = format!("{observed_generation}={expected_state}");
    assert!(
        observation.contains(&expected_observation),
        "the {observing_generation} worker must have observed {observed_generation} as {expected_state}, recorded {observation:?}"
    );
}

pub(super) fn recorded_worker_identifier(
    temporary_directory: &std::path::Path,
    generation: &str,
) -> String {
    std::fs::read_to_string(temporary_directory.join(format!(
        "{WORKER_PROCESS_IDENTIFIER_PREFIX}{generation}.pid"
    )))
    .expect("each launched worker fixture should record its process identifier")
}

pub(super) fn process_is_running(process_identifier: &str) -> bool {
    let process_identifier = process_identifier.trim();
    if process_identifier.is_empty() {
        return false;
    }
    std::process::Command::new("kill")
        .args(["-0", process_identifier])
        .output()
        .is_ok_and(|process_probe| process_probe.status.success())
}

pub(super) fn demotion_fixture_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-daemon-memory-demotion-worker").unwrap_or_else(
            |_| {
                panic!(
                    "Cargo should provide the \
                     astronomical-daemon-memory-demotion-worker fixture path"
                )
            },
        ),
    )
}

pub(super) fn startup_configuration(
    configuration_generation: &str,
    temporary_directory: &std::path::Path,
    configured_maximum_mlx_memory_bytes: Option<u64>,
) -> WorkerStartupConfiguration {
    WorkerStartupConfiguration {
        configuration_generation: configuration_generation.to_owned(),
        global_prompt_cache_root_directory: temporary_directory.join("cache"),
        global_prompt_cache_maximum_size_bytes: 1_000_000_000,
        persistent_prompt_cache_enabled: true,
        configured_maximum_mlx_memory_bytes,
        performance_attribution_enabled: true,
        logging_directory: temporary_directory.to_path_buf(),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
    }
}

pub(super) fn model_policy_catalog(model_id: &str) -> Arc<HashMap<String, RuntimeModelPolicy>> {
    Arc::new(HashMap::from([(
        model_id.to_owned(),
        runtime_model_policy(model_id, &format!("/fictional/models/{model_id}"), 128),
    )]))
}
