//! Acceptance tests for the stop-first memory-demotion worker restart.
//!
//! The contract these tests pin is that the trusted worker process has exited
//! and been reaped before any replacement process is launched. Asserting that
//! from the supervisor's own control flow would only restate the implementation,
//! so the ordering is observed from the replacement process itself: the worker
//! fixture records the liveness of every sibling worker at the instant it
//! starts. A stop-first restart therefore records its predecessor as gone, and a
//! candidate-first restart would record it as alive.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{WorkerLogLevel, WorkerStartupConfiguration};
use astronomical_supervisor::{
    ChatGenerationExecutor, GenerationPerformanceLog, RuntimeModelPolicy, WorkerHandle,
    WorkerHealthStatus,
};
use tokio::time::timeout;

use super::worker_model_swap::{
    assert_generation_completed, chat_command, runtime_model_policy, wait_for_ready_worker,
};

const TRUSTED_CONFIGURATION_GENERATION: &str = "trusted-generation";
const REPLACEMENT_CONFIGURATION_GENERATION: &str = "replacement-generation";
const TRUSTED_MODEL_ID: &str = "astronomical/trusted-model";
const RESTORE_FAILURE_MODEL_ID: &str = "astronomical/restore-failure-model";
const SILENT_CANDIDATE_GENERATION: &str = "silent-acknowledgement-generation";
const MISMATCHED_CANDIDATE_GENERATION: &str = "mismatched-acknowledgement-generation";

const DEMOTION_CEILING_BYTES: u64 = 19_000_000_000;
/// Also the demotion acknowledgement timeout. Wide enough to absorb a fully
/// loaded continuous-integration runner, where a two-second window once let a
/// starved handshake masquerade as a lifecycle failure; still far inside the
/// repository's 120-second test ceiling.
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(5);
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(30);
const LAUNCH_ORDER_LOG_FILE: &str = "demotion-launch-order.log";
const WORKER_PROCESS_IDENTIFIER_PREFIX: &str = "demotion-worker-";

#[tokio::test]
async fn should_reap_the_trusted_worker_before_launching_the_demotion_replacement() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget");
    let restored_configuration = demotion_outcome.unwrap_or_else(|demotion_error| {
        panic!(
            "the demotion restart should restore the candidate configuration: {demotion_error:?}; worker_health={:?}",
            demotion_context.worker_handle.worker_health_snapshot()
        )
    });

    assert_eq!(
        restored_configuration.configuration_generation,
        REPLACEMENT_CONFIGURATION_GENERATION
    );
    assert_launch_observation(
        &demotion_context.temporary_directory,
        REPLACEMENT_CONFIGURATION_GENERATION,
        TRUSTED_CONFIGURATION_GENERATION,
        "gone",
    );
    assert!(
        !process_is_running(&recorded_worker_identifier(
            &demotion_context.temporary_directory,
            TRUSTED_CONFIGURATION_GENERATION,
        )),
        "the trusted worker process must be reaped once the demotion reports success"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_restore_the_selected_model_before_reporting_demotion_success() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget")
    .expect("the demotion restart should succeed for a restorable model");

    let restored_health = demotion_context.worker_handle.worker_health_snapshot();
    assert_eq!(
        restored_health.status,
        WorkerHealthStatus::Ready,
        "success must not be reported while the worker is not serving"
    );
    assert_eq!(
        restored_health.ready_model_id.as_deref(),
        Some(TRUSTED_MODEL_ID),
        "the same selected model must be restored before the endpoint reports success"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_end_the_demotion_in_idle_when_no_model_had_been_selected() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;

    let restored_configuration = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget")
    .expect("an idle demotion should succeed without an unnecessary model load");

    assert_eq!(
        restored_configuration.configuration_generation,
        REPLACEMENT_CONFIGURATION_GENERATION
    );
    assert_eq!(
        demotion_context
            .worker_handle
            .worker_health_snapshot()
            .ready_model_id,
        None,
        "a demotion with no prior selection must not invent one"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_not_publish_readiness_when_the_selected_model_cannot_be_restored() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, RESTORE_FAILURE_MODEL_ID)
            .await;
    demotion_context
        .load_selected_model(RESTORE_FAILURE_MODEL_ID)
        .await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(RESTORE_FAILURE_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the restart attempt should complete inside the journey budget");

    assert!(
        demotion_outcome.is_err(),
        "a failed model restoration must not report demotion success"
    );
    let restored_health = demotion_context.worker_handle.worker_health_snapshot();
    assert_ne!(
        restored_health.ready_model_id.as_deref(),
        Some(RESTORE_FAILURE_MODEL_ID),
        "a model the replacement never loaded must never be published as ready"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_a_candidate_that_never_acknowledges_its_configuration() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    SILENT_CANDIDATE_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the rejection should complete inside the journey budget");

    // The contract under test is that a silent candidate is never accepted:
    // which boundary reports the failure first (the acknowledgement timeout
    // here, or trusted-worker containment when a loaded runner starves the
    // handshake) is scheduler-dependent, so the post-condition is deliberately
    // variant-independent.
    assert!(
        demotion_outcome.is_err(),
        "a silent candidate must never be accepted, received {demotion_outcome:?}"
    );
    assert_eq!(
        demotion_context
            .worker_handle
            .worker_health_snapshot()
            .status,
        WorkerHealthStatus::Unavailable,
        "a rejected demotion must publish a truthful unavailable state"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_a_candidate_that_acknowledges_an_unrequested_generation() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    MISMATCHED_CANDIDATE_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the rejection should complete inside the journey budget");

    assert!(
        demotion_outcome.is_err(),
        "candidate configuration identity must bind, received {demotion_outcome:?}"
    );
    demotion_context.shutdown().await;
}

struct DemotionTestContext {
    temporary_directory: PathBuf,
    worker_handle: WorkerHandle,
}

impl DemotionTestContext {
    async fn launch(configuration_generation: &str, model_id: &str) -> Self {
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
    async fn load_selected_model(&self, model_id: &str) {
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

    async fn shutdown(self) {
        let _unused_shutdown_result = self.worker_handle.shutdown().await;
    }
}

/// Asserts what the named worker observed about another worker at the instant it
/// started. `alive` proves overlapping model-bearing processes existed, which is
/// exactly what the stop-first contract forbids.
fn assert_launch_observation(
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

fn recorded_worker_identifier(temporary_directory: &std::path::Path, generation: &str) -> String {
    std::fs::read_to_string(temporary_directory.join(format!(
        "{WORKER_PROCESS_IDENTIFIER_PREFIX}{generation}.pid"
    )))
    .expect("each launched worker fixture should record its process identifier")
}

fn process_is_running(process_identifier: &str) -> bool {
    let process_identifier = process_identifier.trim();
    if process_identifier.is_empty() {
        return false;
    }
    std::process::Command::new("kill")
        .args(["-0", process_identifier])
        .output()
        .is_ok_and(|process_probe| process_probe.status.success())
}

fn demotion_fixture_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-memory-demotion-worker")
            .unwrap_or_else(|_| {
                panic!(
                    "Cargo should provide the \
                     astronomical-supervisor-memory-demotion-worker fixture path"
                )
            }),
    )
}

fn startup_configuration(
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

fn model_policy_catalog(model_id: &str) -> Arc<HashMap<String, RuntimeModelPolicy>> {
    Arc::new(HashMap::from([(
        model_id.to_owned(),
        runtime_model_policy(model_id, &format!("/fictional/models/{model_id}"), 128),
    )]))
}
