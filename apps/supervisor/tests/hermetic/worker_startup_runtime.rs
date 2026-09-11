//! Proves the first generation still completes when InitializeWorker
//! acknowledgement is still in flight after Idle has made the worker ready.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{WorkerLogLevel, WorkerStartupConfiguration};
use astronomical_supervisor::{ChatGenerationExecutor, GenerationPerformanceLog, WorkerHandle};
use tokio::time::timeout;

use super::worker_model_swap::{
    TELEMETRY_BEFORE_SWAP_MODEL_ID, assert_generation_completed, chat_command,
    runtime_model_policy, wait_for_ready_worker,
};

const DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION: &str =
    "delayed-startup-runtime-configuration";

#[tokio::test]
async fn should_complete_generation_when_startup_runtime_acknowledgement_is_still_in_flight() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_log_directory =
        tempfile::tempdir().expect("test performance log directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(1),
        GenerationPerformanceLog::open(temporary_log_directory.path())
            .expect("test performance log should be created"),
        Arc::new(HashMap::from([(
            TELEMETRY_BEFORE_SWAP_MODEL_ID.to_owned(),
            runtime_model_policy(
                TELEMETRY_BEFORE_SWAP_MODEL_ID,
                "/models/telemetry-before-swap-model",
                64,
            ),
        )])),
        WorkerStartupConfiguration {
            configuration_generation: DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION.to_owned(),
            global_prompt_cache_root_directory: temporary_log_directory.path().join("prompt-cache"),
            global_prompt_cache_maximum_size_bytes: 50_000_000_000,
            persistent_prompt_cache_enabled: true,
            configured_maximum_mlx_memory_bytes: None,
            performance_attribution_enabled: false,
            logging_directory: temporary_log_directory.path().to_path_buf(),
            logging_level: WorkerLogLevel::Warn,
            retained_log_file_count: 7,
        },
    )
    .await
    .expect("the idle worker should launch");
    wait_for_ready_worker(&worker_handle).await;

    let mut generation_events = timeout(
        Duration::from_secs(2),
        worker_handle.start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 1)),
    )
    .await
    .expect("startup runtime acknowledgement should not block generation admission")
    .expect("the first model should load after delayed startup runtime acknowledgement");
    assert_generation_completed(
        &mut generation_events,
        "delayed startup runtime acknowledgement",
    )
    .await;

    worker_handle
        .shutdown()
        .await
        .expect("the worker should remain available for graceful shutdown");
}
