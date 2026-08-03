use std::time::Duration;

use astronomical_supervisor::{ChatGenerationExecutor, WorkerHandle, WorkerHealthStatus};
use tokio::time::{Instant, sleep};

use crate::common::supervisor::launch_test_worker_with_model_load_timeout;

#[tokio::test]
async fn should_force_terminate_and_reap_a_worker_that_keeps_loading() {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-loading-forever-worker")
            .expect("Cargo should provide the loading-forever worker fixture path");
    let worker_executor = launch_test_worker_with_model_load_timeout(
        worker_executable_path,
        Duration::from_millis(40),
    )
    .await
    .expect("the loading-forever worker should start");

    wait_for_worker_health_status(
        &worker_executor,
        WorkerHealthStatus::Unavailable,
        Duration::from_secs(2),
    )
    .await;

    let shutdown_outcome = worker_executor
        .shutdown()
        .await
        .expect("the model-load timeout should already have terminated and reaped the worker");
    assert!(shutdown_outcome.was_successful());
}

#[tokio::test]
async fn should_report_loading_health_until_engine_readiness() {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-delayed-engine-ready-worker")
            .expect("Cargo should provide the delayed-engine-ready worker fixture path");
    let worker_executor =
        launch_test_worker_with_model_load_timeout(worker_executable_path, Duration::from_secs(2))
            .await
            .expect("the delayed-readiness worker should start");

    let loading_health = worker_executor.worker_health_snapshot();
    assert_eq!(loading_health.status, WorkerHealthStatus::Loading);
    assert_eq!(loading_health.status.as_str(), "loading");
    assert!(!loading_health.status.is_ready());

    worker_executor
        .shutdown()
        .await
        .expect("the delayed-readiness worker should be terminated and reaped");
}

async fn wait_for_worker_health_status(
    worker_executor: &WorkerHandle,
    expected_worker_health_status: WorkerHealthStatus,
    maximum_wait: Duration,
) {
    let health_deadline = Instant::now() + maximum_wait;

    loop {
        let worker_health_snapshot = worker_executor.worker_health_snapshot();
        if worker_health_snapshot.status == expected_worker_health_status {
            return;
        }

        assert!(
            Instant::now() < health_deadline,
            "worker health did not become {expected_worker_health_status:?}; last status was {:?}",
            worker_health_snapshot.status
        );
        sleep(Duration::from_millis(10)).await;
    }
}
