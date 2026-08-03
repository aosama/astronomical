use std::{path::PathBuf, time::Duration};

use astronomical_supervisor::{WorkerProcess, WorkerTerminationOutcome};

#[tokio::test]
async fn should_stop_after_graceful_shutdown_timeout_escalates_and_reaps_worker() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker-stubborn-eof-worker")
            .expect("Cargo should provide the stubborn EOF worker fixture path"),
    );
    let mut worker_client = WorkerProcess::launch_with_timeouts(
        worker_executable_path,
        Duration::from_secs(1),
        Duration::from_millis(500),
    )
    .await
    .expect("the stubborn EOF worker should report readiness");
    assert!(worker_client.process_id().is_some());

    let termination_outcome = worker_client
        .close()
        .await
        .expect("graceful timeout should escalate, terminate, and reap the worker");

    assert_eq!(
        termination_outcome,
        WorkerTerminationOutcome::Forced {
            process_exit_successful: false,
        }
    );
}
