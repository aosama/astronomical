use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};

use astronomical_supervisor::{GenerationPerformanceLog, WorkerControlError, WorkerHandle};

const DEFAULT_TEST_MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_TEST_MAX_OUTPUT_TOKENS: u32 = 20480;

/// Creates a performance log backed by a temporary directory for test use.
/// The temporary directory is kept (not cleaned up) so it outlives the worker.
pub(crate) fn test_performance_log() -> GenerationPerformanceLog {
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let leaked_temp_dir = temp_dir.keep();
    GenerationPerformanceLog::open(&leaked_temp_dir)
        .expect("test performance log file should be created")
}

pub(crate) async fn launch_test_executor(
    worker_executable_path: impl AsRef<Path>,
) -> Result<WorkerHandle, WorkerControlError> {
    launch_test_worker_with_model_load_timeout(
        worker_executable_path,
        DEFAULT_TEST_MODEL_LOAD_TIMEOUT,
    )
    .await
}

pub(crate) async fn launch_test_executor_with_performance_log_directory(
    worker_executable_path: impl AsRef<Path>,
    performance_log_directory: &Path,
) -> Result<WorkerHandle, WorkerControlError> {
    let performance_log = GenerationPerformanceLog::open(performance_log_directory)
        .expect("test performance log file should be created");
    WorkerHandle::launch(
        worker_executable_path,
        DEFAULT_TEST_MODEL_LOAD_TIMEOUT,
        performance_log,
        Arc::new(HashMap::new()),
        DEFAULT_TEST_MAX_OUTPUT_TOKENS,
    )
    .await
}

pub(crate) async fn launch_test_executor_with_cancellation_acknowledgement_timeout(
    worker_executable_path: impl AsRef<Path>,
    worker_cancellation_acknowledgement_timeout: Duration,
) -> Result<WorkerHandle, WorkerControlError> {
    WorkerHandle::launch_with_cancellation_acknowledgement_timeout(
        worker_executable_path,
        DEFAULT_TEST_MODEL_LOAD_TIMEOUT,
        worker_cancellation_acknowledgement_timeout,
        test_performance_log(),
        Arc::new(HashMap::new()),
        DEFAULT_TEST_MAX_OUTPUT_TOKENS,
    )
    .await
}

pub(crate) async fn launch_test_worker_with_model_load_timeout(
    worker_executable_path: impl AsRef<Path>,
    worker_model_load_timeout: Duration,
) -> Result<WorkerHandle, WorkerControlError> {
    WorkerHandle::launch(
        worker_executable_path,
        worker_model_load_timeout,
        test_performance_log(),
        Arc::new(HashMap::new()),
        DEFAULT_TEST_MAX_OUTPUT_TOKENS,
    )
    .await
}
