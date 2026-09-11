//! Applies InitializeWorker runtime-policy acknowledgement before SwapModel.
//!
//! Launch writes InitializeWorker and returns as soon as the process is
//! spawned. Idle can mark the worker ready while that acknowledgement is
//! still in the pipe. A SwapModel in that window treats a missing health
//! generation as "policy already ready", then the later
//! RuntimeFeatureConfigurationApplied looks like an illegal loaded-model
//! change and the supervisor contains the worker.
//!
//! The wait is once per worker process. Live memory updates may change the
//! configuration generation, and a rejected swap may clear it from health;
//! neither should wait for InitializeWorker again.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::time::{Instant, timeout};

use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, WorkerControlError, WorkerHealthSnapshot,
    WorkerProcess, worker_event_handler::handle_worker_event,
    worker_loop_types::ActiveWorkerRequest,
};

#[allow(clippy::too_many_arguments)]
pub(super) async fn wait_for_startup_runtime_configuration(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    model_load_timeout: Duration,
) -> Result<(), WorkerControlError> {
    if worker_process.expected_configuration_generation().is_none() {
        return Ok(());
    }
    if worker_process.startup_runtime_configuration_applied() {
        return Ok(());
    }
    if health_has_runtime_configuration(health_snapshot) {
        worker_process.mark_startup_runtime_configuration_applied();
        return Ok(());
    }
    timeout(
        model_load_timeout,
        drain_until_startup_runtime_configuration(
            worker_process,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_request,
            performance_log,
            completion_log,
        ),
    )
    .await
    .map_err(|_| WorkerControlError::ModelLoadTimeout {
        model_load_timeout_millis: model_load_timeout.as_millis(),
    })?
}

#[allow(clippy::too_many_arguments)]
async fn drain_until_startup_runtime_configuration(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    loop {
        if health_has_runtime_configuration(health_snapshot) {
            worker_process.mark_startup_runtime_configuration_applied();
            return Ok(());
        }
        let worker_event = worker_process
            .next_event()
            .await?
            .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
        handle_worker_event(
            worker_event,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_request,
            performance_log,
            completion_log,
        )?;
    }
}

fn health_has_runtime_configuration(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> bool {
    health_snapshot
        .read()
        .ok()
        .is_some_and(|snapshot| snapshot.worker_runtime_feature_configuration.is_some())
}
