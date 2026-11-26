//! What chat and image admission both need when they start a request: the
//! channel that answers the caller, the published model and configuration
//! the swap decision reads, and containment of a worker that failed while
//! the request was starting.
//!
//! Neither admission owns these, so neither imports the other. The bounded
//! model-swap wait they also share lives with the swap itself in
//! `process::model_swap`.

use std::sync::{Arc, RwLock};

use crate::worker::loop_types::WorkerLoopState;
use crate::worker::process::containment::contain_worker_failure;
use crate::{GenerationStartError, WorkerControlError, WorkerHealthSnapshot};

/// Answers the caller of a generation: `Ok` once it runs, or why it did not start.
pub(in crate::worker) type GenerationStartSender =
    tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>;

/// Contains a worker that failed while a request was starting, and marks the
/// loop not ready so no further request is admitted to it.
pub(in crate::worker) async fn contain_failed_worker(
    worker_loop_state: &mut WorkerLoopState<'_>,
    worker_failure: WorkerControlError,
) {
    contain_worker_failure(
        worker_loop_state.worker_process,
        worker_loop_state.health_snapshot,
        worker_loop_state.active_request,
        worker_failure,
    )
    .await;
    *worker_loop_state.is_ready = false;
}

/// The model the worker last published as ready, if any.
pub(in crate::worker) fn loaded_model_id(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Option<String> {
    health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone())
}

/// The configuration generation the worker last acknowledged, if any. A model
/// swap's wait compares the worker's next acknowledgement against it.
pub(in crate::worker) fn published_configuration_generation(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Option<String> {
    health_snapshot.read().ok().and_then(|snapshot| {
        snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .map(|configuration| configuration.configuration_generation.clone())
    })
}

#[cfg(test)]
#[path = "request_start_tests.rs"]
mod request_start_tests;
