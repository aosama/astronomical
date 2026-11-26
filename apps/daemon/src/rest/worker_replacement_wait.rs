//! Holds a generation request while the worker is being replaced.
//!
//! The daemon replaces its worker after a configuration reload that needs a
//! new process, an idle unload or a contained failure. The replacement takes
//! seconds, and a request that arrives inside it would otherwise be refused
//! with HTTP 503 although the very next attempt succeeds. Clients such as
//! coding agents surface that refusal as an error, so the request waits here
//! for the new worker instead.

use std::time::Duration;

use tokio::sync::{Mutex as AsyncMutex, MutexGuard as AsyncMutexGuard};
use tokio::time::Instant;

use crate::WorkerHealthStatus;
use crate::application::ApplicationState;

/// Longest a request waits for a loading worker before it is refused.
///
/// A replacement closes the old worker and then loads the model under the
/// worker's 60-second load timeout, after which the status leaves `Loading`
/// either way. The extra quarter covers closing the old worker, so a request
/// that is refused here was refused because the load itself failed.
const WORKER_LOADING_WAIT_LIMIT: Duration = Duration::from_secs(75);

/// How often a waiting request looks at the worker again.
///
/// A replacement lasts seconds, so a tenth of a second adds no delay a person
/// would notice and keeps a waiting request from spinning on the status lock.
const WORKER_LOADING_RECHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Where a waiting request reads whether the worker is still loading.
trait WorkerStatusSource {
    /// The worker's availability as the worker loop last published it.
    fn worker_status(&self) -> WorkerHealthStatus;
}

impl WorkerStatusSource for ApplicationState {
    fn worker_status(&self) -> WorkerHealthStatus {
        self.generation_executor.worker_health_snapshot().status
    }
}

/// Takes the application's configuration-transition lock for a generation
/// request, waiting first for a worker that is being replaced.
pub(crate) async fn lock_configuration_transition_for_generation(
    application_state: &ApplicationState,
) -> AsyncMutexGuard<'_, ()> {
    lock_configuration_transition_once_worker_stops_loading(
        application_state,
        &application_state.configuration_transition_lock,
    )
    .await
}

/// Takes the configuration-transition lock once the worker has stopped
/// loading, or once the wait limit has passed.
///
/// The lock is released between looks: a configuration change needs it to
/// finish the very replacement this request is waiting for. A worker that is
/// `Unavailable` returns at once, so a request for a worker that is not
/// coming back is refused immediately as before. The caller reads the status
/// again under the returned guard and refuses the request if it still is not
/// ready.
async fn lock_configuration_transition_once_worker_stops_loading<'transition>(
    worker_status_source: &impl WorkerStatusSource,
    configuration_transition_lock: &'transition AsyncMutex<()>,
) -> AsyncMutexGuard<'transition, ()> {
    let wait_deadline = Instant::now() + WORKER_LOADING_WAIT_LIMIT;
    loop {
        let configuration_transition_guard = configuration_transition_lock.lock().await;
        let is_worker_loading = worker_status_source.worker_status() == WorkerHealthStatus::Loading;
        if !is_worker_loading || Instant::now() >= wait_deadline {
            return configuration_transition_guard;
        }
        drop(configuration_transition_guard);
        tokio::time::sleep(WORKER_LOADING_RECHECK_INTERVAL).await;
    }
}

#[cfg(test)]
#[path = "worker_replacement_wait_tests.rs"]
mod worker_replacement_wait_tests;
