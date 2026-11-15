use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::Mutex as AsyncMutex;
use tokio::time::Instant;

use super::{
    WORKER_LOADING_WAIT_LIMIT, WorkerStatusSource,
    lock_configuration_transition_once_worker_stops_loading,
};
use crate::WorkerHealthStatus;

/// A worker that reports `Loading` for a set number of looks and then
/// settles on one status for good.
struct ScriptedWorkerStatus {
    loading_look_count: usize,
    settled_status: WorkerHealthStatus,
    observed_look_count: AtomicUsize,
}

impl ScriptedWorkerStatus {
    fn settling_after(loading_look_count: usize, settled_status: WorkerHealthStatus) -> Self {
        Self {
            loading_look_count,
            settled_status,
            observed_look_count: AtomicUsize::new(0),
        }
    }

    fn observed_look_count(&self) -> usize {
        self.observed_look_count.load(Ordering::SeqCst)
    }
}

impl WorkerStatusSource for ScriptedWorkerStatus {
    fn worker_status(&self) -> WorkerHealthStatus {
        let earlier_look_count = self.observed_look_count.fetch_add(1, Ordering::SeqCst);
        if earlier_look_count < self.loading_look_count {
            WorkerHealthStatus::Loading
        } else {
            self.settled_status
        }
    }
}

#[tokio::test(start_paused = true)]
async fn should_not_wait_when_the_worker_is_ready() {
    let worker_status = ScriptedWorkerStatus::settling_after(0, WorkerHealthStatus::Ready);
    let configuration_transition_lock = AsyncMutex::new(());
    let wait_started_at = Instant::now();

    let _configuration_transition_guard = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    )
    .await;

    assert_eq!(wait_started_at.elapsed(), Duration::ZERO);
    assert_eq!(worker_status.observed_look_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn should_not_wait_for_a_worker_that_is_unavailable() {
    let worker_status = ScriptedWorkerStatus::settling_after(0, WorkerHealthStatus::Unavailable);
    let configuration_transition_lock = AsyncMutex::new(());
    let wait_started_at = Instant::now();

    let _configuration_transition_guard = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    )
    .await;

    assert_eq!(wait_started_at.elapsed(), Duration::ZERO);
}

#[tokio::test(start_paused = true)]
async fn should_wait_until_the_replacement_worker_is_ready() {
    let worker_status = ScriptedWorkerStatus::settling_after(30, WorkerHealthStatus::Ready);
    let configuration_transition_lock = AsyncMutex::new(());
    let wait_started_at = Instant::now();

    let _configuration_transition_guard = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    )
    .await;

    assert_eq!(wait_started_at.elapsed(), Duration::from_secs(3));
    assert_eq!(worker_status.observed_look_count(), 31);
}

#[tokio::test(start_paused = true)]
async fn should_stop_waiting_as_soon_as_the_replacement_fails() {
    let worker_status = ScriptedWorkerStatus::settling_after(5, WorkerHealthStatus::Unavailable);
    let configuration_transition_lock = AsyncMutex::new(());
    let wait_started_at = Instant::now();

    let _configuration_transition_guard = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    )
    .await;

    assert_eq!(wait_started_at.elapsed(), Duration::from_millis(500));
}

#[tokio::test(start_paused = true)]
async fn should_give_up_at_the_wait_limit_when_the_worker_never_stops_loading() {
    let worker_status = ScriptedWorkerStatus::settling_after(usize::MAX, WorkerHealthStatus::Ready);
    let configuration_transition_lock = AsyncMutex::new(());
    let wait_started_at = Instant::now();

    let _configuration_transition_guard = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    )
    .await;

    assert_eq!(wait_started_at.elapsed(), WORKER_LOADING_WAIT_LIMIT);
}

#[tokio::test(start_paused = true)]
async fn should_leave_the_lock_free_for_the_replacement_while_waiting() {
    let worker_status = ScriptedWorkerStatus::settling_after(usize::MAX, WorkerHealthStatus::Ready);
    let configuration_transition_lock = AsyncMutex::new(());
    let waiting_request = lock_configuration_transition_once_worker_stops_loading(
        &worker_status,
        &configuration_transition_lock,
    );
    tokio::pin!(waiting_request);

    // One poll takes the request to its first sleep between looks.
    let first_look = tokio::time::timeout(Duration::from_millis(1), &mut waiting_request).await;

    assert!(first_look.is_err(), "the request must still be waiting");
    assert!(
        configuration_transition_lock.try_lock().is_ok(),
        "a waiting request must not hold the lock the replacement needs"
    );
}
