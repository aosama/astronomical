//! The last finished reply's peak MLX memory, as the daemon publishes it.

use std::sync::{Arc, RwLock};

use super::{WorkerHealthSnapshot, WorkerHealthStatus, record_reply_peak_mlx_memory};

fn health_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )))
}

fn published_peak(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> Option<u64> {
    health_snapshot
        .read()
        .expect("the snapshot is readable")
        .last_reply_peak_mlx_memory_bytes
}

#[test]
fn should_report_no_peak_before_any_reply_finishes() {
    assert_eq!(published_peak(&health_snapshot()), None);
}

#[test]
fn should_keep_the_last_replys_peak_and_not_an_earlier_higher_one() {
    let health_snapshot = health_snapshot();
    record_reply_peak_mlx_memory(&health_snapshot, Some(18 << 30));
    record_reply_peak_mlx_memory(&health_snapshot, Some(12 << 30));

    assert_eq!(published_peak(&health_snapshot), Some(12 << 30));
}

#[test]
fn should_clear_the_peak_when_a_reply_finished_with_no_memory_reading() {
    let health_snapshot = health_snapshot();
    record_reply_peak_mlx_memory(&health_snapshot, Some(18 << 30));
    record_reply_peak_mlx_memory(&health_snapshot, None);

    assert_eq!(published_peak(&health_snapshot), None);
}
