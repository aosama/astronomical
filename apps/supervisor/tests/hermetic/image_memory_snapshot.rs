//! End-to-end coverage for live MLX memory snapshots published from image
//! render step boundaries (menu shows a measured footprint mid-render).

use std::time::Duration;

use astronomical_ipc_protocol::MlxMemorySnapshotSource;
use astronomical_supervisor::{ChatGenerationExecutor, ImageGenerationExecutor};
use tokio::time::{Instant, sleep, timeout};

use super::image_generation;

#[tokio::test]
async fn should_publish_a_live_mlx_memory_snapshot_from_image_progress_steps() {
    let worker_handle = image_generation::launch_scripted_worker().await;
    let mut image_receiver = worker_handle
        .start_image_generation(image_generation::image_command(
            120,
            "progress-snapshot-image-fixture",
        ))
        .await
        .expect("the snapshot-carrying image should start");

    wait_for_latest_memory_snapshot_source(
        &worker_handle,
        MlxMemorySnapshotSource::ImageGenerationStep,
        "the image render snapshot should surface mid-generation",
    )
    .await;
    let render_snapshot = worker_handle
        .worker_health_snapshot()
        .latest_mlx_memory_snapshot
        .expect("the render snapshot should remain readable");
    assert!(render_snapshot.active_memory_bytes > 0);

    let image_outcome = timeout(Duration::from_secs(3), image_receiver.recv())
        .await
        .expect("the snapshot image should finish")
        .expect("the image outcome should arrive");
    assert!(
        image_outcome.is_ok(),
        "snapshot image failed: {image_outcome:?}"
    );

    wait_for_latest_memory_snapshot_source(
        &worker_handle,
        MlxMemorySnapshotSource::Finalized,
        "the finalized cleanup snapshot should replace the render sample",
    )
    .await;
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}

async fn wait_for_latest_memory_snapshot_source(
    worker_handle: &astronomical_supervisor::WorkerHandle,
    expected_source: MlxMemorySnapshotSource,
    timeout_message: &str,
) {
    let snapshot_deadline = Instant::now() + Duration::from_secs(2);
    while !worker_handle
        .worker_health_snapshot()
        .latest_mlx_memory_snapshot
        .is_some_and(|snapshot| snapshot.source == expected_source)
    {
        assert!(Instant::now() < snapshot_deadline, "{timeout_message}");
        sleep(Duration::from_millis(10)).await;
    }
}
