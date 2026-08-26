use std::{
    future::Future,
    path::PathBuf,
    sync::LazyLock,
    time::{Duration, Instant},
};

use astronomical_config::AstronomicalConfig;
use tokio::{
    sync::{Mutex, MutexGuard},
    time::{MissedTickBehavior, interval, sleep},
};

mod aligned_expert_pack_preparation;
mod aligned_expert_pack_projection;
mod expert_storage_data_plane_measurement;

const ALIGNED_EXPERT_PACK_TEST_TIMEOUT: Duration = Duration::from_secs(120);
const ORNITH_OQ6_MODEL_ID: &str = "Ornith-1.5-35B-A3B-oQ6e-mtp";
const ORNITH_OQ6_PROVIDER_MODEL_ID: &str = "scottlowry/Ornith-1.5-35B-A3B-oQ6e-mtp";
static DIRECT_MLX_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

async fn direct_mlx_test_guard() -> MutexGuard<'static, ()> {
    DIRECT_MLX_TEST_LOCK.lock().await
}

fn configured_model_directory_by_id(model_id: &str) -> Option<PathBuf> {
    AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for experiment qualification")
        .find_configured_model_directory_by_id(model_id)
        .unwrap_or_else(|discovery_error| {
            panic!("model directory discovery should complete for {model_id}: {discovery_error}")
        })
}

async fn require_aligned_expert_pack_completion(test_future: impl Future<Output = ()>) {
    let started_at = Instant::now();
    let timeout_deadline = sleep(ALIGNED_EXPERT_PACK_TEST_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    loop {
        tokio::select! {
            () = &mut test_future => return,
            () = &mut timeout_deadline => panic!("the experimental aligned expert-pack qualification exceeded {} seconds", ALIGNED_EXPERT_PACK_TEST_TIMEOUT.as_secs()),
            _ = progress_interval.tick() => eprintln!(
                "[experimental-aligned-expert-pack] status=running elapsed_seconds={:.0} ETA_seconds<={:.0}",
                started_at.elapsed().as_secs_f64(),
                ALIGNED_EXPERT_PACK_TEST_TIMEOUT.saturating_sub(started_at.elapsed()).as_secs_f64()
            ),
        }
    }
}
