use std::{
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use astronomical_runtime_integration::{compiled_metallib_path, validate_metallib_path};

const NATIVE_PROBE_TIMEOUT: Duration = Duration::from_secs(105);
const NATIVE_PROBE_POLL_INTERVAL: Duration = Duration::from_millis(50);
const NATIVE_PROBE_PROGRESS_INTERVAL: Duration = Duration::from_secs(10);

#[test]
#[ignore = "runs the pinned native MLX allocator probe in a fresh process"]
fn should_pass_the_pinned_native_mlx_memory_contract_probe() {
    // The probe subprocess inherits the published metallib location because
    // MLX's own baked-in staging path is removed right after the native build
    // store publishes an entry.
    let metallib_path = compiled_metallib_path().to_path_buf();
    validate_metallib_path(&metallib_path)
        .expect("the published native metallib should validate for the probe");
    let mut native_probe_child = Command::new(env!("ASTRONOMICAL_MLX_MEMORY_CONTRACT_PROBE"))
        .env("ASTRONOMICAL_MLX_METALLIB_PATH", &metallib_path)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the feature-gated native MLX memory contract probe should be executable");
    let probe_started_at = Instant::now();
    let mut last_progress_report_at = probe_started_at;
    eprintln!(
        "[mlx-memory-contract] status=start phase=native_probe timeout_seconds={}",
        NATIVE_PROBE_TIMEOUT.as_secs(),
    );

    loop {
        if let Some(probe_exit_status) = native_probe_child
            .try_wait()
            .expect("the native MLX memory contract probe status should be readable")
        {
            assert!(
                probe_exit_status.success(),
                "the native MLX memory contract probe failed with {probe_exit_status}"
            );
            eprintln!(
                "[mlx-memory-contract] status=success phase=native_probe elapsed_seconds={}",
                probe_started_at.elapsed().as_secs(),
            );
            return;
        }
        if probe_started_at.elapsed() >= NATIVE_PROBE_TIMEOUT {
            let native_probe_kill_error = native_probe_child.kill().err();
            let native_probe_exit_status = native_probe_child
                .wait()
                .expect("the timed-out native MLX memory contract probe should be reaped");
            panic!(
                "the native MLX memory contract probe exceeded {} seconds; exit_status={native_probe_exit_status}; kill_error={native_probe_kill_error:?}",
                NATIVE_PROBE_TIMEOUT.as_secs(),
            );
        }
        if last_progress_report_at.elapsed() >= NATIVE_PROBE_PROGRESS_INTERVAL {
            eprintln!(
                "[mlx-memory-contract] status=progress phase=native_probe elapsed_seconds={} timeout_seconds={}",
                probe_started_at.elapsed().as_secs(),
                NATIVE_PROBE_TIMEOUT.as_secs(),
            );
            last_progress_report_at = Instant::now();
        }
        thread::sleep(NATIVE_PROBE_POLL_INTERVAL);
    }
}
