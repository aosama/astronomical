use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const NATIVE_PROBE_TIMEOUT: Duration = Duration::from_secs(105);
const NATIVE_PROBE_POLL_INTERVAL: Duration = Duration::from_millis(50);
const NATIVE_PROBE_PROGRESS_INTERVAL: Duration = Duration::from_secs(10);
/// The former lane script slept 5 seconds between the model-swap phase and
/// the native probe so Metal could finish releasing the previous process's
/// GPU state. The cargo alias runs the binaries back-to-back with no gap, so
/// the probe owns the wait now: `#[serial]` ordering is cross-process only
/// in spirit, and the probe is the phase that must not start mid-teardown.
const METAL_QUIESCENCE_WAIT: Duration = Duration::from_secs(5);

/// Resolves the native memory-contract probe from the shared native build
/// store. The acceptance package has no build script of its own, so instead
/// of the compile-time `ASTRONOMICAL_MLX_MEMORY_CONTRACT_PROBE` that
/// runtime-integration's build script emits, the probe is located through the
/// same identity/profile chain the build script uses (native identity
/// library →
/// store entry → `bin/mlx_memory_contract_probe` under the
/// `core+memory-contract` profile).
fn resolve_native_probe_executable_path() -> PathBuf {
    let repository_root = Path::new(
        &std::env::var("CARGO_MANIFEST_DIR")
            .expect("the acceptance package manifest directory should be set for the test process"),
    )
    .join("../..")
    .canonicalize()
    .expect("the repository root should resolve above the acceptance package");
    let identity_request = astronomical_native_identity::FingerprintRequest {
        repository_root: &repository_root,
        profile: astronomical_native_identity::NativeBuildProfile::CoreMemoryContract,
        scope: astronomical_native_identity::IdentityScope::Full,
        // The probe inherits the historical environment contract: fixture
        // overrides flow through the same variables the CI steps define.
        overrides: astronomical_native_identity::probes::CompatibilityOverrides::from_environment(),
    };
    let native_identity = astronomical_native_identity::native_build_identity(&identity_request)
        .unwrap_or_else(|error| panic!("the native build identity failed for the probe: {error}"));
    let store_root = std::env::var_os("ASTRONOMICAL_NATIVE_BUILD_STORE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                std::env::var("HOME").expect("HOME should be set to locate the native build store"),
            )
            .join("Library/Caches/Astronomical/native-builds")
        });
    let probe_executable_path = store_root
        .join("v1/entries")
        .join(&native_identity)
        .join("bin/mlx_memory_contract_probe");
    if !probe_executable_path.is_file() {
        panic!(
            "the native memory-contract probe was not found at {}; build it first with: \
             cargo test -p astronomical-runtime-integration --features mlx-memory-contract-probe --no-run",
            probe_executable_path.display()
        );
    }
    probe_executable_path
}

#[test]
#[ignore = "runs the pinned native MLX allocator probe in a fresh process"]
fn should_pass_the_pinned_native_mlx_memory_contract_probe() {
    // The probe subprocess inherits the published metallib location because
    // MLX's own baked-in staging path is removed right after the native build
    // store publishes an entry.
    let metallib_path = astronomical_runtime_integration::compiled_metallib_path().to_path_buf();
    astronomical_runtime_integration::validate_metallib_path(&metallib_path)
        .expect("the published native metallib should validate for the probe");
    let native_probe_executable_path = resolve_native_probe_executable_path();
    thread::sleep(METAL_QUIESCENCE_WAIT);
    let mut native_probe_child = Command::new(native_probe_executable_path)
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
