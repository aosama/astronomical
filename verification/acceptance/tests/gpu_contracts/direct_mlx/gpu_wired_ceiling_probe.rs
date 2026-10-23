//! GPU wired-ceiling probe: the bottom-of-the-call-stack reproduction of the
//! 2026-10-15 decode degradation.
//!
//! Incident facts this probe interrogates: with a 41 GB configured ceiling on
//! a 48.96 GB machine, decode fell from 50.9 to 7.8 tokens per second while
//! MLX active memory plateaued at exactly 36,827,991,686 bytes — a number no
//! Astronomical policy chose (the ceiling was 41 GB and the machine reports
//! 48.96 GB). No application-level memory event fired during the degraded
//! decodes. The remaining suspect is the hardware-side GPU wired limit that
//! macOS enforces below total RAM; when MLX active memory presses against it,
//! per-allocation latency collapses even before outright failure.
//!
//! The probe allocates bf16 zero arrays in 512 MB steps under a 44 GB MLX
//! limit, holds every array resident, and reports per-step latency, MLX
//! active bytes, progress, and ETA. Whichever boundary appears — allocation
//! failure, latency crawl, or clean completion past the incident plateau —
//! is the fact that explains the plateau.
//!
//! >36 GB agreement (AGENTS.md test ceiling rule): Ahmed explicitly approved
//! exceeding the 36 GB configured-memory test ceiling on 2026-10-15 to
//! reproduce the 41 GB decode degradation; the probe caps at 43 GB so the
//! machine always keeps headroom for macOS.

use std::time::{Duration, Instant};

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

/// The plateau the incident session sat at through both degraded decodes.
const INCIDENT_ACTIVE_MEMORY_PLATEAU_BYTES: u64 = 36_827_991_686;
/// MLX limit high enough that MLX itself never refuses before the hardware.
const PROBE_MLX_ACTIVE_LIMIT_BYTES: usize = 44_000_000_000;
/// One probe step: 512 MB of bf16 weights, the granularity of one complete
/// expert table of the incident artifact.
const PROBE_STEP_BYTES: u64 = 512 * 1024 * 1024;
/// Never wire more than this, so macOS always keeps RAM headroom.
const PROBE_MAXIMUM_BYTES: u64 = 43_000_000_000;
/// OOM-reproduction timeout exception: the budget every step is checked
/// against (AGENTS.md allows exceeding 60 s for OOM reproductions).
const PROBE_BUDGET: Duration = Duration::from_secs(110);
/// Progress must be reported at least this often.
const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_secs(5);

#[test]
#[serial]
#[ignore = "wires GPU memory beyond the 36 GB test ceiling; run explicitly: ASTRONOMICAL_GPU_WIRED_CEILING_PROBE=1 (user agreement 2026-10-15)"]
fn should_expose_the_gpu_wired_ceiling_that_caps_mlx_active_memory() {
    if std::env::var("ASTRONOMICAL_GPU_WIRED_CEILING_PROBE")
        .ok()
        .as_deref()
        != Some("1")
    {
        panic!("set ASTRONOMICAL_GPU_WIRED_CEILING_PROBE=1 to run the GPU wired-ceiling probe");
    }
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(PROBE_MLX_ACTIVE_LIMIT_BYTES, PROBE_MLX_ACTIVE_LIMIT_BYTES)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let step_element_count =
        i32::try_from(PROBE_STEP_BYTES / 2).expect("one probe step fits the i32 element range");

    let probe_started_at = Instant::now();
    let mut held_arrays: Vec<MlxArray> = Vec::new();
    let mut held_bytes: u64 = 0;
    let mut plateau_reached = false;
    let mut last_report_at = Instant::now();
    let mut step_throughput_bytes_per_second;
    println!(
        "gpu wired-ceiling probe: step_bytes={PROBE_STEP_BYTES} incident_plateau_bytes={INCIDENT_ACTIVE_MEMORY_PLATEAU_BYTES} mlx_limit={PROBE_MLX_ACTIVE_LIMIT_BYTES}"
    );
    loop {
        if held_bytes + PROBE_STEP_BYTES > PROBE_MAXIMUM_BYTES {
            println!(
                "gpu wired-ceiling probe: completed the maximum probe span at held_bytes={held_bytes} without an allocation boundary"
            );
            break;
        }
        if probe_started_at.elapsed() > PROBE_BUDGET {
            println!(
                "gpu wired-ceiling probe: stopped on the {PROBE_BUDGET:?} budget at held_bytes={held_bytes}"
            );
            break;
        }
        let step_started_at = Instant::now();
        let step_allocation = match runtime.zeros(&[1, step_element_count], MlxDtype::BFloat16) {
            Ok(step_allocation) => step_allocation,
            Err(allocation_error) => {
                println!(
                    "ALLOCATION FAILED at held_bytes={held_bytes} step_elapsed_millis={} error={allocation_error}",
                    step_started_at.elapsed().as_millis()
                );
                break;
            }
        };
        if let Err(evaluation_error) = runtime.evaluate_arrays(&[&step_allocation]) {
            println!(
                "EVALUATION FAILED at held_bytes={held_bytes} step_elapsed_millis={} error={evaluation_error}",
                step_started_at.elapsed().as_millis()
            );
            break;
        }
        held_arrays.push(step_allocation);
        held_bytes += PROBE_STEP_BYTES;
        let step_elapsed = step_started_at.elapsed();
        step_throughput_bytes_per_second =
            PROBE_STEP_BYTES as f64 / step_elapsed.as_secs_f32().max(f32::EPSILON) as f64;
        if !plateau_reached && held_bytes >= INCIDENT_ACTIVE_MEMORY_PLATEAU_BYTES {
            plateau_reached = true;
            println!(
                "gpu wired-ceiling probe: PASSED the incident plateau at held_bytes={held_bytes} with step_latency_millis={} — the plateau is not a hardware wired boundary",
                step_elapsed.as_millis()
            );
        }
        if last_report_at.elapsed() >= PROGRESS_REPORT_INTERVAL
            || held_bytes + PROBE_STEP_BYTES > PROBE_MAXIMUM_BYTES
        {
            let remaining_bytes = PROBE_MAXIMUM_BYTES.saturating_sub(held_bytes);
            let eta_seconds = if step_throughput_bytes_per_second > 0.0 {
                remaining_bytes as f64 / step_throughput_bytes_per_second
            } else {
                f64::INFINITY
            };
            let mlx_active_bytes = runtime
                .memory_snapshot()
                .map(|memory_snapshot| memory_snapshot.active_memory_bytes())
                .unwrap_or(0);
            println!(
                "gpu wired-ceiling probe: held_gib={:.2} ({:.0}%) step_latency_millis={} mlx_active_bytes={} eta_seconds={:.1}",
                held_bytes as f64 / 1_073_741_824.0,
                100.0 * held_bytes as f64 / PROBE_MAXIMUM_BYTES as f64,
                step_elapsed.as_millis(),
                mlx_active_bytes,
                eta_seconds
            );
            last_report_at = Instant::now();
        }
    }
    let mlx_active_bytes_at_end = runtime
        .memory_snapshot()
        .map(|memory_snapshot| memory_snapshot.active_memory_bytes())
        .unwrap_or(0);
    println!(
        "gpu wired-ceiling probe: final held_bytes={held_bytes} mlx_active_bytes={mlx_active_bytes_at_end} elapsed_seconds={:.1} plateau_reached={plateau_reached}",
        probe_started_at.elapsed().as_secs_f32()
    );
    assert!(
        held_bytes >= PROBE_STEP_BYTES,
        "the probe must wire at least one step before any boundary"
    );
    drop(held_arrays);
    runtime
        .synchronize_gpu_stream()
        .expect("the probe should synchronize the stream after releasing its arrays");
}
