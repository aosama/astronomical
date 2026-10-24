//! Graphics-processor wired-ceiling allocation diagnostic.
//!
//! A previously observed active-memory plateau motivates checking whether
//! allocation latency degrades before the configured MLX limit. This probe
//! cannot assume the host's safe wired-memory capacity.
//!
//! The probe allocates bf16 zero arrays in 512 MB steps under a 44 GB MLX
//! limit, holds every array resident, and reports per-step latency, MLX
//! active bytes, progress, and ETA. Whichever boundary appears — allocation
//! failure, latency crawl, or clean completion past the incident plateau —
//! is the fact that explains the plateau.
//!
//! This ignored serial test refuses to initialize MLX until the explicit
//! `cargo xtask test-run` configuration contains
//! `"allow_large_wired_memory_probe": true`. Permission is run-scoped, never
//! inferred from the selected test name. The allocation span caps at 43 GB.

use std::time::{Duration, Instant};

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_test_harness::test_run::TestRunContext;
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

/// The plateau the incident session sat at through both degraded decodes.
const INCIDENT_ACTIVE_MEMORY_PLATEAU_BYTES: u64 = 36_827_991_686;
/// MLX limit high enough that MLX itself never refuses before the hardware.
const PROBE_MLX_ACTIVE_LIMIT_BYTES: usize = 44 * (1 << 30);
/// One probe step: 512 MB of bf16 weights, the granularity of one complete
/// expert table of the incident artifact.
const PROBE_STEP_BYTES: u64 = 512 * 1024 * 1024;
/// Bounds the explicit probe request without assuming the host's memory capacity.
const PROBE_MAXIMUM_BYTES: u64 = 43 * (1 << 30);
/// OOM-reproduction timeout exception: the budget every step is checked
/// against (AGENTS.md allows exceeding 60 s for OOM reproductions).
const PROBE_BUDGET: Duration = Duration::from_secs(110);
/// Progress must be reported at least this often.
const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_secs(5);

#[test]
#[serial]
#[ignore = "wires graphics-processor memory beyond the normal ceiling; requires allow_large_wired_memory_probe in explicit runner configuration"]
fn should_expose_the_gpu_wired_ceiling_that_caps_mlx_active_memory() {
    require_large_wired_memory_probe_permission(
        TestRunContext::current()
            .configuration
            .allow_large_wired_memory_probe,
    );
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
                "gpu wired-ceiling probe: held_gigabytes={:.2} ({:.0}%) step_latency_millis={} mlx_active_bytes={} eta_seconds={:.1}",
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

fn require_large_wired_memory_probe_permission(allow_large_wired_memory_probe: bool) {
    assert!(
        allow_large_wired_memory_probe,
        "set allow_large_wired_memory_probe to true in the configuration passed to cargo xtask test-run before running this probe"
    );
}

#[cfg(test)]
#[path = "gpu_wired_ceiling_probe_tests.rs"]
mod gpu_wired_ceiling_probe_tests;
