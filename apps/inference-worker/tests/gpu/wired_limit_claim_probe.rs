//! Repeatable probe: how far can GPU memory actually be wired under the
//! default kernel policy, and does an MLX wired-limit claim extend it?
//!
//! Standing evidence for the machine-ceiling policy (physical RAM minus
//! 8 sticker-scale GB): measured on the reference machine (sold as 48 GB,
//! macOS 27, `iogpu.wired_limit_mb=0`, idle) the default policy wired past
//! the full product target of 42,949,672,960 bytes (the 40 GB on the
//! sticker scale) with every probe healthy (23–215 ms) — past the kernel's
//! own recommendation. Performance lesson recorded twice since: the idle
//! default is a heuristic, not a guarantee. On 2026-10-16, with a 34 GB
//! resident model loaded on the same machine under real neighbor load
//! (compressor holding ~25 GB, swap full, pressure level 2), the default
//! heuristic had wired only ~3.8 GB — macOS compressed the model's own
//! weights and the first prefill after load faulted them back at 43–53 s
//! per small chunk. The worker therefore claims wired residency up to the
//! kernel recommendation by default (`worker_startup`), which is
//! deterministic where the idle heuristic is not; this probe remains the
//! measured boundary walk for both policies. The claim fallback in this
//! probe exists for machines where the default policy genuinely crawls
//! below the recommendation.
//!
//! Probe semantics mirror the production startup walk so both agree on
//! what a boundary is: geometric probes (512 MiB step granularity, 2 GiB
//! cap), one probe is "slow" past 2 s, two consecutive slow probes confirm
//! the crawl boundary, each walk carries a 20 s budget, and the whole test
//! a 60 s budget. Holding the walk target briefly (~RAM minus 8 GB) is the
//! measurement; the two-slow-probe rule bounds any crawl window.
//!
//! Machine-state validity: run while the daemon holds no model so nothing
//! else contends for the wired pool. A claim is process-local and is
//! restored on every exit path through a Drop guard.

use std::process::Command;
use std::time::{Duration, Instant};

use astronomical_runtime_integration::{
    MlxMemoryLimits, MlxRuntime, maximum_recommended_gpu_working_set_size_bytes,
};
use mlx_c_rust::{MlxArray, MlxDtype, set_wired_limit};
use serial_test::serial;

const RUN_PROBE_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_RUN_WIRED_CLAIM_PROBE";
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";
const PROBE_STEP_BYTES: usize = 512 * 1024 * 1024;
const PROBE_MAXIMUM_BYTES: usize = 2 * 1024 * 1024 * 1024;
const PROBE_LATENCY_BUDGET: Duration = Duration::from_secs(2);
const WALK_TOTAL_BUDGET: Duration = Duration::from_secs(20);
const WHOLE_TEST_BUDGET: Duration = Duration::from_secs(60);
/// The operating system keeps this much of physical RAM for itself — 8 GB
/// on the same sticker scale users know; the walk never aims beyond it so
/// the probe cannot starve the host.
const OPERATING_SYSTEM_SURVIVAL_HEADROOM_BYTES: usize = 8 * 1_073_741_824;
/// Marker from the vendored active-memory-ceiling patch; hitting it means
/// the walk reached its own configured MLX limit, not a kernel boundary.
const MLX_ACTIVE_CEILING_ERROR_MARKER: &str = "ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED";

#[derive(Clone, Debug, Eq, PartialEq)]
enum WalkStopReason {
    TargetReached,
    CrawlBoundary,
    MlxActiveCeiling,
    AllocationFailed(String),
    EvaluationFailed(String),
    BudgetExhausted,
}

struct WalkOutcome {
    healthy_wired_bytes: usize,
    stop_reason: WalkStopReason,
}

/// Restores the wired limit even when an assertion fails or the probe is
/// interrupted mid-walk; changing it while arrays are alive would leak the
/// claim past the test's lifetime.
struct WiredLimitClaimGuard {
    previous_limit_bytes: usize,
}

impl Drop for WiredLimitClaimGuard {
    fn drop(&mut self) {
        if let Err(restore_error) = set_wired_limit(self.previous_limit_bytes) {
            eprintln!("WARN could not restore the wired limit: {restore_error}");
        }
    }
}

#[test]
#[serial]
fn should_extend_the_wired_boundary_beyond_default_policy_when_mlx_claims_a_wired_limit() {
    if !probe_is_requested() {
        eprintln!(
            "skipping: set {RUN_PROBE_ENVIRONMENT_VARIABLE}=1 to run the wired-limit claim probe"
        );
        return;
    }
    let test_started_at = Instant::now();
    let test_deadline = test_started_at + WHOLE_TEST_BUDGET;

    let kernel_wired_limit_mebibytes =
        read_sysctl_integer("iogpu.wired_limit_mb").expect("reading iogpu.wired_limit_mb");
    let physical_ram_bytes = read_sysctl_integer("hw.memsize").expect("reading hw.memsize");
    let recommended_working_set_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("reading the MLX recommended working set");
    let walk_target_bytes =
        physical_ram_bytes.saturating_sub(OPERATING_SYSTEM_SURVIVAL_HEADROOM_BYTES as u64) as usize;
    eprintln!(
        "machine: ram={physical_ram_bytes} bytes, sysctl iogpu.wired_limit_mb={kernel_wired_limit_mebibytes}, recommended_working_set={recommended_working_set_bytes} bytes, walk_target={walk_target_bytes} bytes"
    );

    // Soft limits sit at the walk target, not at any estimate, so MLX's own
    // reclamation (and the patched hard active ceiling) never interfere
    // below the boundary the kernel itself enforces.
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(walk_target_bytes, walk_target_bytes)
            .expect("forming MLX memory limits from the walk target"),
    )
    .expect("initializing the MLX runtime for the probe");

    eprintln!("--- walk 1: no wired-limit claim (default kernel policy) ---");
    let unclaimed_walk = walk_wired_boundary(&runtime, walk_target_bytes, test_deadline);
    report_walk("unclaimed", &unclaimed_walk);

    if walk_reached_target(&unclaimed_walk, walk_target_bytes) {
        eprintln!(
            "verdict: the default kernel policy wires the full product target of RAM minus 8 GB; no wired-limit claim and no elevation are needed"
        );
        return;
    }

    let above_recommended_claim_bytes = recommended_working_set_bytes + 8 * 1024 * 1024;
    assert!(
        set_wired_limit(above_recommended_claim_bytes).is_err(),
        "claiming above the system wired limit must be rejected; that wall is exactly what requires elevation"
    );

    let claim_bytes = recommended_working_set_bytes.min(walk_target_bytes);
    eprintln!("--- walk 2: wired-limit claim of {claim_bytes} bytes ---");
    let previous_limit_bytes = set_wired_limit(claim_bytes).expect(
        "claiming a wired limit up to the recommended working set must succeed without elevation",
    );
    eprintln!("previous wired limit reported by MLX: {previous_limit_bytes} bytes");
    let _claim_guard = WiredLimitClaimGuard {
        previous_limit_bytes,
    };
    let claimed_walk = walk_wired_boundary(&runtime, walk_target_bytes, test_deadline);
    report_walk("claimed", &claimed_walk);

    let boundary_advanced_bytes = claimed_walk
        .healthy_wired_bytes
        .saturating_sub(unclaimed_walk.healthy_wired_bytes);
    eprintln!(
        "verdict: unclaimed boundary {:.2} GB, claimed boundary {:.2} GB, advanced by {:.2} GB (walk target {:.2} GB, sticker scale)",
        unclaimed_walk.healthy_wired_bytes as f64 / 1_073_741_824.0,
        claimed_walk.healthy_wired_bytes as f64 / 1_073_741_824.0,
        boundary_advanced_bytes as f64 / 1_073_741_824.0,
        walk_target_bytes as f64 / 1_073_741_824.0,
    );
    assert!(
        walk_reached_target(&claimed_walk, walk_target_bytes)
            || boundary_advanced_bytes >= PROBE_STEP_BYTES,
        "neither the default policy nor a wired-limit claim wires RAM minus 8 GB (unclaimed {} bytes, claimed {} bytes)",
        unclaimed_walk.healthy_wired_bytes,
        claimed_walk.healthy_wired_bytes
    );
}

fn walk_reached_target(walk_outcome: &WalkOutcome, target_bytes: usize) -> bool {
    walk_outcome
        .healthy_wired_bytes
        .saturating_add(PROBE_STEP_BYTES)
        > target_bytes
}

fn probe_is_requested() -> bool {
    std::env::var(RUN_PROBE_ENVIRONMENT_VARIABLE)
        .ok()
        .as_deref()
        == Some("1")
}

fn read_sysctl_integer(sysctl_key: &str) -> Option<u64> {
    let sysctl_output = Command::new(SYSCTL_EXECUTABLE_PATH)
        .arg("-n")
        .arg(sysctl_key)
        .output()
        .ok()?;
    if !sysctl_output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&sysctl_output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

fn next_probe_bytes(previous_probe_bytes: usize, remaining_bytes: usize) -> usize {
    let doubled_probe_bytes = previous_probe_bytes.saturating_mul(2);
    let capped_probe_bytes = doubled_probe_bytes
        .min(PROBE_MAXIMUM_BYTES)
        .max(PROBE_STEP_BYTES);
    capped_probe_bytes.min(remaining_bytes)
}

fn walk_wired_boundary(
    runtime: &MlxRuntime,
    target_bytes: usize,
    test_deadline: Instant,
) -> WalkOutcome {
    let walk_started_at = Instant::now();
    let mut held_arrays: Vec<MlxArray> = Vec::new();
    let mut healthy_wired_bytes = 0_usize;
    let mut previous_probe_bytes = PROBE_STEP_BYTES;
    let mut consecutive_slow_probes = 0_usize;
    let mut stop_reason = WalkStopReason::TargetReached;

    while healthy_wired_bytes.saturating_add(PROBE_STEP_BYTES) <= target_bytes {
        if Instant::now() > test_deadline || walk_started_at.elapsed() > WALK_TOTAL_BUDGET {
            stop_reason = WalkStopReason::BudgetExhausted;
            break;
        }
        let remaining_bytes = target_bytes - healthy_wired_bytes;
        let probe_bytes = if consecutive_slow_probes > 0 {
            // Refinement mode: after one slow probe the walk continues in
            // step-granular probes so the clamp lands within one step of
            // the true boundary (same discipline as the production walk).
            PROBE_STEP_BYTES
        } else {
            next_probe_bytes(previous_probe_bytes, remaining_bytes)
        };
        let probe_element_count =
            i32::try_from(probe_bytes / 2).expect("probe stays inside the i32 element range");
        let probe_started_at = Instant::now();
        let step_allocation = match runtime.zeros(&[1, probe_element_count], MlxDtype::BFloat16) {
            Ok(step_allocation) => step_allocation,
            Err(allocation_error) => {
                stop_reason = classify_allocation_failure(&allocation_error.to_string());
                eprintln!("probe of {probe_bytes} bytes failed to allocate: {allocation_error}");
                break;
            }
        };
        if let Err(evaluation_error) = runtime.evaluate_arrays(&[&step_allocation]) {
            stop_reason = WalkStopReason::EvaluationFailed(evaluation_error.to_string());
            eprintln!("probe of {probe_bytes} bytes failed to evaluate: {evaluation_error}");
            break;
        }
        let probe_elapsed = probe_started_at.elapsed();
        if probe_elapsed > PROBE_LATENCY_BUDGET {
            consecutive_slow_probes += 1;
            eprintln!(
                "slow probe: {probe_bytes} bytes took {} ms (budget {} ms), consecutive={consecutive_slow_probes}",
                probe_elapsed.as_millis(),
                PROBE_LATENCY_BUDGET.as_millis(),
            );
            if consecutive_slow_probes >= 2 {
                stop_reason = WalkStopReason::CrawlBoundary;
                break;
            }
            continue;
        }
        consecutive_slow_probes = 0;
        healthy_wired_bytes += probe_bytes;
        previous_probe_bytes = probe_bytes;
        held_arrays.push(step_allocation);
        eprintln!(
            "healthy probe: {probe_bytes} bytes in {} ms (wired so far {:.2} GB)",
            probe_elapsed.as_millis(),
            healthy_wired_bytes as f64 / 1_073_741_824.0,
        );
    }

    drop(held_arrays);
    if let Err(cleanup_error) = runtime.synchronize_gpu_stream_and_clear_allocator_cache() {
        eprintln!("WARN allocator cache cleanup after the walk failed: {cleanup_error}");
    }
    WalkOutcome {
        healthy_wired_bytes,
        stop_reason,
    }
}

fn classify_allocation_failure(error_message: &str) -> WalkStopReason {
    if error_message.contains(MLX_ACTIVE_CEILING_ERROR_MARKER) {
        WalkStopReason::MlxActiveCeiling
    } else {
        WalkStopReason::AllocationFailed(error_message.to_owned())
    }
}

fn report_walk(walk_label: &str, walk_outcome: &WalkOutcome) {
    eprintln!(
        "{walk_label} walk: healthy {:.2} GB, stopped because {:?}",
        walk_outcome.healthy_wired_bytes as f64 / 1_073_741_824.0,
        walk_outcome.stop_reason,
    );
}
