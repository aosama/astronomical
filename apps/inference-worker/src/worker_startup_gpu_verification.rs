//! Startup verification that the machine can actually wire the effective
//! graphics-processor memory ceiling (issue #1279).
//!
//! Reading the ceiling is not enough: under the default kernel wired policy
//! the enforced boundary (~66–75% of memory, machine dependent) is not exposed
//! by any query, and a ceiling composed from the larger recommended working
//! set can be unreachable in practice. The incident signature is silent:
//! active memory plateaus at the boundary while decode throughput collapses,
//! with no application-level memory event. Before the worker promises a
//! ceiling it therefore wires the memory in geometrically growing probes,
//! measures per-probe latency, refines in step-granular probes once a probe
//! signals the boundary, and clamps the effective ceiling to the last healthy
//! wired byte count when the boundary answers with a crawl instead of an
//! error.
//!
//! An explicit environment override skips verification for acceptance cells
//! that deliberately serve beyond the boundary to exercise SSD streaming.

use std::time::{Duration, Instant};

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// Wiring granularity: one complete expert table of a large sparse artifact.
/// Also the clamp refinement granularity: after a slow probe signals the
/// boundary, only this-sized steps continue, so a clamp lands within one
/// expert table of the true boundary.
pub(crate) const VERIFICATION_STEP_BYTES: usize = 512 * 1024 * 1024;
/// Largest single probe: geometric growth doubles probe sizes from the step
/// granularity up to this cap. The cap bounds every probe on a throttled
/// machine (a crawl manifests as one slow probe of at most this size, never
/// an unbounded stall), and it stays inside the element-count range of one
/// bfloat16 array.
pub(crate) const VERIFICATION_MAX_PROBE_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// A healthy step wires in tens of milliseconds; thirty times that is the
/// boundary's crawl signature, not slow hardware.
pub(crate) const VERIFICATION_STEP_LATENCY_BUDGET: Duration = Duration::from_millis(2_000);
/// The whole verification must stay bounded even on large ceilings.
pub(crate) const VERIFICATION_TOTAL_BUDGET: Duration = Duration::from_secs(20);
/// Skipping is reserved for deliberate over-boundary acceptance testing and
/// for offline tests that spawn the production worker: offline tests are
/// sealed from machine state by definition, and the walk is machine state.
pub(crate) const SKIP_VERIFICATION_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_SKIP_GPU_WIRED_VERIFICATION";

/// One step's latency verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VerificationStepVerdict {
    /// The step wired within budget; verification may continue.
    Continue,
    /// The step exceeded the latency budget: the operating system is
    /// throttling wiring at its boundary. Verification must stop here.
    BoundaryHit,
}

/// The final outcome of verifying one candidate ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WiredCapacityVerification {
    /// Every step up to the candidate wired within budget.
    Verified { wired_bytes: usize },
    /// A step hit the boundary; the ceiling must clamp to the last healthy
    /// step.
    ClampedToLastHealthyStep { wired_bytes: usize },
}

/// Classifies one wiring step's elapsed time.
#[must_use]
pub(crate) fn verification_step_verdict(
    step_elapsed: Duration,
    step_latency_budget: Duration,
) -> VerificationStepVerdict {
    if step_elapsed > step_latency_budget {
        VerificationStepVerdict::BoundaryHit
    } else {
        VerificationStepVerdict::Continue
    }
}

/// Next probe size for the geometric walk: doubles from the step granularity
/// up to the probe cap, and never overshoots the remaining distance.
/// Performance lesson from the 2026-10-15 verification rollout: a fixed
/// 512 MiB linear walk costs one allocate-and-synchronize round trip per
/// step — 94 round trips on a 48 GB machine, and every parallel test worker
/// paid machine-scale walks that starved sibling processes. Geometric probes
/// verify the same total bytes with a logarithmic number of round trips
/// while every probe stays bounded by the cap.
#[must_use]
pub(crate) fn next_verification_probe_bytes(
    previous_probe_bytes: usize,
    remaining_bytes: usize,
) -> usize {
    let doubled_probe_bytes = previous_probe_bytes.saturating_mul(2);
    let capped_probe_bytes = doubled_probe_bytes
        .min(VERIFICATION_MAX_PROBE_BYTES)
        .max(VERIFICATION_STEP_BYTES);
    capped_probe_bytes.min(remaining_bytes)
}

/// Wires `candidate_ceiling_bytes` of graphics memory in steps and reports the
/// highest healthy wired byte count.
///
/// The runtime is initialized with the candidate as its MLX limit so MLX's own
/// reclamation never interferes: only the operating system's wired policy can
/// answer. Every array is evaluated (zero pages mapped), held resident across
/// the whole walk, and released before returning, leaving the worker's memory
/// state clean for model loading.
pub(crate) async fn verify_effective_gpu_wired_capacity(
    candidate_ceiling_bytes: usize,
) -> Result<WiredCapacityVerification, crate::worker_startup_error::WorkerStartupError> {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(candidate_ceiling_bytes, candidate_ceiling_bytes).map_err(|_| {
            crate::worker_startup_error::WorkerStartupError::InvalidGpuWiredCapacityVerificationGeometry {
                description: "the candidate ceiling cannot form valid MLX memory limits",
            }
        })?,
    )
    .map_err(crate::worker_startup_error::WorkerStartupError::InitializeVerificationRuntime)?;
    let verification_started_at = Instant::now();
    let mut held_arrays: Vec<MlxArray> = Vec::new();
    let mut wired_bytes = 0_usize;
    let mut slowest_healthy_step = Duration::ZERO;
    let mut outcome = WiredCapacityVerification::Verified { wired_bytes: 0 };
    let mut floor_reached = false;
    // The walk runs two phases: geometric probes (doubling to the cap) while
    // every probe stays healthy, then — only after a slow probe signals the
    // boundary — 512 MiB refinement steps so the clamp lands within one
    // expert table of the true boundary instead of on the coarse geometric
    // grid.
    let mut previous_probe_bytes = VERIFICATION_STEP_BYTES;
    let mut boundary_hit = false;
    while wired_bytes + VERIFICATION_STEP_BYTES <= candidate_ceiling_bytes {
        if verification_started_at.elapsed() > VERIFICATION_TOTAL_BUDGET {
            tracing::warn!(
                wired_bytes,
                candidate_ceiling_bytes,
                "GPU wired-capacity verification stopped on its total budget"
            );
            outcome = WiredCapacityVerification::ClampedToLastHealthyStep { wired_bytes };
            break;
        }
        let probe_bytes = if boundary_hit {
            VERIFICATION_STEP_BYTES
        } else {
            next_verification_probe_bytes(
                previous_probe_bytes,
                candidate_ceiling_bytes - wired_bytes,
            )
        };
        let probe_element_count = i32::try_from(probe_bytes / 2).map_err(|_| {
            crate::worker_startup_error::WorkerStartupError::InvalidGpuWiredCapacityVerificationGeometry {
                description: "one verification probe exceeds the i32 element range",
            }
        })?;
        let step_started_at = Instant::now();
        let step_allocation = match runtime.zeros(&[1, probe_element_count], MlxDtype::BFloat16) {
            Ok(step_allocation) => step_allocation,
            Err(allocation_error) => {
                tracing::warn!(
                    wired_bytes,
                    error = %allocation_error,
                    "GPU wired-capacity verification hit an allocation failure"
                );
                outcome = WiredCapacityVerification::ClampedToLastHealthyStep { wired_bytes };
                break;
            }
        };
        if let Err(evaluation_error) = runtime.evaluate_arrays(&[&step_allocation]) {
            tracing::warn!(
                wired_bytes,
                error = %evaluation_error,
                "GPU wired-capacity verification hit an evaluation failure"
            );
            outcome = WiredCapacityVerification::ClampedToLastHealthyStep { wired_bytes };
            break;
        }
        let step_elapsed = step_started_at.elapsed();
        if verification_step_verdict(step_elapsed, VERIFICATION_STEP_LATENCY_BUDGET)
            == VerificationStepVerdict::BoundaryHit
        {
            tracing::warn!(
                wired_bytes,
                step_latency_millis = step_elapsed.as_millis(),
                step_latency_budget_millis = VERIFICATION_STEP_LATENCY_BUDGET.as_millis(),
                "GPU wired-capacity verification hit the operating-system wired boundary"
            );
            outcome = WiredCapacityVerification::ClampedToLastHealthyStep { wired_bytes };
            if wired_bytes == 0 {
                floor_reached = true;
                break;
            }
            if boundary_hit {
                // A second slow probe during refinement means the boundary is
                // confirmed between two adjacent step-granular points.
                break;
            }
            boundary_hit = true;
            continue;
        }
        slowest_healthy_step = slowest_healthy_step.max(step_elapsed);
        held_arrays.push(step_allocation);
        wired_bytes += probe_bytes;
        previous_probe_bytes = probe_bytes;
        outcome = WiredCapacityVerification::Verified { wired_bytes };
    }
    drop(held_arrays);
    if floor_reached {
        return Err(
            crate::worker_startup_error::WorkerStartupError::GpuWiredCapacityVerificationFloor,
        );
    }
    if let Err(cleanup_error) = runtime.synchronize_gpu_stream_and_clear_allocator_cache() {
        tracing::warn!(
            error = %cleanup_error,
            "GPU wired-capacity verification cleanup failed"
        );
    }
    tracing::info!(
        candidate_ceiling_bytes,
        verified_wired_bytes = match outcome {
            WiredCapacityVerification::Verified { wired_bytes }
            | WiredCapacityVerification::ClampedToLastHealthyStep { wired_bytes } => wired_bytes,
        },
        slowest_healthy_step_millis = slowest_healthy_step.as_millis(),
        elapsed_millis = verification_started_at.elapsed().as_millis(),
        "verified the effective graphics-processor memory ceiling by wiring"
    );
    Ok(outcome)
}
