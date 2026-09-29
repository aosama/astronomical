//! Resolution of the native CMake build's parallel job count.
//!
//! The multi-minute MLX C++ compile dominates cold native builds, so it must
//! never silently run serial. Cargo hands build scripts its job count through
//! `NUM_JOBS`, hosted CI exports `CARGO_BUILD_JOBS`, and the machine's
//! available parallelism covers any context where neither variable is present.

/// Resolves the parallel job count passed to `cmake --parallel`, preferring
/// the explicit `CARGO_BUILD_JOBS` override, then cargo's own `NUM_JOBS`,
/// then the machine's available parallelism.
pub fn resolve_native_parallel_job_count(
    cargo_build_jobs_override: Option<&str>,
    cargo_num_jobs: Option<&str>,
    machine_parallelism: usize,
) -> String {
    cargo_build_jobs_override
        .or(cargo_num_jobs)
        .map(str::to_owned)
        .unwrap_or_else(|| machine_parallelism.to_string())
}
