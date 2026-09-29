//! Resolution of the native CMake build's parallel job count.
//!
//! The multi-minute MLX C++ compile dominates cold native builds, so it must
//! never silently run serial. Cargo compiles Rust crates concurrently while
//! the build script runs, so the native build's parallelism is decoupled from
//! cargo's own job count: a native-specific override wins, then cargo's job
//! count as the historical default, then the machine's available parallelism.

// Referenced by build.rs, which is not part of the hermetic test target that
// includes this module through #[path]; the warning would otherwise fire there.
#[allow(dead_code)]
pub const NATIVE_BUILD_JOBS_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_JOBS";

/// Resolves the parallel job count passed to `cmake --parallel`, preferring
/// the native-specific override, then the explicit `CARGO_BUILD_JOBS`
/// override, then cargo's own `NUM_JOBS`, then the machine's available
/// parallelism.
pub fn resolve_native_parallel_job_count(
    native_build_jobs_override: Option<&str>,
    cargo_build_jobs_override: Option<&str>,
    cargo_num_jobs: Option<&str>,
    machine_parallelism: usize,
) -> String {
    native_build_jobs_override
        .or(cargo_build_jobs_override)
        .or(cargo_num_jobs)
        .map(str::to_owned)
        .unwrap_or_else(|| machine_parallelism.to_string())
}
