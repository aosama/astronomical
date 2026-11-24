//! Resolution of the native CMake build's parallel job count.
//!
//! # Why native parallelism is decoupled from Cargo's
//!
//! The multi-minute MLX C++ compile dominates cold native builds, so it must
//! never silently run serial. Cargo compiles Rust crates concurrently while
//! the build script runs, so the native build's parallelism is decoupled from
//! cargo's own job count: a native-specific override wins, then cargo's job
//! count as the historical default, then the machine's available parallelism.
//! (Inheriting cargo's count directly would starve either the native build
//! or the concurrent Rust compile on small machines.)

// Part of the crate's public surface: the runtime-integration build-script
// wiring contract pins this variable name by reading the module's source.
pub const NATIVE_BUILD_JOBS_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_JOBS";

#[cfg(test)]
mod tests;

/// Resolves the parallel job count passed to `cmake --parallel`, preferring
/// the native-specific override, then the explicit `CARGO_BUILD_JOBS`
/// override, then cargo's own `NUM_JOBS`, then the machine's available
/// parallelism.
///
/// The precedence is a deliberate compatibility choice: teams that already
/// pin `CARGO_BUILD_JOBS` in CI get the same clamp here without a second
/// variable, while `ASTRONOMICAL_NATIVE_BUILD_JOBS` exists for the case
/// where the two builds need different budgets on one machine.
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
