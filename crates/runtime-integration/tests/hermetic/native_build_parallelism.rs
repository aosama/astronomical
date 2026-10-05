//! Contracts for resolving the parallel job count the native CMake build uses,
//! so the multi-minute MLX compile can never silently fall back to serial.

#[path = "../../build_parallelism.rs"]
mod build_parallelism;

#[test]
fn should_prefer_the_native_specific_override() {
    let resolved_job_count =
        build_parallelism::resolve_native_parallel_job_count(Some("2"), Some("3"), Some("3"), 3);
    assert_eq!(
        resolved_job_count, "2",
        "the native-specific override should win so CI can reserve a core for cargo's concurrent rust compile"
    );
}

#[test]
fn should_prefer_the_explicit_cargo_build_jobs_override() {
    let resolved_job_count =
        build_parallelism::resolve_native_parallel_job_count(None, Some("6"), Some("15"), 15);
    assert_eq!(
        resolved_job_count, "6",
        "an explicit CARGO_BUILD_JOBS override should win over cargo's own job count"
    );
}

#[test]
fn should_fall_back_to_the_cargo_provided_job_count() {
    let resolved_job_count =
        build_parallelism::resolve_native_parallel_job_count(None, None, Some("15"), 15);
    assert_eq!(
        resolved_job_count, "15",
        "cargo's NUM_JOBS should drive the native build when no override is set"
    );
}

#[test]
fn should_default_to_the_machine_parallelism() {
    let resolved_job_count =
        build_parallelism::resolve_native_parallel_job_count(None, None, None, 15);
    assert_eq!(
        resolved_job_count, "15",
        "without cargo-provided counts the machine's parallelism should keep the native build off the serial path"
    );
}
