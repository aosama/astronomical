//! Wiring contracts between the build script and its progress and parallelism
//! modules, asserted on the build script source the same way Cargo sees it.

const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
const NATIVE_BUILD_COMPILE_MODULE: &str = include_str!("../../build_native_compile.rs");
const NATIVE_BUILD_PARALLELISM_MODULE: &str = include_str!("../../build_parallelism.rs");

#[test]
fn should_stream_native_build_lifecycle_through_the_progress_recorder() {
    for required_wiring in [
        // build.rs owns the recorder lifecycle: it resolves the recorder from
        // the environment and opens and closes the overall build stream.
        "NativeBuildProgress::from_environment()",
        "record_build_start(",
        "record_build_completion(",
        // The compile module owns per-operation progress streaming, and
        // build.rs hands the recorder to it for every native operation.
        "fn run_command(",
        "native_build_progress: &NativeBuildProgress",
        "native_build_progress.begin_operation(operation)",
        "operation_progress.complete(",
        "&native_build_progress",
    ] {
        assert!(
            NATIVE_BUILD_SCRIPT.contains(required_wiring)
                || NATIVE_BUILD_COMPILE_MODULE.contains(required_wiring),
            "the native build must stream {required_wiring}"
        );
    }
}

#[test]
fn should_never_silently_fall_back_to_a_serial_native_build() {
    assert!(
        !NATIVE_BUILD_SCRIPT.contains("cargo_build_job_count"),
        "the serial fallback helper must stay removed"
    );
    assert!(
        !NATIVE_BUILD_COMPILE_MODULE.contains("unwrap_or_else(|_| \"1\".to_owned())"),
        "a silent serial fallback must not return"
    );
    assert!(
        NATIVE_BUILD_PARALLELISM_MODULE.contains("resolve_native_parallel_job_count("),
        "the native build must resolve its parallel job count explicitly"
    );
    assert!(
        NATIVE_BUILD_COMPILE_MODULE.contains(".arg(\"--parallel\")"),
        "the resolved job count must reach the native CMake build"
    );
    assert!(
        NATIVE_BUILD_COMPILE_MODULE.contains(".arg(native_parallel_job_count())"),
        "the resolved job count must feed the native CMake build"
    );
}

#[test]
fn should_resolve_native_parallelism_decoupled_from_cargo() {
    // Cargo compiles Rust crates concurrently while the build script runs, so
    // the native CMake build must not inherit cargo's job count: a
    // native-specific override wins, then the machine's available parallelism.
    for required_wiring in [
        "NATIVE_BUILD_JOBS_VARIABLE",
        "resolve_native_parallel_job_count(",
        "available_parallelism()",
    ] {
        assert!(
            NATIVE_BUILD_PARALLELISM_MODULE.contains(required_wiring)
                || NATIVE_BUILD_COMPILE_MODULE.contains(required_wiring),
            "the native build must resolve its parallelism decoupled from cargo through {required_wiring}"
        );
    }
}
