//! Wiring contracts between the build script and its progress and parallelism
//! modules, asserted on the build script source the same way Cargo sees it.

const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");

#[test]
fn should_stream_native_build_lifecycle_through_the_progress_recorder() {
    for required_wiring in [
        "NativeBuildProgress::from_environment()",
        "run_operation(\"resolve-native-runtime\"",
        "fn run_command(",
        "native_build_progress: &NativeBuildProgress",
        "native_build_progress.run_operation(operation",
        "status=parallelism jobs=",
    ] {
        assert!(
            NATIVE_BUILD_SCRIPT.contains(required_wiring),
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
        !NATIVE_BUILD_SCRIPT.contains("unwrap_or_else(|_| \"1\".to_owned())"),
        "a silent serial fallback must not return"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains("resolve_native_build_jobs_from_environment()"),
        "the native build must resolve its job count explicitly"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains(".arg(\"--parallel\")"),
        "the native build must pass its resolved job count to CMake"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains("resolved_native_build_jobs.job_count"),
        "the resolved job count must feed the CMake parallel flag"
    );
}
