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
        "parallelism_strategy.progress_line()",
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
        NATIVE_BUILD_SCRIPT.contains("resolve_native_build_parallelism_from_environment()"),
        "the native build must resolve its parallelism strategy explicitly"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains(".arg(\"-j\")"),
        "the fixed job count must reach the native build tool"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains("apply_native_build_parallelism("),
        "the resolved strategy must feed the native build tool"
    );
}

#[test]
fn should_join_the_cargo_jobserver_for_native_compiles() {
    for required_wiring in [
        "Command::new(\"make\")",
        "native_build_tool_command(",
        "env(\"MAKEFLAGS\", makeflags)",
    ] {
        assert!(
            NATIVE_BUILD_SCRIPT.contains(required_wiring),
            "the native build must join cargo's jobserver through {required_wiring}"
        );
    }
}
