//! Wiring contracts between the build script and its progress and parallelism
//! modules, asserted on the build script source the same way Cargo sees it.

use std::path::Path;

const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
const NATIVE_BUILD_COMPILE_MODULE: &str =
    include_str!("../../build-orchestration/native_compile.rs");
const NATIVE_BUILD_PARALLELISM_MODULE: &str =
    include_str!("../../build-orchestration/native_parallelism.rs");

/// Collects the literal file and directory paths the build script watches
/// through rerun-if-changed inside emit_native_source_rerun_contracts.
/// The dynamic dependency cache directory is environment-dependent and
/// intentionally excluded.
fn watched_paths_from_rerun_contracts() -> Vec<String> {
    let contracts_section_start = NATIVE_BUILD_SCRIPT
        .find("fn emit_native_source_rerun_contracts")
        .expect("the build script must keep its rerun contract emitter");
    let contracts_section = &NATIVE_BUILD_SCRIPT[contracts_section_start..];
    let mut watched_paths = Vec::new();
    let mut cursor = 0;
    while let Some(join_offset) = contracts_section[cursor..].find(".join(\"") {
        let entry_start = cursor + join_offset + ".join(\"".len();
        let entry_end = contracts_section[entry_start..]
            .find('"')
            .expect("unterminated join literal in the rerun contracts")
            + entry_start;
        let joined_path = &contracts_section[entry_start..entry_end];
        // native_source_directory joins are rooted under native/, every other
        // join in this function is rooted at the package directory, which is
        // also the working directory of this test binary.
        let receiver_start = contracts_section[..cursor + join_offset]
            .rfind(|character: char| character == '.' || character.is_whitespace())
            .expect("join literal without a receiver")
            + 1;
        let receiver = &contracts_section[receiver_start..cursor + join_offset];
        if receiver == "native_source_directory" {
            watched_paths.push(format!("native/{joined_path}"));
        } else {
            watched_paths.push(joined_path.to_owned());
        }
        cursor = entry_end;
    }
    watched_paths
}

#[test]
fn should_watch_only_paths_that_exist() {
    // A rerun-if-changed path that does not exist dirties the build script at
    // every cargo freshness check: cargo reports the missing file, reruns the
    // script, and rebuilds runtime-integration and its dependents in every
    // later cargo invocation of the CI job — a measured 25s rebuild per step
    // while a phantom build_bindings.rs watch was live.
    let watched_paths = watched_paths_from_rerun_contracts();
    assert!(
        !watched_paths.is_empty(),
        "the rerun contract watch list disappeared; the extraction or the contracts rotted"
    );
    for watched_path in &watched_paths {
        assert!(
            Path::new(watched_path).exists(),
            "the build script watches {watched_path}, which does not exist; \
             a phantom rerun-if-changed path dirties the build script on every cargo check"
        );
    }
}

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
