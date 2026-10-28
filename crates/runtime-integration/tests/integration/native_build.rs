//! Structural and filesystem contracts for native configuration that Cargo cannot infer.

use std::{fs, path::Path, process::Command};

use sha2::{Digest, Sha256};

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_COMPILATION: &str =
    include_str!("../../../native-build-orchestration/src/native_compile.rs");
const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
const NATIVE_TILE_SELECTION_PATCH: &str =
    include_str!("../../../../third-party/patches/mlx-0.32.4-nax-qmm-tile-variants.patch");
const ARCHIVE_VARIABLES: [&str; 5] = [
    "ASTRONOMICAL_MLX_SOURCE_ARCHIVE",
    "ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE",
    "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE",
    "ASTRONOMICAL_JSON_SOURCE_ARCHIVE",
    "ASTRONOMICAL_FMT_SOURCE_ARCHIVE",
];
const PINNED_ARCHIVE_CONTENTS: &[u8] = b"verified source archive";

// The bindgen-allowlist completeness contract that lived here moved with the
// bridge into the published mlx-c-rust crate; the live coverage contract
// (mlx_c_coverage_contract_tests, mlx feature) is the authoritative proof —
// any narrowing of the published surface fails it as unbridged symbols.

#[test]
fn should_enable_runtime_metal_kernel_selection_for_the_current_apple_gpu() {
    assert!(
        NATIVE_BUILD_CONFIGURATION.contains("set(MLX_METAL_JIT ON"),
        "the native runtime must let MLX select and cache NAX kernels on capable Apple GPUs"
    );
}

#[test]
fn should_control_the_compiler_and_sdk_that_define_the_native_toolchain() {
    for required_configuration in [
        "-DCMAKE_C_COMPILER=",
        "-DCMAKE_CXX_COMPILER=",
        "-DCMAKE_OSX_SYSROOT=",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "remove_uncontrolled_native_environment",
    ] {
        assert!(
            NATIVE_BUILD_COMPILATION.contains(required_configuration),
            "the native build must control toolchain input {required_configuration}"
        );
    }
}

#[test]
fn should_resolve_archives_without_per_archive_rust_environment_inputs() {
    for archive_variable in ARCHIVE_VARIABLES {
        assert!(!NATIVE_BUILD_COMPILATION.contains(archive_variable));
        assert!(!NATIVE_BUILD_SCRIPT.contains(archive_variable));
        assert!(NATIVE_BUILD_CONFIGURATION.contains(archive_variable));
    }
    for verification_step in ["file(SHA256", "URL_HASH", "require_pinned_archive("] {
        assert!(NATIVE_BUILD_CONFIGURATION.contains(verification_step));
    }
}

#[test]
fn should_resolve_pinned_cache_archives_despite_stale_definitions_and_environment_entries() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let dependency_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&dependency_cache_directory).unwrap();
    fs::write(
        dependency_cache_directory.join("pinned.tar.gz"),
        PINNED_ARCHIVE_CONTENTS,
    )
    .unwrap();

    let archive_check = check_pinned_archives(test_directory.path(), &dependency_cache_directory);

    assert!(
        archive_check.status.success(),
        "{}",
        String::from_utf8_lossy(&archive_check.stderr)
    );
}

#[test]
fn should_reject_a_missing_cache_archive_even_when_an_override_points_to_verified_contents() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let dependency_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&dependency_cache_directory).unwrap();

    let archive_check = check_pinned_archives(test_directory.path(), &dependency_cache_directory);

    assert!(!archive_check.status.success());
    assert!(
        String::from_utf8_lossy(&archive_check.stderr).contains("Missing test dependency archive")
    );
}

#[test]
fn should_reject_a_cache_archive_with_the_wrong_digest() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let dependency_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&dependency_cache_directory).unwrap();
    fs::write(
        dependency_cache_directory.join("pinned.tar.gz"),
        b"corrupted source archive",
    )
    .unwrap();

    let archive_check = check_pinned_archives(test_directory.path(), &dependency_cache_directory);

    assert!(!archive_check.status.success());
    let archive_error = String::from_utf8_lossy(&archive_check.stderr)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        archive_error.contains("SHA-256") && archive_error.contains("does not match"),
        "{archive_error}"
    );
}

#[test]
fn should_reject_a_relative_dependency_cache_directory() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");

    let archive_check = check_pinned_archives(test_directory.path(), Path::new("relative-cache"));

    assert!(!archive_check.status.success());
    assert!(String::from_utf8_lossy(&archive_check.stderr).contains("must be an absolute path"));
}

#[test]
fn should_keep_production_tile_selection_shape_driven_without_environment_overrides() {
    for required_selection in [
        "select_qmm_t_nax_tile(int token_count)",
        "if (token_count <= 32)",
        "return {32, 64, 64, 2, 2};",
        "return {64, 64, 64, 4, 1};",
        "select_qmm_t_nax_tile(M)",
    ] {
        assert!(NATIVE_TILE_SELECTION_PATCH.contains(required_selection));
    }
    assert!(!NATIVE_TILE_SELECTION_PATCH.contains("getenv"));
    assert!(!NATIVE_TILE_SELECTION_PATCH.contains("override"));
}

fn check_pinned_archives(
    test_directory: &Path,
    dependency_cache_directory: &Path,
) -> std::process::Output {
    let archive_function_start = NATIVE_BUILD_CONFIGURATION
        .find("function(require_pinned_archive ")
        .expect("the native configuration should own archive verification");
    let archive_function = NATIVE_BUILD_CONFIGURATION[archive_function_start..]
        .split_once("endfunction()")
        .expect("the archive verification function should be complete")
        .0;
    let archive_digest = Sha256::digest(PINNED_ARCHIVE_CONTENTS)
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect::<String>();
    let mut check_script = format!(
        "set(ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR \"{}\")\n{archive_function}endfunction()\n",
        dependency_cache_directory.display()
    );
    let ignored_archive_path = test_directory.join("ignored-archive.tar.gz");
    fs::write(&ignored_archive_path, PINNED_ARCHIVE_CONTENTS).unwrap();
    let mut archive_check_command = Command::new("cmake");
    for archive_variable in ARCHIVE_VARIABLES {
        check_script.push_str(&format!(
            "require_pinned_archive({archive_variable} \"test dependency\" \"pinned.tar.gz\" \"{archive_digest}\")\n\
             if(NOT \"${{{archive_variable}}}\" STREQUAL \"${{ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}}/pinned.tar.gz\")\n\
             message(FATAL_ERROR \"archive escaped the pinned dependency cache\")\n\
             endif()\n"
        ));
        archive_check_command
            .arg(format!(
                "-D{archive_variable}={}",
                ignored_archive_path.display()
            ))
            .env(archive_variable, &ignored_archive_path);
    }
    let check_script_path = test_directory.join("check-pinned-archives.cmake");
    fs::write(&check_script_path, check_script).unwrap();
    archive_check_command
        .arg("-P")
        .arg(check_script_path)
        .output()
        .expect("CMake should run the archive verification without configuring a native build")
}
