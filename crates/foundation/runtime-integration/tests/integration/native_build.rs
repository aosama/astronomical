//! Structural and filesystem contracts for native configuration that Cargo cannot infer.

use std::{fs, path::Path, process::Command};

use sha2::{Digest, Sha256};

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_COMPILATION: &str =
    include_str!("../../../native-build-orchestration/src/native_compile.rs");
const NATIVE_HOST_PLATFORM: &str =
    include_str!("../../../native-build-orchestration/src/native_host_platform.rs");
const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
const ARCHIVE_VARIABLES: [&str; 5] = [
    "ASTRONOMICAL_MLX_SOURCE_ARCHIVE",
    "ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE",
    "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE",
    "ASTRONOMICAL_JSON_SOURCE_ARCHIVE",
    "ASTRONOMICAL_FMT_SOURCE_ARCHIVE",
];
const PINNED_ARCHIVE_CONTENTS: &[u8] = b"verified source archive";

// The bindgen-allowlist completeness contract that lived here moved with the
// bridge into the published mlx-bridge crate; the live coverage contract
// (mlx_c_coverage_contract_tests, mlx feature) is the authoritative proof —
// any narrowing of the published surface fails it as unbridged symbols.

#[test]
fn should_enable_runtime_metal_kernel_selection_for_the_current_apple_gpu() {
    let darwin_branch = NATIVE_BUILD_CONFIGURATION
        .split("elseif(CMAKE_SYSTEM_NAME STREQUAL \"Linux\")")
        .next()
        .expect("the configuration should have a macOS branch");
    assert!(
        darwin_branch.contains("set(ASTRONOMICAL_MLX_BUILD_METAL ON)"),
        "macOS must build the Metal backend"
    );
    assert!(
        NATIVE_BUILD_CONFIGURATION.contains("set(MLX_METAL_JIT ${ASTRONOMICAL_MLX_BUILD_METAL}"),
        "the native runtime must let MLX select NAX kernels and keep them in the `metal_kernel_cache` on capable Apple GPUs"
    );
}

#[test]
fn should_build_the_cpu_backend_without_metal_on_linux() {
    let linux_branch = NATIVE_BUILD_CONFIGURATION
        .split("elseif(CMAKE_SYSTEM_NAME STREQUAL \"Linux\")")
        .nth(1)
        .and_then(|after_linux_condition| after_linux_condition.split("else()").next())
        .expect("the configuration should have a Linux branch");
    assert!(linux_branch.contains("set(ASTRONOMICAL_MLX_BUILD_METAL OFF)"));
    assert!(
        NATIVE_HOST_PLATFORM.contains("share/cmake/MLX/MLXTargets.cmake"),
        "a Linux build must publish MLX's installed package, which records the libraries MLX links"
    );
    assert!(
        NATIVE_BUILD_COMPILATION.contains("installs_mlx_package(native_target_platform)"),
        "the compile step must install MLX's package where the platform links from it"
    );
}

#[test]
fn should_control_the_compiler_and_sdk_that_define_the_native_toolchain() {
    for required_configuration in [
        "-DCMAKE_C_COMPILER=",
        "-DCMAKE_CXX_COMPILER=",
        "-DCMAKE_OSX_SYSROOT=",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
    ] {
        assert!(
            NATIVE_HOST_PLATFORM.contains(required_configuration),
            "the native build must control toolchain input {required_configuration}"
        );
    }
    assert!(
        NATIVE_BUILD_COMPILATION.contains("compiler_configure_arguments(native_target_platform)"),
        "the configure step must take its compilers from the host platform recipe"
    );
    assert!(
        NATIVE_BUILD_COMPILATION.contains("remove_uncontrolled_native_environment"),
        "the native build must strip ambient compiler variables"
    );
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
fn should_resolve_pinned_astro_native_build_cache_archives_despite_stale_definitions_and_environment_entries()
 {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let astro_native_build_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&astro_native_build_cache_directory).unwrap();
    fs::write(
        astro_native_build_cache_directory.join("pinned.tar.gz"),
        PINNED_ARCHIVE_CONTENTS,
    )
    .unwrap();

    let archive_check =
        check_pinned_archives(test_directory.path(), &astro_native_build_cache_directory);

    assert!(
        archive_check.status.success(),
        "{}",
        String::from_utf8_lossy(&archive_check.stderr)
    );
}

#[test]
fn should_reject_a_missing_astro_native_build_cache_archive_even_when_an_override_points_to_verified_contents()
 {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let astro_native_build_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&astro_native_build_cache_directory).unwrap();

    let archive_check =
        check_pinned_archives(test_directory.path(), &astro_native_build_cache_directory);

    assert!(!archive_check.status.success());
    assert!(
        String::from_utf8_lossy(&archive_check.stderr).contains("Missing test dependency archive")
    );
}

#[test]
fn should_reject_an_astro_native_build_cache_archive_with_the_wrong_digest() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");
    let astro_native_build_cache_directory = test_directory.path().join("dependency-cache");
    fs::create_dir_all(&astro_native_build_cache_directory).unwrap();
    fs::write(
        astro_native_build_cache_directory.join("pinned.tar.gz"),
        b"corrupted source archive",
    )
    .unwrap();

    let archive_check =
        check_pinned_archives(test_directory.path(), &astro_native_build_cache_directory);

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
fn should_reject_a_relative_astro_native_build_cache_directory() {
    let test_directory = tempfile::tempdir().expect("the test directory should be created");

    let archive_check = check_pinned_archives(
        test_directory.path(),
        Path::new("relative-astro-native-build-cache"),
    );

    assert!(!archive_check.status.success());
    assert!(String::from_utf8_lossy(&archive_check.stderr).contains("must be an absolute path"));
}

fn check_pinned_archives(
    test_directory: &Path,
    astro_native_build_cache_directory: &Path,
) -> std::process::Output {
    let archive_function_start = NATIVE_BUILD_CONFIGURATION
        .find("function(require_pinned_archive ")
        .expect("the native configuration should own archive verification");
    let archive_function = NATIVE_BUILD_CONFIGURATION[archive_function_start..]
        .split_once("endfunction()")
        .expect("the archive verification function should be complete")
        .0;
    let archive_digest =
        lowercase_hex::encode_lowercase_hex(&Sha256::digest(PINNED_ARCHIVE_CONTENTS));
    let mut check_script = format!(
        "set(ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR \"{}\")\n{archive_function}endfunction()\n",
        astro_native_build_cache_directory.display()
    );
    let ignored_archive_path = test_directory.join("ignored-archive.tar.gz");
    fs::write(&ignored_archive_path, PINNED_ARCHIVE_CONTENTS).unwrap();
    let mut archive_check_command = Command::new("cmake");
    for archive_variable in ARCHIVE_VARIABLES {
        check_script.push_str(&format!(
            "require_pinned_archive({archive_variable} \"test dependency\" \"pinned.tar.gz\" \"{archive_digest}\")\n\
             if(NOT \"${{{archive_variable}}}\" STREQUAL \"${{ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}}/pinned.tar.gz\")\n\
             message(FATAL_ERROR \"archive escaped the pinned astro native build cache\")\n\
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
