//! Structural and filesystem contracts for native configuration that Cargo cannot infer.

use std::fs;

#[path = "../../build_legacy_native_output.rs"]
mod build_legacy_native_output;

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_COMPILATION: &str = include_str!("../../build_native_compile.rs");
const BINDGEN_CONFIGURATION: &str = include_str!("../../../mlx-c-rust/build.rs");

#[test]
fn should_enable_runtime_metal_kernel_selection_for_the_current_apple_gpu() {
    assert!(
        NATIVE_BUILD_CONFIGURATION.contains("set(MLX_METAL_JIT ON"),
        "the native runtime must let MLX select and cache NAX kernels on capable Apple GPUs"
    );
}

#[test]
fn should_allowlist_the_complete_mlx_c_surface() {
    // The bridge contract guarantees the complete MLX C surface, not a
    // hand-picked subset: both allowlists must stay the broad mlx_.* pattern
    // so narrowing them back fails here instead of silently un-bridging
    // functions the coverage inventory already declares.
    for required_allowlist in [
        "BINDGEN_FUNCTION_ALLOWLIST: &str = \"mlx_.*\"",
        "BINDGEN_TYPE_ALLOWLIST: &str = \"mlx_.*\"",
    ] {
        assert!(
            BINDGEN_CONFIGURATION.contains(required_allowlist),
            "the bindgen allowlist must stay complete: {required_allowlist}"
        );
    }
}

#[test]
fn should_remove_only_the_retired_native_tree_from_the_current_cargo_output() {
    let cargo_output = tempfile::tempdir().expect("the test should create Cargo output");
    let legacy_native_output = cargo_output.path().join("mlx-c-runtime-build");
    let retained_bindings = cargo_output.path().join("mlx_c_bindings.rs");
    fs::create_dir(&legacy_native_output).expect("the test should create legacy native output");
    fs::write(legacy_native_output.join("CMakeCache.txt"), "retired")
        .expect("the test should create retired native evidence");
    fs::write(&retained_bindings, "bindings")
        .expect("the test should create retained binding evidence");

    build_legacy_native_output::remove_legacy_cargo_native_build_directory(cargo_output.path())
        .expect("the exact retired native directory should be removable");

    assert!(!legacy_native_output.exists());
    assert!(retained_bindings.is_file());
}

#[cfg(unix)]
#[test]
fn should_refuse_a_symbolic_link_at_the_retired_native_output_boundary() {
    use std::os::unix::fs::symlink;

    let cargo_output = tempfile::tempdir().expect("the test should create Cargo output");
    let unowned_directory = tempfile::tempdir().expect("the test should create unowned output");
    let unowned_evidence = unowned_directory.path().join("evidence");
    fs::write(&unowned_evidence, "preserve").expect("the test should create unowned evidence");
    symlink(
        unowned_directory.path(),
        cargo_output.path().join("mlx-c-runtime-build"),
    )
    .expect("the test should create a symbolic-link boundary");

    let cleanup_error =
        build_legacy_native_output::remove_legacy_cargo_native_build_directory(cargo_output.path())
            .expect_err("symbolic-link cleanup must be refused");

    assert!(cleanup_error.to_string().contains("refusing"));
    assert!(unowned_evidence.is_file());
}

#[test]
fn should_control_the_compiler_and_sdk_that_define_native_compatibility() {
    for required_configuration in [
        "-DCMAKE_C_COMPILER=",
        "-DCMAKE_CXX_COMPILER=",
        "-DCMAKE_OSX_SYSROOT=",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "remove_uncontrolled_native_environment",
    ] {
        assert!(
            NATIVE_BUILD_COMPILATION.contains(required_configuration),
            "the native build must control compatibility input {required_configuration}"
        );
    }
}
