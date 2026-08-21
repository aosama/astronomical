//! Structural and filesystem contracts for native configuration that Cargo cannot infer.

use std::fs;

#[path = "../../build_legacy_native_output.rs"]
mod build_legacy_native_output;

use build_legacy_native_output::remove_legacy_cargo_native_build_directory;

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
const NATIVE_BUILD_STORE: &str = include_str!("../../build_native_store.rs");
const BINDGEN_CONFIGURATION: &str = include_str!("../../build_bindings.rs");

#[test]
fn should_enable_runtime_metal_kernel_selection_for_the_current_apple_gpu() {
    assert!(
        NATIVE_BUILD_CONFIGURATION.contains("set(MLX_METAL_JIT ON"),
        "the native runtime must let MLX select and cache NAX kernels on capable Apple GPUs"
    );
}

#[test]
fn should_not_build_the_retired_native_expert_page_store() {
    assert!(!NATIVE_BUILD_CONFIGURATION.contains("astronomical_native_expert_page_store"));
    assert!(!NATIVE_BUILD_SCRIPT.contains("astronomical_native_expert_page_store"));
}

#[test]
fn should_allowlist_the_pinned_mlx_c_flux_primitives() {
    for required_binding in [
        "data_(float32|uint8|uint32)",
        "conv(1d|2d|3d)",
        "|clip|",
        "|full|",
        "|pad|",
        "|sqrt|",
        "random_(categorical|key|normal|split)",
    ] {
        assert!(
            BINDGEN_CONFIGURATION.contains(required_binding),
            "the narrow bindgen surface must include {required_binding}"
        );
    }
}

#[test]
fn should_keep_reusable_native_products_outside_the_cargo_output_directory() {
    assert!(
        NATIVE_BUILD_SCRIPT.contains("ASTRONOMICAL_NATIVE_BUILD_STORE_DIR"),
        "native products need a stable store that does not follow Cargo package-version output paths"
    );
    assert!(
        !NATIVE_BUILD_SCRIPT.contains("output_directory.join(\"mlx-c-runtime-build\")"),
        "the complete CMake tree must not remain under package-version-sensitive OUT_DIR"
    );
    assert!(
        BINDGEN_CONFIGURATION.contains("output_directory.join(\"mlx_c_bindings.rs\")"),
        "generated Rust bindings remain Cargo-owned because rustc includes them from OUT_DIR"
    );
    assert!(
        NATIVE_BUILD_SCRIPT.contains("remove_legacy_cargo_native_build_directory"),
        "the current Cargo build unit must remove the retired complete native tree from OUT_DIR"
    );
    assert!(
        NATIVE_BUILD_STORE.contains("entries"),
        "the reusable store must separate complete compatibility-keyed entries"
    );
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

    remove_legacy_cargo_native_build_directory(cargo_output.path())
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

    let cleanup_error = remove_legacy_cargo_native_build_directory(cargo_output.path())
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
            NATIVE_BUILD_SCRIPT.contains(required_configuration),
            "the native build must control compatibility input {required_configuration}"
        );
    }
}
