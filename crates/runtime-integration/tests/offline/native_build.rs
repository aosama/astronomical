//! Structural and filesystem contracts for native configuration that Cargo cannot infer.

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_COMPILATION: &str = include_str!("../../build-orchestration/native_compile.rs");

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
