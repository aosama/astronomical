//! Structural contracts for native configuration that Cargo cannot infer.

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");

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
