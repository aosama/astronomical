//! Structural contracts for native configuration that Cargo cannot infer.

const NATIVE_BUILD_CONFIGURATION: &str = include_str!("../../native/CMakeLists.txt");
const NATIVE_BUILD_SCRIPT: &str = include_str!("../../build.rs");
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
