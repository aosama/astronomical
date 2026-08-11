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
fn should_track_every_native_expert_cache_source_as_a_cargo_build_input() {
    let native_target_declaration_start = NATIVE_BUILD_CONFIGURATION
        .find("add_library(\n    astronomical_native_expert_cache\n    STATIC")
        .expect("the native build should declare the expert-cache library");
    let native_target_declaration = &NATIVE_BUILD_CONFIGURATION[native_target_declaration_start..];
    let native_target_declaration_end = native_target_declaration
        .find("\n)")
        .expect("the native expert-cache source list should terminate");

    for native_source_path in native_target_declaration[..native_target_declaration_end]
        .lines()
        .map(str::trim)
        .filter(|line| line.ends_with(".cpp"))
    {
        assert!(
            NATIVE_BUILD_SCRIPT.contains(native_source_path),
            "Cargo must rebuild when native expert-cache source {native_source_path} changes"
        );
    }
}
