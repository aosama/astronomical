const MLX_RUNTIME_MEMORY_POLICY_SOURCE: &str =
    include_str!("../../src/mlx_runtime/memory_policy.rs");

#[test]
fn should_leave_mlx_per_buffer_wired_residency_disabled() {
    assert!(
        !MLX_RUNTIME_MEMORY_POLICY_SOURCE.contains("raw::mlx_set_wired_limit"),
        "production memory policy must not enable the MLX residency-set path that can panic IOGPU during allocation-pressure reclamation"
    );
}
