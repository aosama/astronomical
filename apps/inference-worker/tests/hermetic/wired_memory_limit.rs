use astronomical_inference_worker::worker_startup::{
    GpuWiredMemoryLimitSetting, derive_mlx_memory_limits_from_gpu_wired_limit,
    parse_iogpu_wired_limit_setting, resolve_effective_mlx_memory_ceiling_bytes,
    sample_iogpu_wired_limit_bytes,
};

#[test]
fn should_parse_a_gpu_wired_limit_as_bytes() {
    let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("1024\n")
        .expect("the synthetic GPU wired-memory limit should parse");

    assert_eq!(
        wired_memory_limit_setting,
        GpuWiredMemoryLimitSetting::ExplicitLimitBytes(1024 * 1024 * 1024)
    );
}

#[tokio::test]
async fn should_bound_mlx_allocator_reuse_by_the_operating_system_gpu_wired_limit() {
    let operating_system_gpu_wired_limit_bytes = sample_iogpu_wired_limit_bytes()
        .await
        .expect("the operating system GPU wired-memory limit should be available");

    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        derive_mlx_memory_limits_from_gpu_wired_limit(operating_system_gpu_wired_limit_bytes);

    assert_eq!(
        active_memory_limit_bytes,
        operating_system_gpu_wired_limit_bytes
    );
    assert_eq!(
        allocator_cache_memory_limit_bytes,
        operating_system_gpu_wired_limit_bytes
    );
}

#[test]
fn should_treat_zero_iogpu_wired_limit_as_system_default_policy() {
    let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("0\n")
        .expect("zero should mean the system default GPU wired-memory policy");

    assert_eq!(
        wired_memory_limit_setting,
        GpuWiredMemoryLimitSetting::SystemDefaultPolicy
    );
}

#[test]
fn should_use_machine_mlx_ceiling_when_config_override_is_absent() {
    assert_eq!(
        resolve_effective_mlx_memory_ceiling_bytes(None, 40_000_000_000),
        40_000_000_000
    );
}

#[test]
fn should_use_lower_configured_mlx_ceiling() {
    assert_eq!(
        resolve_effective_mlx_memory_ceiling_bytes(Some(32_000_000_000), 40_000_000_000),
        32_000_000_000
    );
}

#[test]
fn should_clamp_a_higher_configured_value_to_the_machine_ceiling() {
    assert_eq!(
        resolve_effective_mlx_memory_ceiling_bytes(Some(48_000_000_000), 40_000_000_000),
        40_000_000_000
    );
}
