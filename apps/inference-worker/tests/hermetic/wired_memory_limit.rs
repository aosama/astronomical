use astronomical_inference_worker::worker_startup::{
    GpuWiredMemoryLimitSetting, WorkerProcessError, WorkerStartupError,
    parse_iogpu_wired_limit_setting, resolve_effective_mlx_memory_ceiling_bytes,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_parse_a_gpu_wired_limit_as_bytes() {
    let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("1024\n")
        .expect("the synthetic GPU wired-memory limit should parse");

    assert_eq!(
        wired_memory_limit_setting,
        GpuWiredMemoryLimitSetting::ExplicitLimitBytes(1024 * 1024 * 1024)
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
fn should_include_the_exact_mlx_cause_in_a_worker_startup_failure() {
    let worker_process_error = WorkerProcessError::Startup(
        WorkerStartupError::ReadMlxRecommendedGpuWorkingSet(MlxRuntimeError::RuntimeOperation {
            operation: "read MLX GPU recommended working set",
            description: "synthetic device-info key failure".to_owned(),
        }),
    );

    let worker_failure_diagnostic = worker_process_error.to_string();

    assert!(worker_failure_diagnostic.contains("failed to read MLX recommended GPU working set"));
    assert!(worker_failure_diagnostic.contains("read MLX GPU recommended working set"));
    assert!(worker_failure_diagnostic.contains("synthetic device-info key failure"));
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
