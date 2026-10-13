use astronomical_inference_worker::worker_startup::{
    GpuWiredMemoryLimitSetting, WorkerProcessError, WorkerStartupError,
    derive_serving_mlx_memory_ceiling_bytes, parse_iogpu_wired_limit_setting,
};
use astronomical_model_serving::{
    effective_mlx_serving_memory_ceiling_bytes, recommended_gpu_serving_memory_ceiling_bytes,
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
fn should_reject_working_sets_that_cannot_produce_a_positive_serving_ceiling() {
    for recommended_gpu_working_set_bytes in [0, 1_000_000_000, 2 * 1024 * 1024 * 1024] {
        let startup_error =
            derive_serving_mlx_memory_ceiling_bytes(recommended_gpu_working_set_bytes)
                .expect_err("a zero serving ceiling must fail worker startup");

        assert!(matches!(
            &startup_error,
            WorkerStartupError::RecommendedGpuWorkingSetTooSmall
        ));
        assert!(
            startup_error
                .to_string()
                .contains("cannot produce a positive serving memory ceiling")
        );
    }
}

#[test]
fn should_accept_a_working_set_that_produces_a_positive_serving_ceiling() {
    assert_eq!(
        derive_serving_mlx_memory_ceiling_bytes(2 * 1024 * 1024 * 1024 + 1)
            .expect("a positive serving ceiling should be derived"),
        1
    );
}

#[test]
fn should_use_the_serving_ceiling_when_config_override_is_absent() {
    assert_eq!(
        effective_mlx_serving_memory_ceiling_bytes(
            None,
            40_000_000_000,
            recommended_gpu_serving_memory_ceiling_bytes(40_000_000_000),
        ),
        35_000_000_000
    );
}

#[test]
fn should_use_lower_configured_mlx_ceiling() {
    assert_eq!(
        effective_mlx_serving_memory_ceiling_bytes(
            Some(32_000_000_000),
            40_000_000_000,
            recommended_gpu_serving_memory_ceiling_bytes(40_000_000_000),
        ),
        32_000_000_000
    );
}

#[test]
fn should_clamp_a_higher_configured_value_to_the_serving_ceiling() {
    assert_eq!(
        effective_mlx_serving_memory_ceiling_bytes(
            Some(48_000_000_000),
            40_000_000_000,
            recommended_gpu_serving_memory_ceiling_bytes(40_000_000_000),
        ),
        35_000_000_000
    );
}
