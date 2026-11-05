use std::ffi::OsStr;
use std::path::PathBuf;

use astronomical_ipc_protocol::{
    WorkerDecodeExpertCachePolicy, WorkerDiagnosticQwen3_5EngineOverride, WorkerLogLevel,
    WorkerStartupConfiguration,
};
use tokio::process::Command;

use super::{MLX_RESIDENCY_REFRESH_VARIABLE, apply_metal_residency_refresh_policy};

fn startup_configuration(is_refresh_enabled: bool) -> WorkerStartupConfiguration {
    WorkerStartupConfiguration {
        neural_engine_program_directory: "neural-engine-programs".into(),
        configuration_generation: "generation".to_owned(),
        global_prompt_cache_root_directory: PathBuf::from("prompt-cache"),
        global_prompt_cache_maximum_size_bytes: 1,
        persistent_prompt_cache_enabled: false,
        configured_maximum_mlx_memory_bytes: None,
        performance_attribution_enabled: false,
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy::Baseline,
        diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride::Automatic,
        diagnostic_qwen3_5_resident_prefill_activation_floor_bytes: None,
        logging_directory: PathBuf::from("logs"),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
        metal_residency_refresh_enabled: is_refresh_enabled,
    }
}

/// What the command will do to the variable: `None` when it has no instruction,
/// `Some(None)` when it removes it, and `Some(Some(value))` when it sets it.
fn refresh_variable_instruction(worker_command: &Command) -> Option<Option<String>> {
    for (variable_name, variable_value) in worker_command.as_std().get_envs() {
        if variable_name == OsStr::new(MLX_RESIDENCY_REFRESH_VARIABLE) {
            return match variable_value {
                Some(variable_value) => Some(Some(variable_value.to_string_lossy().into_owned())),
                None => Some(None),
            };
        }
    }
    None
}

#[test]
fn should_set_the_mlx_interval_when_the_setting_is_on() {
    let mut worker_command = Command::new("worker");

    apply_metal_residency_refresh_policy(&mut worker_command, Some(&startup_configuration(true)));

    assert_eq!(
        refresh_variable_instruction(&worker_command),
        Some(Some("1000".to_owned()))
    );
}

#[test]
fn should_remove_any_inherited_variable_when_the_setting_is_off() {
    let mut worker_command = Command::new("worker");
    worker_command.env(MLX_RESIDENCY_REFRESH_VARIABLE, "5");

    apply_metal_residency_refresh_policy(&mut worker_command, Some(&startup_configuration(false)));

    assert_eq!(refresh_variable_instruction(&worker_command), Some(None));
}

#[test]
fn should_leave_the_environment_alone_without_a_startup_configuration() {
    let mut worker_command = Command::new("worker");

    apply_metal_residency_refresh_policy(&mut worker_command, None);

    assert_eq!(refresh_variable_instruction(&worker_command), None);
}
