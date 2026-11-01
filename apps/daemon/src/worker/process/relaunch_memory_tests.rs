//! Unit tests for the launch snapshot a worker replacement boots from.

use std::path::PathBuf;

use astronomical_ipc_protocol::{
    WorkerDecodeExpertCachePolicy, WorkerDiagnosticQwen3_5EngineOverride, WorkerLogLevel,
    WorkerStartupConfiguration,
};

use super::startup_configuration_carrying_applied_memory_ceiling;

/// Apple gigabyte. The menu stepper and the persisted ceiling both use 2^30.
const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;
/// The 21 GB ceiling the incident's worker processes logged at startup.
const BOOT_CONFIGURED_CEILING_BYTES: u64 = 21 * BYTES_PER_GIGABYTE;
/// The 37 GB ceiling the user asked for. Startup clamps this again against
/// the wired budget; the launch snapshot must keep the request itself.
const RAISED_CONFIGURED_CEILING_BYTES: u64 = 37 * BYTES_PER_GIGABYTE;
const LOWERED_CONFIGURED_CEILING_BYTES: u64 = 16 * BYTES_PER_GIGABYTE;
const BOOT_CONFIGURATION_GENERATION: &str = "boot-generation-21gb";
const RAISED_CONFIGURATION_GENERATION: &str = "raised-generation-37gb";
const LOWERED_CONFIGURATION_GENERATION: &str = "lowered-generation-16gb";
const REFRESHED_CONFIGURATION_GENERATION: &str = "same-bytes-new-generation";

fn boot_startup_configuration() -> WorkerStartupConfiguration {
    WorkerStartupConfiguration {
        configuration_generation: BOOT_CONFIGURATION_GENERATION.to_owned(),
        global_prompt_cache_root_directory: PathBuf::from("prompt-cache"),
        global_prompt_cache_maximum_size_bytes: 50_000_000_000,
        persistent_prompt_cache_enabled: true,
        configured_maximum_mlx_memory_bytes: Some(BOOT_CONFIGURED_CEILING_BYTES),
        performance_attribution_enabled: true,
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy::Candidate,
        diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride::Streaming,
        diagnostic_qwen3_5_resident_prefill_activation_floor_bytes: Some(1_610_612_736),
        logging_directory: PathBuf::from("logs"),
        logging_level: WorkerLogLevel::Debug,
        retained_log_file_count: 4,
    }
}

#[test]
fn should_keep_a_missing_launch_snapshot_missing() {
    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        None,
        RAISED_CONFIGURED_CEILING_BYTES,
        RAISED_CONFIGURATION_GENERATION,
    );

    assert!(adopted_startup_configuration.is_none());
}

#[test]
fn should_leave_the_boot_ceiling_unchanged_when_the_requested_ceiling_is_zero() {
    let boot_startup_configuration = boot_startup_configuration();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration.clone()),
        0,
        RAISED_CONFIGURATION_GENERATION,
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(boot_startup_configuration)
    );
}

#[test]
fn should_leave_the_boot_ceiling_unchanged_when_the_configuration_generation_is_empty() {
    let boot_startup_configuration = boot_startup_configuration();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration.clone()),
        RAISED_CONFIGURED_CEILING_BYTES,
        "",
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(boot_startup_configuration)
    );
}

#[test]
fn should_leave_the_boot_ceiling_unchanged_when_the_configuration_generation_is_whitespace() {
    let boot_startup_configuration = boot_startup_configuration();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration.clone()),
        RAISED_CONFIGURED_CEILING_BYTES,
        " \t\n",
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(boot_startup_configuration)
    );
}

#[test]
fn should_carry_a_lowered_configured_ceiling_and_preserve_the_other_launch_fields() {
    let mut boot_startup_configuration = boot_startup_configuration();
    boot_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(RAISED_CONFIGURED_CEILING_BYTES);
    boot_startup_configuration.configuration_generation =
        RAISED_CONFIGURATION_GENERATION.to_owned();
    let mut expected_startup_configuration = boot_startup_configuration.clone();
    expected_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(LOWERED_CONFIGURED_CEILING_BYTES);
    expected_startup_configuration.configuration_generation =
        LOWERED_CONFIGURATION_GENERATION.to_owned();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration),
        LOWERED_CONFIGURED_CEILING_BYTES,
        LOWERED_CONFIGURATION_GENERATION,
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(expected_startup_configuration)
    );
}

#[test]
fn should_refresh_the_configuration_generation_when_the_applied_ceiling_bytes_are_unchanged() {
    let mut boot_startup_configuration = boot_startup_configuration();
    boot_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(RAISED_CONFIGURED_CEILING_BYTES);
    let mut expected_startup_configuration = boot_startup_configuration.clone();
    expected_startup_configuration.configuration_generation =
        REFRESHED_CONFIGURATION_GENERATION.to_owned();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration),
        RAISED_CONFIGURED_CEILING_BYTES,
        REFRESHED_CONFIGURATION_GENERATION,
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(expected_startup_configuration)
    );
}

#[test]
fn should_install_a_configured_ceiling_when_the_launch_snapshot_had_none() {
    let mut boot_startup_configuration = boot_startup_configuration();
    boot_startup_configuration.configured_maximum_mlx_memory_bytes = None;
    let mut expected_startup_configuration = boot_startup_configuration.clone();
    expected_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(RAISED_CONFIGURED_CEILING_BYTES);
    expected_startup_configuration.configuration_generation =
        RAISED_CONFIGURATION_GENERATION.to_owned();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration),
        RAISED_CONFIGURED_CEILING_BYTES,
        RAISED_CONFIGURATION_GENERATION,
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(expected_startup_configuration)
    );
}

#[test]
fn should_carry_the_raised_configured_ceiling_into_the_relaunch_snapshot() {
    // The worker log for the disconnected session recorded this exact boot ceiling.
    assert_eq!(BOOT_CONFIGURED_CEILING_BYTES, 22_548_578_304);

    let boot_startup_configuration = boot_startup_configuration();
    let mut expected_startup_configuration = boot_startup_configuration.clone();
    expected_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(RAISED_CONFIGURED_CEILING_BYTES);
    expected_startup_configuration.configuration_generation =
        RAISED_CONFIGURATION_GENERATION.to_owned();

    let adopted_startup_configuration = startup_configuration_carrying_applied_memory_ceiling(
        Some(boot_startup_configuration),
        RAISED_CONFIGURED_CEILING_BYTES,
        RAISED_CONFIGURATION_GENERATION,
    );

    assert_eq!(
        adopted_startup_configuration,
        Some(expected_startup_configuration)
    );
}
