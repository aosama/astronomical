//! `astronomical validate config`: loads one instance's configuration file
//! in-process and reports the effective values Astronomical will run with.

use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, AstronomicalInstancePaths,
    AstronomicalRuntimeInstance,
};
use thiserror::Error;

use crate::validate_config_arguments::ValidateConfigArguments;

pub struct ValidateConfigDependencies<'report_writer> {
    pub instance_paths: AstronomicalInstancePaths,
    pub stdout: &'report_writer mut dyn io::Write,
    pub stderr: &'report_writer mut dyn io::Write,
}

#[derive(Debug, Error)]
pub enum ValidateConfigError {
    #[error("No configuration file at {config_file_path}. Start Astronomical once to create one.")]
    ConfigFileMissing { config_file_path: PathBuf },
    #[error("could not read {config_file_path}: {source}")]
    ConfigFileUnreadable {
        config_file_path: PathBuf,
        source: io::Error,
    },
    #[error("invalid configuration at {config_file_path}: {source}")]
    InvalidConfiguration {
        config_file_path: PathBuf,
        source: AstronomicalConfigError,
    },
    #[error("configuration is internally inconsistent: {source}")]
    InconsistentConfiguration { source: AstronomicalConfigError },
    #[error("could not write report: {source}")]
    ReportWriteFailed { source: io::Error },
}

pub fn run_validate_config(
    validate_arguments: &ValidateConfigArguments,
    dependencies: &mut ValidateConfigDependencies<'_>,
) -> Result<(), ValidateConfigError> {
    let config_file_path = dependencies.instance_paths.config_file_path();
    let config_file_bytes = read_config_file_bytes(&config_file_path)?;
    let loaded_config =
        AstronomicalConfig::load_v1_bytes(dependencies.instance_paths.clone(), &config_file_bytes)
            .map_err(|source| ValidateConfigError::InvalidConfiguration {
                config_file_path: config_file_path.clone(),
                source,
            })?;
    let effective_values = effective_values_from_loaded_config(&loaded_config, &config_file_path)?;
    if validate_arguments.render_json {
        render_json_report(&effective_values, dependencies.stdout)
            .map_err(|source| ValidateConfigError::ReportWriteFailed { source })?;
    } else {
        render_text_report(&effective_values, dependencies.stdout)
            .map_err(|source| ValidateConfigError::ReportWriteFailed { source })?;
    }
    Ok(())
}

fn read_config_file_bytes(config_file_path: &Path) -> Result<Vec<u8>, ValidateConfigError> {
    std::fs::read(config_file_path).map_err(|source| match source.kind() {
        io::ErrorKind::NotFound => ValidateConfigError::ConfigFileMissing {
            config_file_path: config_file_path.to_owned(),
        },
        _ => ValidateConfigError::ConfigFileUnreadable {
            config_file_path: config_file_path.to_owned(),
            source,
        },
    })
}

struct EffectiveConfigurationValues {
    configuration_file_path: PathBuf,
    generation: String,
    runtime_instance: AstronomicalRuntimeInstance,
    supervisor_bind_address: SocketAddr,
    model_directories: Vec<PathBuf>,
    configured_model_ids: Vec<String>,
    persistent_prompt_cache_enabled: bool,
    maximum_mlx_memory_bytes: Option<u64>,
    logging_level: String,
    logging_retained_files: usize,
    logging_directory: PathBuf,
    performance_attribution_enabled: bool,
}

fn effective_values_from_loaded_config(
    loaded_config: &AstronomicalConfig,
    config_file_path: &Path,
) -> Result<EffectiveConfigurationValues, ValidateConfigError> {
    let supervisor_bind_address = loaded_config
        .supervisor_bind_address()
        .map_err(|source| ValidateConfigError::InconsistentConfiguration { source })?;
    let maximum_mlx_memory_bytes = loaded_config
        .maximum_mlx_memory_bytes()
        .map_err(|source| ValidateConfigError::InconsistentConfiguration { source })?;
    let logging_values = loaded_config
        .logging()
        .map_err(|source| ValidateConfigError::InconsistentConfiguration { source })?;
    Ok(EffectiveConfigurationValues {
        configuration_file_path: config_file_path.to_owned(),
        generation: loaded_config.generation().to_owned(),
        runtime_instance: loaded_config
            .instance_paths()
            .runtime_instance()
            .unwrap_or(AstronomicalRuntimeInstance::Development),
        supervisor_bind_address,
        model_directories: loaded_config.model_directories().to_vec(),
        configured_model_ids: loaded_config
            .configured_model_ids()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        persistent_prompt_cache_enabled: loaded_config.persistent_prompt_cache_enabled(),
        maximum_mlx_memory_bytes,
        logging_level: logging_values.level().as_str().to_owned(),
        logging_retained_files: logging_values.retained_files(),
        logging_directory: logging_values.directory().to_owned(),
        performance_attribution_enabled: loaded_config.performance_attribution_enabled(),
    })
}

fn render_text_report(
    effective_values: &EffectiveConfigurationValues,
    rendered_output: &mut dyn io::Write,
) -> io::Result<()> {
    let mut report_lines = Vec::new();
    report_lines.push(format!(
        "Configuration file: {}",
        effective_values.configuration_file_path.display()
    ));
    report_lines.push(format!("Generation: {}", effective_values.generation));
    report_lines.push(format!(
        "Runtime instance: {}",
        runtime_instance_slug(effective_values.runtime_instance)
    ));
    report_lines.push(format!(
        "Supervisor bind address: {supervisor_bind_address}",
        supervisor_bind_address = effective_values.supervisor_bind_address
    ));
    report_lines.push(format!(
        "Model directories: {}",
        effective_values.model_directories.len()
    ));
    for model_directory in &effective_values.model_directories {
        report_lines.push(format!("  {}", model_directory.display()));
    }
    report_lines.push(format!(
        "Configured model ids: {}",
        rendered_model_id_list(&effective_values.configured_model_ids)
    ));
    report_lines.push(format!(
        "Persistent prompt cache: {}",
        enabled_word(effective_values.persistent_prompt_cache_enabled)
    ));
    report_lines.push(format!(
        "Maximum MLX memory: {}",
        rendered_maximum_mlx_memory(effective_values.maximum_mlx_memory_bytes)
    ));
    report_lines.push(format!("Logging level: {}", effective_values.logging_level));
    report_lines.push(format!(
        "Logging retained files: {}",
        effective_values.logging_retained_files
    ));
    report_lines.push(format!(
        "Logging directory: {}",
        effective_values.logging_directory.display()
    ));
    report_lines.push(format!(
        "Performance attribution: {}",
        enabled_word(effective_values.performance_attribution_enabled)
    ));
    let rendered_text = report_lines.join("\n");
    writeln!(rendered_output, "{rendered_text}")
}

fn render_json_report(
    effective_values: &EffectiveConfigurationValues,
    rendered_output: &mut dyn io::Write,
) -> io::Result<()> {
    let report_document = serde_json::json!({
        "configuration_file_path": effective_values.configuration_file_path.display().to_string(),
        "generation": effective_values.generation,
        "runtime_instance": runtime_instance_slug(effective_values.runtime_instance),
        "supervisor_bind_address": effective_values.supervisor_bind_address.to_string(),
        "model_directories": effective_values
            .model_directories
            .iter()
            .map(|model_directory| model_directory.display().to_string())
            .collect::<Vec<_>>(),
        "configured_model_ids": effective_values.configured_model_ids,
        "persistent_prompt_cache_enabled": effective_values.persistent_prompt_cache_enabled,
        "maximum_mlx_memory_bytes": effective_values.maximum_mlx_memory_bytes,
        "logging": {
            "level": effective_values.logging_level,
            "retained_files": effective_values.logging_retained_files,
            "directory": effective_values.logging_directory.display().to_string(),
        },
        "performance_attribution_enabled": effective_values.performance_attribution_enabled,
    });
    let rendered_text =
        serde_json::to_string_pretty(&report_document).expect("report documents serialize");
    writeln!(rendered_output, "{rendered_text}")
}

fn runtime_instance_slug(runtime_instance: AstronomicalRuntimeInstance) -> &'static str {
    match runtime_instance {
        AstronomicalRuntimeInstance::Stable => "stable",
        AstronomicalRuntimeInstance::Development => "development",
    }
}

fn enabled_word(is_enabled: bool) -> &'static str {
    if is_enabled { "enabled" } else { "disabled" }
}

fn rendered_model_id_list(configured_model_ids: &[String]) -> String {
    if configured_model_ids.is_empty() {
        "none".to_owned()
    } else {
        configured_model_ids.join(", ")
    }
}

/// Renders a byte count as decimal SI gigabytes (1 GB = 1,000,000,000 bytes),
/// truncated to one tenth so the value never overstates available memory.
fn rendered_maximum_mlx_memory(maximum_mlx_memory_bytes: Option<u64>) -> String {
    const BYTES_PER_GIGABYTE: u64 = 1_000_000_000;
    const TENTHS_PER_GIGABYTE: u64 = 100_000_000;
    match maximum_mlx_memory_bytes {
        None => "not set".to_owned(),
        Some(total_bytes) => {
            let whole_gigabytes = total_bytes / BYTES_PER_GIGABYTE;
            let gigabyte_tenths = (total_bytes % BYTES_PER_GIGABYTE) / TENTHS_PER_GIGABYTE;
            if gigabyte_tenths == 0 {
                format!("{whole_gigabytes} GB")
            } else {
                format!("{whole_gigabytes}.{gigabyte_tenths} GB")
            }
        }
    }
}
