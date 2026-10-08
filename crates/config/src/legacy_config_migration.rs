//! Validates the unversioned document and atomically migrates representable user intent to v1.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use crate::chunking_config::ChunkingConfigFile;
use crate::config_document::{
    DiagnosticsConfigFile, GenerationDefaultsConfigFile, ModelConfigFile, PromptCacheConfigFile,
    RuntimeConfigFile, UserConfigFile,
};
use crate::config_file::{
    parse_and_validate_v1, read_existing_config_file_bytes, strip_retired_config_fields,
    write_adjacent_schema, write_config_file_bytes_atomically,
};
use crate::{AstronomicalConfigError, LogLevel};

const LEGACY_CONFIG_BACKUP_FILE_NAME: &str = "config.legacy-v0.json";

pub(crate) fn migrate_legacy_config(
    config_file_path: &Path,
    legacy_config_bytes: &[u8],
    legacy_json: serde_json::Value,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let migration_started_at = Instant::now();
    tracing::info!(operation = "legacy-config-migration", status = "start");
    let migration_result =
        execute_legacy_config_migration(config_file_path, legacy_config_bytes, legacy_json);
    match &migration_result {
        Ok(_) => tracing::info!(
            operation = "legacy-config-migration",
            status = "success",
            elapsed_milliseconds = migration_started_at.elapsed().as_millis()
        ),
        Err(error) => tracing::warn!(
            operation = "legacy-config-migration",
            status = "failed",
            elapsed_milliseconds = migration_started_at.elapsed().as_millis(),
            error = %error
        ),
    }
    migration_result
}

fn execute_legacy_config_migration(
    config_file_path: &Path,
    legacy_config_bytes: &[u8],
    legacy_json: serde_json::Value,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let validated_config = prepare_legacy_config_migration(config_file_path, legacy_json)?;
    let migrated_bytes = serde_json::to_vec_pretty(&validated_config).map_err(|source| {
        AstronomicalConfigError::SerializeConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        }
    })?;
    // A successful one-way migration must retain recovery material before its commit point.
    preserve_legacy_config_backup(config_file_path, legacy_config_bytes)?;
    write_adjacent_schema(config_file_path)?;
    write_config_file_bytes_atomically(config_file_path, &migrated_bytes)?;
    Ok(validated_config)
}

pub(crate) fn preserve_legacy_config_backup(
    config_file_path: &Path,
    legacy_config_bytes: &[u8],
) -> Result<(), AstronomicalConfigError> {
    let config_directory_path =
        config_file_path
            .parent()
            .ok_or_else(|| AstronomicalConfigError::WriteConfigFile {
                config_file_path: config_file_path.to_owned(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "config file has no parent directory",
                ),
            })?;
    let legacy_backup_path = config_directory_path.join(LEGACY_CONFIG_BACKUP_FILE_NAME);
    let mut legacy_backup_file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&legacy_backup_path)
    {
        Ok(legacy_backup_file) => legacy_backup_file,
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
            return accept_matching_existing_backup(&legacy_backup_path, legacy_config_bytes);
        }
        Err(source) => {
            return Err(AstronomicalConfigError::WriteConfigFile {
                config_file_path: legacy_backup_path,
                source,
            });
        }
    };

    if let Err(source) = legacy_backup_file
        .write_all(legacy_config_bytes)
        .and_then(|()| legacy_backup_file.sync_all())
    {
        let _removed_incomplete_backup = fs::remove_file(&legacy_backup_path);
        return Err(AstronomicalConfigError::WriteConfigFile {
            config_file_path: legacy_backup_path,
            source,
        });
    }
    File::open(config_directory_path)
        .and_then(|config_directory| config_directory.sync_all())
        .map_err(|source| AstronomicalConfigError::WriteConfigFile {
            config_file_path: legacy_backup_path,
            source,
        })?;
    Ok(())
}

fn accept_matching_existing_backup(
    legacy_backup_path: &Path,
    legacy_config_bytes: &[u8],
) -> Result<(), AstronomicalConfigError> {
    let existing_backup_bytes = read_existing_config_file_bytes(legacy_backup_path)?;
    if existing_backup_bytes.as_deref() == Some(legacy_config_bytes) {
        return Ok(());
    }
    Err(AstronomicalConfigError::LegacyMigration {
        description: format!(
            "the one-time backup at {} already exists with different content; preserve both files and resolve the conflict before retrying",
            legacy_backup_path.display()
        ),
    })
}

/// Resolves legacy intent without writing so compare-and-commit callers retain ownership.
pub(crate) fn prepare_legacy_config_migration(
    config_file_path: &Path,
    mut legacy_json: serde_json::Value,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    strip_retired_config_fields(&mut legacy_json);
    let legacy_config: LegacyConfigFile =
        serde_json::from_value(legacy_json).map_err(|source| {
            AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            }
        })?;
    validate_legacy_config(&legacy_config)?;
    let discovered_model_ids = discover_model_ids_required_for_migration(&legacy_config)?;
    let migrated_config = build_migrated_config(legacy_config, &discovered_model_ids);
    let migrated_json = serde_json::to_value(&migrated_config).map_err(|source| {
        AstronomicalConfigError::SerializeConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        }
    })?;
    parse_and_validate_v1(config_file_path, migrated_json)
}

fn discover_model_ids_required_for_migration(
    legacy_config: &LegacyConfigFile,
) -> Result<Vec<String>, AstronomicalConfigError> {
    if legacy_config.max_output_tokens.is_none() {
        return Ok(Vec::new());
    }
    let directory_scans =
        crate::discover_models(&legacy_config.model_directories).map_err(|source| {
            AstronomicalConfigError::LegacyMigration {
                description: format!(
                    "could not discover models needed to preserve global policy: {source}"
                ),
            }
        })?;
    let discovered_model_ids: Vec<String> = directory_scans
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .map(|discovered_model| discovered_model.model_id)
        .collect();
    if discovered_model_ids.is_empty() {
        return Err(AstronomicalConfigError::LegacyMigration {
            description: "global model policy requires at least one currently discovered model; repair model_directories and retry"
                .to_owned(),
        });
    }
    Ok(discovered_model_ids)
}

fn build_migrated_config(
    legacy_config: LegacyConfigFile,
    discovered_model_ids: &[String],
) -> UserConfigFile {
    let mut models = BTreeMap::new();
    for model_id in discovered_model_ids {
        let model_config: &mut ModelConfigFile = models.entry(model_id.clone()).or_default();
        if let Some(maximum_output_tokens) = legacy_config.max_output_tokens {
            model_config.generation_defaults = Some(GenerationDefaultsConfigFile {
                maximum_output_tokens: Some(maximum_output_tokens),
                ..Default::default()
            });
        }
    }
    UserConfigFile {
        schema: "./astronomical-config.schema.json".to_owned(),
        schema_version: 1,
        runtime: RuntimeConfigFile {
            model_directories: legacy_config.model_directories,
            maximum_mlx_memory_gb: legacy_config.maximum_mlx_memory_gb,
            default_model: None,
        },
        prompt_cache: Some(PromptCacheConfigFile {
            enabled: legacy_config.persistent_prompt_cache_enabled,
            maximum_size_gb: legacy_config.prompt_cache_max_size_gb,
        }),
        chunking: Some(legacy_config.chunking),
        models,
        diagnostics: Some(DiagnosticsConfigFile {
            performance_attribution_enabled: legacy_config.performance_attribution_enabled,
            completion_attribution_enabled: None,
            log_level: legacy_config.logging.as_ref().map(|logging| logging.level),
            retained_log_files: legacy_config
                .logging
                .and_then(|logging| logging.retained_files),
        }),
    }
}

fn validate_legacy_config(legacy_config: &LegacyConfigFile) -> Result<(), AstronomicalConfigError> {
    for model_directory in &legacy_config.model_directories {
        if !model_directory.is_absolute() {
            return Err(AstronomicalConfigError::PathMustBeAbsolute {
                field_name: "model_directories".to_owned(),
                configured_path: model_directory.clone(),
            });
        }
    }
    crate::ChunkingConfig::resolve(&legacy_config.chunking)?;
    if legacy_config
        .supervisor
        .as_ref()
        .and_then(|supervisor| supervisor.bind_address.as_ref())
        .is_some()
    {
        return Err(AstronomicalConfigError::LegacyMigration {
            description: "legacy supervisor.bind_address cannot be represented because v1 derives the endpoint from the runtime channel; remove the setting to migrate"
                .to_owned(),
        });
    }
    if legacy_config.max_output_tokens == Some(0) {
        return Err(AstronomicalConfigError::LegacyMigration {
            description: "legacy max_output_tokens must be positive".to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyConfigFile {
    #[serde(default)]
    model_directories: Vec<PathBuf>,
    max_output_tokens: Option<u32>,
    #[serde(default)]
    chunking: LegacyChunkingConfigFile,
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    performance_attribution_enabled: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    persistent_prompt_cache_enabled: Option<bool>,
    maximum_mlx_memory_gb: Option<u64>,
    supervisor: Option<LegacySupervisorConfigFile>,
    prompt_cache_max_size_gb: Option<u64>,
    logging: Option<LegacyLoggingConfigFile>,
}

type LegacyChunkingConfigFile = ChunkingConfigFile;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyLoggingConfigFile {
    #[serde(default)]
    level: LogLevel,
    retained_files: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacySupervisorConfigFile {
    bind_address: Option<String>,
}

fn deserialize_present_boolean<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<bool>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    bool::deserialize(deserializer).map(Some)
}
