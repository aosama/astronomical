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
    AccelerationConfigFile, DiagnosticsConfigFile, GenerationDefaultsConfigFile, ModelConfigFile,
    MtpConfigFile, PromptCacheConfigFile, RuntimeConfigFile, SpeculativePrefillConfigFile,
    UserConfigFile,
};
use crate::config_file::{
    parse_and_validate_v1, read_existing_config_file_bytes, write_adjacent_schema,
    write_config_file_bytes_atomically,
};
use crate::{AstronomicalConfigError, LogLevel, SpeculativePrefillConfig, discover_models};

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
    legacy_json: serde_json::Value,
) -> Result<UserConfigFile, AstronomicalConfigError> {
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
    if legacy_config.max_output_tokens.is_none()
        && legacy_config.mtp_draft_depth.is_none()
        && legacy_config.mtp_enabled != Some(false)
    {
        return Ok(Vec::new());
    }
    let directory_scans = discover_models(&legacy_config.model_directories).map_err(|source| {
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
        if legacy_config.mtp_draft_depth.is_some() || legacy_config.mtp_enabled == Some(false) {
            model_config.acceleration = Some(AccelerationConfigFile {
                mtp: Some(MtpConfigFile {
                    enabled: legacy_config.mtp_enabled.filter(|mtp_enabled| !mtp_enabled),
                    draft_depth: legacy_config.mtp_draft_depth,
                }),
                ..Default::default()
            });
        }
    }
    migrate_speculative_prefill(&legacy_config.speculative_prefill, &mut models);
    UserConfigFile {
        schema: "./astronomical-config.schema.json".to_owned(),
        schema_version: 1,
        runtime: RuntimeConfigFile {
            model_directories: legacy_config.model_directories,
            maximum_mlx_memory_gb: legacy_config.maximum_mlx_memory_gb,
            experimental_qwen_thinking_channel_seed_enabled: None,
        },
        prompt_cache: Some(PromptCacheConfigFile {
            enabled: legacy_config.persistent_prompt_cache_enabled,
            maximum_size_gb: legacy_config.prompt_cache_max_size_gb,
        }),
        chunking: Some(legacy_config.chunking),
        models,
        diagnostics: Some(DiagnosticsConfigFile {
            performance_attribution_enabled: legacy_config.performance_attribution_enabled,
            log_level: legacy_config.logging.as_ref().map(|logging| logging.level),
            retained_log_files: legacy_config
                .logging
                .and_then(|logging| logging.retained_files),
        }),
    }
}

fn migrate_speculative_prefill(
    legacy_speculative_prefill: &LegacySpeculativePrefillConfigFile,
    models: &mut BTreeMap<String, ModelConfigFile>,
) {
    if legacy_speculative_prefill.enabled != Some(true) {
        return;
    }
    let Some(target_model_id) = legacy_speculative_prefill.target_model_id.as_ref() else {
        return;
    };
    let Some(draft_model_id) = legacy_speculative_prefill.draft_model_id.as_ref() else {
        return;
    };
    let model_config = models.entry(target_model_id.trim().to_owned()).or_default();
    let acceleration = model_config
        .acceleration
        .get_or_insert_with(AccelerationConfigFile::default);
    acceleration.speculative_prefill = Some(SpeculativePrefillConfigFile {
        draft_model_id: draft_model_id.trim().to_owned(),
        keep_percentage: legacy_speculative_prefill.keep_percentage,
        minimum_prompt_tokens: legacy_speculative_prefill.minimum_prompt_tokens,
    });
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
    if legacy_config
        .mtp_draft_depth
        .is_some_and(|draft_depth| !(1..=3).contains(&draft_depth))
    {
        return Err(AstronomicalConfigError::InvalidMtpDraftDepth);
    }
    validate_legacy_speculative_prefill(&legacy_config.speculative_prefill)
}

fn validate_legacy_speculative_prefill(
    speculative_prefill: &LegacySpeculativePrefillConfigFile,
) -> Result<(), AstronomicalConfigError> {
    if speculative_prefill
        .target_model_id
        .as_deref()
        .is_some_and(|model_id| model_id.trim().is_empty())
    {
        return Err(AstronomicalConfigError::SpeculativePrefillTargetModelIdMustNotBeEmpty);
    }
    if speculative_prefill
        .draft_model_id
        .as_deref()
        .is_some_and(|model_id| model_id.trim().is_empty())
    {
        return Err(AstronomicalConfigError::SpeculativePrefillDraftModelIdMustNotBeEmpty);
    }
    if speculative_prefill.enabled == Some(true) {
        if speculative_prefill.target_model_id.is_none() {
            return Err(AstronomicalConfigError::SpeculativePrefillTargetModelRequired);
        }
        if speculative_prefill.draft_model_id.is_none() {
            return Err(AstronomicalConfigError::SpeculativePrefillDraftModelRequired);
        }
        if speculative_prefill.keep_percentage.is_none() {
            return Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageRequired);
        }
    }
    if speculative_prefill
        .keep_percentage
        .is_some_and(|keep_percentage| !(1..=100).contains(&keep_percentage))
    {
        return Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageOutOfRange);
    }
    if speculative_prefill.minimum_prompt_tokens == Some(0) {
        return Err(AstronomicalConfigError::SpeculativePrefillMinimumPromptTokensMustBePositive);
    }
    for (configured_value, retired_field_name, v1_value) in [
        (
            speculative_prefill.selection_chunck_token_count,
            "selection_chunck_token_count",
            SpeculativePrefillConfig::DEFAULT_SELECTION_CHUNK_TOKEN_COUNT,
        ),
        (
            speculative_prefill.mandatory_trailing_token_count,
            "mandatory_trailing_token_count",
            SpeculativePrefillConfig::DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT,
        ),
        (
            speculative_prefill.lookahead_token_count,
            "lookahead_token_count",
            SpeculativePrefillConfig::DEFAULT_LOOKAHEAD_TOKEN_COUNT,
        ),
        (
            speculative_prefill.importance_pooling_kernel_token_count,
            "importance_pooling_kernel_token_count",
            SpeculativePrefillConfig::DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT,
        ),
    ] {
        if configured_value.is_some_and(|configured_value| configured_value != v1_value) {
            return Err(AstronomicalConfigError::LegacyMigration {
                description: format!(
                    "legacy speculative_prefill.{retired_field_name} differs from the fixed v1 execution policy and cannot be migrated without changing behavior"
                ),
            });
        }
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
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    mtp_enabled: Option<bool>,
    mtp_draft_depth: Option<u8>,
    #[serde(default)]
    speculative_prefill: LegacySpeculativePrefillConfigFile,
    supervisor: Option<LegacySupervisorConfigFile>,
    prompt_cache_max_size_gb: Option<u64>,
    logging: Option<LegacyLoggingConfigFile>,
}

type LegacyChunkingConfigFile = ChunkingConfigFile;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacySpeculativePrefillConfigFile {
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    enabled: Option<bool>,
    target_model_id: Option<String>,
    draft_model_id: Option<String>,
    minimum_prompt_tokens: Option<u32>,
    keep_percentage: Option<u32>,
    selection_chunck_token_count: Option<u32>,
    mandatory_trailing_token_count: Option<u32>,
    lookahead_token_count: Option<u32>,
    importance_pooling_kernel_token_count: Option<u32>,
}

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
