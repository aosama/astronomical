//! Owns first-run creation, strict v1 loading, validation, and atomic persistence.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config_document::UserConfigFile;
use crate::duplicate_key_json::parse_json_rejecting_duplicates;
use crate::legacy_config_migration::migrate_legacy_config;
use crate::{AstronomicalConfigError, maximum_mlx_memory_gb_to_bytes};

const CONFIG_SCHEMA_FILE_NAME: &str = "astronomical-config.schema.json";
const CONFIG_SCHEMA_BYTES: &[u8] =
    include_bytes!("../../../site/schemas/config/v1/astronomical-config.schema.json");
const MAXIMUM_CONFIG_FILE_BYTES: usize = 1_048_576;
static TEMPORARY_CONFIG_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn read_user_config_file(
    config_file_path: PathBuf,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let config_file_bytes = match read_existing_config_file_bytes(&config_file_path)? {
        Some(config_file_bytes) => config_file_bytes,
        None => {
            return create_first_run_config(&config_file_path);
        }
    };
    let config_json = parse_json_rejecting_duplicates(&config_file_path, &config_file_bytes)?;
    if config_json.get("schema_version").is_none() {
        let mut migrated_user_config =
            migrate_legacy_config(&config_file_path, &config_file_bytes, config_json)?;
        persist_mandatory_chunking_fields(&config_file_path, &mut migrated_user_config)?;
        return Ok(migrated_user_config);
    }
    let mut user_config_file = parse_and_validate_v1(&config_file_path, config_json)?;
    persist_mandatory_chunking_fields(&config_file_path, &mut user_config_file)?;
    Ok(user_config_file)
}

pub(crate) fn parse_and_validate_v1(
    config_file_path: &Path,
    config_json: serde_json::Value,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let user_config_file: UserConfigFile =
        serde_json::from_value(config_json).map_err(|source| {
            AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            }
        })?;
    validate_user_config_file(&user_config_file)?;
    Ok(user_config_file)
}

fn persist_mandatory_chunking_fields(
    config_file_path: &Path,
    user_config_file: &mut UserConfigFile,
) -> Result<(), AstronomicalConfigError> {
    let chunking = user_config_file
        .chunking
        .get_or_insert_with(Default::default);
    let mut should_persist = false;
    if chunking
        .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
        .is_none()
    {
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens =
            Some(crate::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS);
        should_persist = true;
    }
    if chunking
        .experimental_ssd_paging_prefill_graph_submission_layer_interval
        .is_none()
    {
        // The former prefill_graph_submission_layer_interval owned SSD-paged
        // prefill; resident submission was hardcoded to zero. Split the value
        // once so later resident edits cannot change paging, and vice versa.
        chunking.experimental_ssd_paging_prefill_graph_submission_layer_interval =
            Some(chunking.prefill_graph_submission_layer_interval.unwrap_or(
                crate::DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            ));
        chunking.prefill_graph_submission_layer_interval =
            Some(crate::DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL);
        should_persist = true;
    }
    if !should_persist {
        return Ok(());
    }
    write_adjacent_schema(config_file_path)?;
    validate_user_config_file(user_config_file)?;
    let persisted_config_bytes = serde_json::to_vec_pretty(user_config_file).map_err(|source| {
        AstronomicalConfigError::SerializeConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        }
    })?;
    write_config_file_bytes_atomically(config_file_path, &persisted_config_bytes)
}

pub(crate) fn validate_user_config_file(
    user_config_file: &UserConfigFile,
) -> Result<(), AstronomicalConfigError> {
    if user_config_file.schema_version != 1 {
        return Err(AstronomicalConfigError::UnsupportedSchemaVersion {
            schema_version: user_config_file.schema_version,
        });
    }
    if user_config_file.schema != "./astronomical-config.schema.json" {
        return Err(AstronomicalConfigError::InvalidSchemaReference);
    }
    for model_directory_path in &user_config_file.runtime.model_directories {
        if !model_directory_path.is_absolute() {
            return Err(AstronomicalConfigError::PathMustBeAbsolute {
                field_name: "runtime.model_directories".to_owned(),
                configured_path: model_directory_path.clone(),
            });
        }
    }
    if let Some(maximum_mlx_memory_gb) = user_config_file.runtime.maximum_mlx_memory_gb {
        maximum_mlx_memory_gb_to_bytes(maximum_mlx_memory_gb)?;
    }
    user_config_file.validate()
}

fn create_first_run_config(
    config_file_path: &Path,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let first_run_config = UserConfigFile::minimal();
    let first_run_config_bytes =
        serde_json::to_vec_pretty(&first_run_config).map_err(|source| {
            AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            }
        })?;
    write_adjacent_schema(config_file_path)?;
    write_config_file_bytes_atomically(config_file_path, &first_run_config_bytes)?;
    Ok(first_run_config)
}

pub(crate) fn write_adjacent_schema(
    config_file_path: &Path,
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
    write_config_file_bytes_atomically(
        &config_directory_path.join(CONFIG_SCHEMA_FILE_NAME),
        CONFIG_SCHEMA_BYTES,
    )
}

pub(crate) fn read_existing_config_file_bytes(
    config_file_path: &Path,
) -> Result<Option<Vec<u8>>, AstronomicalConfigError> {
    let config_file = match File::open(config_file_path) {
        Ok(config_file) => config_file,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(AstronomicalConfigError::ReadConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            });
        }
    };
    let mut config_file_bytes = Vec::new();
    config_file
        .take((MAXIMUM_CONFIG_FILE_BYTES + 1) as u64)
        .read_to_end(&mut config_file_bytes)
        .map_err(|source| AstronomicalConfigError::ReadConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        })?;
    if config_file_bytes.len() > MAXIMUM_CONFIG_FILE_BYTES {
        return Err(AstronomicalConfigError::ConfigFileTooLarge {
            config_file_path: config_file_path.to_owned(),
            maximum_bytes: MAXIMUM_CONFIG_FILE_BYTES,
        });
    }
    Ok(Some(config_file_bytes))
}

pub(crate) fn write_config_file_bytes_atomically(
    config_file_path: &Path,
    config_file_bytes: &[u8],
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
    fs::create_dir_all(config_directory_path).map_err(|source| {
        AstronomicalConfigError::WriteConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        }
    })?;
    let config_directory = File::open(config_directory_path).map_err(|source| {
        AstronomicalConfigError::WriteConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        }
    })?;
    let temporary_config_file_path =
        create_temporary_config_file(config_directory_path, config_file_bytes, config_file_path)?;
    if let Err(source) = fs::rename(&temporary_config_file_path, config_file_path) {
        let _removed_temporary_config_file = fs::remove_file(&temporary_config_file_path);
        return Err(AstronomicalConfigError::WriteConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        });
    }
    // Rename is the atomic commit point. A later sync error cannot be reported as a
    // failed write because callers would incorrectly attempt to restore replaced bytes.
    if let Err(directory_sync_error) = config_directory.sync_all() {
        tracing::warn!(
            config_file = %config_file_path.display(),
            error = %directory_sync_error,
            "configuration was committed but its directory entry could not be synchronized"
        );
    }
    Ok(())
}

fn create_temporary_config_file(
    config_directory_path: &Path,
    config_file_bytes: &[u8],
    config_file_path: &Path,
) -> Result<PathBuf, AstronomicalConfigError> {
    for attempt_number in 0..100_u64 {
        let sequence_number = TEMPORARY_CONFIG_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_config_file_path = config_directory_path.join(format!(
            ".config.json.tmp.{}.{}",
            std::process::id(),
            sequence_number.saturating_add(attempt_number),
        ));
        let mut temporary_config_file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary_config_file_path)
        {
            Ok(temporary_config_file) => temporary_config_file,
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(AstronomicalConfigError::WriteConfigFile {
                    config_file_path: config_file_path.to_owned(),
                    source,
                });
            }
        };
        if let Err(source) = temporary_config_file
            .write_all(config_file_bytes)
            .and_then(|()| temporary_config_file.sync_all())
        {
            let _removed_temporary_config_file = fs::remove_file(&temporary_config_file_path);
            return Err(AstronomicalConfigError::WriteConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            });
        }
        return Ok(temporary_config_file_path);
    }
    Err(AstronomicalConfigError::WriteConfigFile {
        config_file_path: config_file_path.to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary config file",
        ),
    })
}
