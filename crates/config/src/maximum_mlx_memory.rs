use std::path::Path;

use crate::config_document::UserConfigFile;
use crate::config_error::AstronomicalConfigError;
use crate::config_file::{
    parse_and_validate_v1, read_existing_config_file_bytes, validate_user_config_file,
    write_adjacent_schema, write_config_file_bytes_atomically,
};
use crate::duplicate_key_json::parse_json_rejecting_duplicates;
use crate::legacy_config_migration::prepare_legacy_config_migration;

const BYTES_PER_DECIMAL_GIGABYTE: u64 = 1_000_000_000;

/// Converts a positive decimal SI gigabyte setting to exact bytes.
pub fn maximum_mlx_memory_gb_to_bytes(
    maximum_mlx_memory_gb: u64,
) -> Result<u64, AstronomicalConfigError> {
    if maximum_mlx_memory_gb == 0 {
        return Err(AstronomicalConfigError::InvalidMaximumMlxMemoryGb {
            description: "maximum MLX memory must be positive",
        });
    }
    maximum_mlx_memory_gb
        .checked_mul(BYTES_PER_DECIMAL_GIGABYTE)
        .ok_or(AstronomicalConfigError::InvalidMaximumMlxMemoryGb {
            description: "maximum MLX memory exceeds the byte range",
        })
}

/// Exact before/after bytes for one atomic memory configuration mutation.
pub struct MaximumMlxMemoryConfigUpdate {
    pub prior_config_bytes: Option<Vec<u8>>,
    pub candidate_config_bytes: Vec<u8>,
}

/// Builds a validated byte transaction without mutating the source of truth.
pub fn prepare_maximum_mlx_memory_gb_update(
    state_directory: impl AsRef<Path>,
    maximum_mlx_memory_gb: Option<u64>,
) -> Result<MaximumMlxMemoryConfigUpdate, AstronomicalConfigError> {
    if let Some(maximum_mlx_memory_gb) = maximum_mlx_memory_gb {
        maximum_mlx_memory_gb_to_bytes(maximum_mlx_memory_gb)?;
    }

    let config_file_path = state_directory.as_ref().join("config.json");
    let prior_config_bytes = read_existing_config_file_bytes(&config_file_path)?;
    let mut candidate_user_config_file = match prior_config_bytes.as_deref() {
        Some(config_file_bytes) => {
            let config_json =
                parse_json_rejecting_duplicates(&config_file_path, config_file_bytes)?;
            if config_json.get("schema_version").is_none() {
                prepare_legacy_config_migration(&config_file_path, config_json)?
            } else {
                parse_and_validate_v1(&config_file_path, config_json)?
            }
        }
        None => UserConfigFile::minimal(),
    };
    candidate_user_config_file.runtime.maximum_mlx_memory_gb = maximum_mlx_memory_gb;
    validate_user_config_file(&candidate_user_config_file)?;

    let candidate_config_bytes =
        serde_json::to_vec_pretty(&candidate_user_config_file).map_err(|source| {
            AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?;
    Ok(MaximumMlxMemoryConfigUpdate {
        prior_config_bytes,
        candidate_config_bytes,
    })
}

/// Commits a prepared update only while the document it was based on still owns the file.
pub fn commit_maximum_mlx_memory_gb_update(
    state_directory: impl AsRef<Path>,
    config_update: &MaximumMlxMemoryConfigUpdate,
) -> Result<(), AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join("config.json");
    if read_existing_config_file_bytes(&config_file_path)? != config_update.prior_config_bytes {
        return Err(AstronomicalConfigError::ConfigChangedDuringUpdate);
    }
    // The schema precedes the document so every committed config remains locally inspectable.
    write_adjacent_schema(&config_file_path)?;
    write_config_file_bytes_atomically(&config_file_path, &config_update.candidate_config_bytes)
}

/// Persists the optional MLX memory override and returns its exact byte transaction.
pub fn write_maximum_mlx_memory_gb(
    state_directory: impl AsRef<Path>,
    maximum_mlx_memory_gb: Option<u64>,
) -> Result<MaximumMlxMemoryConfigUpdate, AstronomicalConfigError> {
    let state_directory = state_directory.as_ref();
    let config_update =
        prepare_maximum_mlx_memory_gb_update(state_directory, maximum_mlx_memory_gb)?;
    commit_maximum_mlx_memory_gb_update(state_directory, &config_update)?;
    Ok(config_update)
}
