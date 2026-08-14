use std::fs::{self, File};
use std::path::Path;

use serde_json::Value;

use crate::config_error::AstronomicalConfigError;
use crate::config_file::{
    UserConfigFile, validate_user_config_file, write_config_file_bytes_atomically,
};

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

/// Persists the optional MLX memory override and returns the exact prior file bytes.
pub fn write_maximum_mlx_memory_gb(
    state_directory: impl AsRef<Path>,
    maximum_mlx_memory_gb: Option<u64>,
) -> Result<Option<Vec<u8>>, AstronomicalConfigError> {
    if let Some(maximum_mlx_memory_gb) = maximum_mlx_memory_gb {
        maximum_mlx_memory_gb_to_bytes(maximum_mlx_memory_gb)?;
    }

    let config_file_path = state_directory.as_ref().join("config.json");
    let prior_config_bytes = match fs::read(&config_file_path) {
        Ok(config_bytes) => Some(config_bytes),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(AstronomicalConfigError::ReadConfigFile {
                config_file_path,
                source,
            });
        }
    };

    let mut candidate_config = match prior_config_bytes.as_deref() {
        Some(config_bytes) => serde_json::from_slice::<Value>(config_bytes).map_err(|source| {
            AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?,
        None => Value::Object(serde_json::Map::new()),
    };
    let Some(candidate_config_object) = candidate_config.as_object_mut() else {
        return Err(AstronomicalConfigError::ConfigFileMustBeJsonObject { config_file_path });
    };

    match maximum_mlx_memory_gb {
        Some(maximum_mlx_memory_gb) => {
            candidate_config_object.insert(
                "maximum_mlx_memory_gb".to_owned(),
                Value::from(maximum_mlx_memory_gb),
            );
        }
        None => {
            candidate_config_object.remove("maximum_mlx_memory_gb");
        }
    }

    let candidate_user_config_file: UserConfigFile =
        serde_json::from_value(candidate_config.clone()).map_err(|source| {
            AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?;
    validate_user_config_file(&candidate_user_config_file)?;

    let candidate_config_bytes =
        serde_json::to_vec_pretty(&candidate_config).map_err(|source| {
            AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?;
    write_config_file_bytes_atomically(&config_file_path, &candidate_config_bytes)?;
    Ok(prior_config_bytes)
}

/// Restores the exact config bytes returned by write_maximum_mlx_memory_gb.
pub fn restore_config_file(
    state_directory: impl AsRef<Path>,
    prior_config_bytes: Option<&[u8]>,
) -> Result<(), AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join("config.json");
    match prior_config_bytes {
        Some(prior_config_bytes) => {
            write_config_file_bytes_atomically(&config_file_path, prior_config_bytes)
        }
        None => {
            match fs::remove_file(&config_file_path) {
                Ok(()) => {}
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(AstronomicalConfigError::WriteConfigFile {
                        config_file_path,
                        source,
                    });
                }
            }
            let config_directory_path = config_file_path.parent().ok_or_else(|| {
                AstronomicalConfigError::WriteConfigFile {
                    config_file_path: config_file_path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "config file has no parent directory",
                    ),
                }
            })?;
            File::open(config_directory_path)
                .and_then(|config_directory| config_directory.sync_all())
                .map_err(|source| AstronomicalConfigError::WriteConfigFile {
                    config_file_path,
                    source,
                })
        }
    }
}
