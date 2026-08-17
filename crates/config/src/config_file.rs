use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use super::logging_config::LogLevel;
use super::{
    AstronomicalConfigError,
    DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
    DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
    DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS, DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB,
    DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS, maximum_mlx_memory_gb_to_bytes,
    parse_loopback_bind_address, prompt_cache_size_gb_to_bytes,
};

static TEMPORARY_CONFIG_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UserConfigFile {
    /// Absolute directories to recursively scan for Qwen3.5-family models.
    #[serde(default)]
    pub(crate) model_directories: Vec<PathBuf>,
    /// Per-request output-token ceiling. Defaults to 20,480 when not set.
    pub(crate) max_output_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) chunking: ChunkingConfigFile,
    /// Enables bounded model-loading and request performance-attribution reports.
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    pub(crate) performance_attribution_enabled: Option<bool>,
    /// Enables the SSD-backed persistent prompt and key-value cache.
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    pub(crate) persistent_prompt_cache_enabled: Option<bool>,
    /// Optional decimal SI GB ceiling for MLX memory. Omission selects the machine maximum.
    pub(crate) maximum_mlx_memory_gb: Option<u64>,
    /// Enables qualified Qwen multi-token prediction when the artifact supports it.
    /// Defaults to enabled when omitted.
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    pub(crate) mtp_enabled: Option<bool>,
    /// Optional fixed MTP proposal depth. Omission selects artifact metadata.
    pub(crate) mtp_draft_depth: Option<u8>,
    /// Optional draft-assisted sparse prompt-prefill policy.
    #[serde(default)]
    pub(crate) speculative_prefill: SpeculativePrefillConfigFile,
    pub(crate) supervisor: Option<SupervisorConfigFile>,
    pub(crate) prompt_cache_max_size_gb: Option<u64>,
    pub(crate) logging: Option<LoggingConfigFile>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChunkingConfigFile {
    /// Prompt-processing chunk length; omission selects the qualified default.
    pub(crate) fixed_prompt_processing_chunk_size_tokens: Option<u32>,
    /// Fixed prompt-processing chunk length while sparse experts stream from storage.
    ///
    /// Omitted values keep the complete-resident fixed size for every residency
    /// mode. A smaller positive value shortens each forward only when experts
    /// are not fully retained in active memory.
    pub(crate) fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    /// Append-only attention capacity added by each storage growth operation.
    pub(crate) full_attention_key_value_growth_tokens: Option<u32>,
    /// Maximum prompt rows evaluated by one speculative drafter forward.
    pub(crate) speculative_prefill_draft_forward_tokens: Option<u32>,
    /// Experimental multi-token layer interval used only while experts stream from disk.
    pub(crate) experimental_ssd_paging_prefill_graph_submission_layer_interval: Option<u32>,
    /// Experimental one-token layer interval used only while experts stream from disk.
    pub(crate) experimental_ssd_paging_generation_graph_submission_layer_interval: Option<u32>,
    /// Exact persistent-cache block length, or null for model-derived sizing.
    pub(crate) prompt_cache_block_tokens: Option<u32>,
    /// Cache-block interval between retained common-prefix restart checkpoints.
    pub(crate) prompt_cache_common_prefix_stride_blocks: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpeculativePrefillConfigFile {
    #[serde(default, deserialize_with = "deserialize_present_boolean")]
    pub(crate) enabled: Option<bool>,
    pub(crate) target_model_id: Option<String>,
    pub(crate) draft_model_id: Option<String>,
    pub(crate) minimum_prompt_tokens: Option<u32>,
    pub(crate) keep_percentage: Option<u32>,
    pub(crate) selection_chunck_token_count: Option<u32>,
    pub(crate) mandatory_trailing_token_count: Option<u32>,
    pub(crate) lookahead_token_count: Option<u32>,
    pub(crate) importance_pooling_kernel_token_count: Option<u32>,
}

fn deserialize_present_boolean<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<bool>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    bool::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoggingConfigFile {
    #[serde(default)]
    pub(crate) level: LogLevel,
    pub(crate) retained_files: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SupervisorConfigFile {
    pub(crate) bind_address: Option<String>,
}

pub(crate) fn read_user_config_file(
    config_file_path: PathBuf,
) -> Result<UserConfigFile, AstronomicalConfigError> {
    let config_file_text = match fs::read_to_string(&config_file_path) {
        Ok(config_file_text) => config_file_text,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            let first_run_config_bytes = serde_json::to_vec_pretty(&serde_json::json!({
                "model_directories": [],
                "mtp_enabled": true,
                "persistent_prompt_cache_enabled": true,
                "chunking": {
                    "fixed_prompt_processing_chunk_size_tokens": DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
                    "full_attention_key_value_growth_tokens": DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
                    "speculative_prefill_draft_forward_tokens": DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
                    "experimental_ssd_paging_prefill_graph_submission_layer_interval": DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
                    "experimental_ssd_paging_generation_graph_submission_layer_interval": DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
                    "prompt_cache_block_tokens": null,
                    "prompt_cache_common_prefix_stride_blocks": DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS
                },
                "prompt_cache_max_size_gb": DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB,
            }))
            .map_err(|serialization_error| {
                AstronomicalConfigError::SerializeConfigFile {
                    config_file_path: config_file_path.clone(),
                    source: serialization_error,
                }
            })?;
            write_config_file_bytes_atomically(&config_file_path, &first_run_config_bytes)?;
            String::from_utf8(first_run_config_bytes).map_err(|source| {
                AstronomicalConfigError::ReadConfigFile {
                    config_file_path: config_file_path.clone(),
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, source),
                }
            })?
        }
        Err(source) => {
            return Err(AstronomicalConfigError::ReadConfigFile {
                config_file_path,
                source,
            });
        }
    };
    let user_config_file: UserConfigFile =
        serde_json::from_str(&config_file_text).map_err(|source| {
            AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?;
    validate_user_config_file(&user_config_file)?;
    Ok(user_config_file)
}

pub(crate) fn validate_user_config_file(
    user_config_file: &UserConfigFile,
) -> Result<(), AstronomicalConfigError> {
    for model_directory_path in &user_config_file.model_directories {
        validate_optional_absolute_path("model_directories", Some(model_directory_path))?;
    }
    super::chunking_config::ChunkingConfig::resolve(&user_config_file.chunking)?;
    super::speculative_prefill_config::resolve_speculative_prefill_config(
        &user_config_file.speculative_prefill,
    )?;
    if let Some(maximum_mlx_memory_gb) = user_config_file.maximum_mlx_memory_gb {
        maximum_mlx_memory_gb_to_bytes(maximum_mlx_memory_gb)?;
    }
    if user_config_file
        .mtp_draft_depth
        .is_some_and(|depth| !(1..=3).contains(&depth))
    {
        return Err(AstronomicalConfigError::InvalidMtpDraftDepth);
    }
    if let Some(supervisor_config) = user_config_file.supervisor.as_ref() {
        if let Some(bind_address) = supervisor_config.bind_address.as_ref() {
            parse_loopback_bind_address(bind_address)?;
        }
    }
    if let Some(prompt_cache_max_size_gb) = user_config_file.prompt_cache_max_size_gb {
        prompt_cache_size_gb_to_bytes(prompt_cache_max_size_gb)?;
    }
    if let Some(logging_config) = user_config_file.logging.as_ref() {
        if logging_config.retained_files == Some(0) {
            return Err(AstronomicalConfigError::InvalidRetainedLogFileCount);
        }
    }
    Ok(())
}

pub(crate) fn write_config_file_bytes_atomically(
    config_file_path: &Path,
    config_file_bytes: &[u8],
) -> Result<(), AstronomicalConfigError> {
    let Some(config_directory_path) = config_file_path.parent() else {
        return Err(AstronomicalConfigError::WriteConfigFile {
            config_file_path: config_file_path.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "config file has no parent directory",
            ),
        });
    };
    fs::create_dir_all(config_directory_path).map_err(|source| {
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
    File::open(config_directory_path)
        .and_then(|config_directory| config_directory.sync_all())
        .map_err(|source| AstronomicalConfigError::WriteConfigFile {
            config_file_path: config_file_path.to_owned(),
            source,
        })
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

fn validate_optional_absolute_path(
    field_name: impl Into<String>,
    configured_path: Option<&PathBuf>,
) -> Result<(), AstronomicalConfigError> {
    if let Some(configured_path) = configured_path
        && !configured_path.is_absolute()
    {
        return Err(AstronomicalConfigError::PathMustBeAbsolute {
            field_name: field_name.into(),
            configured_path: configured_path.clone(),
        });
    }
    Ok(())
}
