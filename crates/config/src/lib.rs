#![forbid(unsafe_code)]

use std::{env, net::SocketAddr, path::PathBuf};

mod config_error;
mod config_file;
mod logging_config;
mod maximum_mlx_memory;
mod model_discovery;
mod model_discovery_huggingface_cache;
mod model_identity;
mod prompt_cache_config;

pub use config_error::AstronomicalConfigError;
pub use logging_config::{LogLevel, LoggingConfig};
pub use maximum_mlx_memory::{
    maximum_mlx_memory_gb_to_bytes, restore_config_file, write_maximum_mlx_memory_gb,
};
pub use model_discovery::{
    DiscoveredModel, DiscoveredModelError, ModelDiscoveryDirectoryScan, discover_qwen3_5_models,
};
pub use model_identity::{decode_huggingface_cache_directory_name, resolve_model_id};
pub use prompt_cache_config::PromptCacheConfig;

use config_file::{UserConfigFile, config_file_path_for_home, read_user_config_file};

const CONFIG_DIRECTORY_NAME: &str = ".astronomical";
const CONFIG_FILE_NAME: &str = "config.json";
const DEFAULT_SUPERVISOR_BIND_ADDRESS: &str = "127.0.0.1:6732";
pub const DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB: u64 = 50;
const BYTES_PER_CONFIGURED_GIGABYTE: u64 = 1_000_000_000;
const DEFAULT_RETAINED_LOG_FILES: usize = 7;

/// User-level Astronomical runtime configuration loaded from `~/.astronomical/config.json`.
#[derive(Clone, Debug)]
pub struct AstronomicalConfig {
    home_directory: Option<PathBuf>,
    user_config_file: UserConfigFile,
}

impl AstronomicalConfig {
    /// Loads `$HOME/.astronomical/config.json` when a home directory is known.
    /// Missing files are accepted; malformed existing files fail startup.
    pub fn load_from_default_location() -> Result<Self, AstronomicalConfigError> {
        let Some(home_directory) = env::var_os("HOME").map(PathBuf::from) else {
            return Ok(Self {
                home_directory: None,
                user_config_file: UserConfigFile::default(),
            });
        };
        Self::load_from_home_directory(home_directory)
    }

    /// Loads the strict user config beneath an explicit home directory.
    pub fn load_from_home_directory(
        home_directory: impl Into<PathBuf>,
    ) -> Result<Self, AstronomicalConfigError> {
        let home_directory = home_directory.into();
        let config_file_path = config_file_path_for_home(&home_directory);
        let user_config_file = read_user_config_file(config_file_path)?;
        Ok(Self {
            home_directory: Some(home_directory),
            user_config_file,
        })
    }

    /// Returns absolute directories to recursively scan for supported models.
    #[must_use]
    pub fn model_directories(&self) -> &[PathBuf] {
        &self.user_config_file.model_directories
    }

    /// Finds one exact model ID beneath the configured recursive model roots.
    ///
    /// The discovery rules are shared with supervisor model registration, so the
    /// returned path is an executable discovered model directory rather than an
    /// arbitrary directory whose name happens to match.
    pub fn find_configured_model_directory_by_id(
        &self,
        model_id: &str,
    ) -> Result<Option<PathBuf>, DiscoveredModelError> {
        let configured_model_directory_scans =
            discover_qwen3_5_models(self.model_directories(), self.max_output_tokens())?;
        Ok(configured_model_directory_scans
            .into_iter()
            .flat_map(|configured_model_directory_scan| {
                configured_model_directory_scan.discovered_models
            })
            .find(|discovered_model| discovered_model.model_id == model_id)
            .map(|discovered_model| discovered_model.model_directory))
    }

    /// Resolves whether Qwen3.5-MoE uses adaptive or fixed prompt-processing chunks.
    pub fn prefill_chunck_sizing_policy(
        &self,
    ) -> Result<PrefillChunckSizingPolicy, AstronomicalConfigError> {
        resolve_prefill_chunck_sizing_policy(
            self.user_config_file.prefill_chunck_size_optimizer_enabled,
            self.user_config_file.fixed_prefill_chunck_tokens,
        )
    }

    /// Returns the `fixed_prefill_chunck_tokens` value the daemon ignored because
    /// `prefill_chunck_size_optimizer_enabled` was `true`. The menu bar app flashes a
    /// callout when this is `Some` so the user knows the fixed value has no effect.
    #[must_use]
    pub fn ignored_fixed_prefill_chunck_tokens(&self) -> Option<u32> {
        resolve_ignored_fixed_prefill_chunck_tokens(
            self.user_config_file.prefill_chunck_size_optimizer_enabled,
            self.user_config_file.fixed_prefill_chunck_tokens,
        )
    }

    /// Resolves and validates the loopback-only HTTP bind address.
    pub fn supervisor_bind_address(&self) -> Result<SocketAddr, AstronomicalConfigError> {
        let raw_bind_address = self
            .user_config_file
            .supervisor
            .as_ref()
            .and_then(|supervisor_config| supervisor_config.bind_address.clone())
            .unwrap_or_else(|| DEFAULT_SUPERVISOR_BIND_ADDRESS.to_owned());
        parse_loopback_bind_address(&raw_bind_address)
    }

    /// Resolves the always-enabled SSD-backed prompt cache policy.
    pub fn prompt_cache(&self) -> Result<PromptCacheConfig, AstronomicalConfigError> {
        let global_prompt_cache_root_directory =
            self.default_global_prompt_cache_root_directory()?;
        let global_prompt_cache_maximum_size_bytes = prompt_cache_size_gb_to_bytes(
            self.user_config_file
                .prompt_cache_max_size_gb
                .unwrap_or(DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB),
        )?;
        Ok(PromptCacheConfig::new(
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
        ))
    }

    /// Returns whether the SSD-backed prompt cache is enabled.
    #[must_use]
    pub fn persistent_prompt_cache_enabled(&self) -> bool {
        self.user_config_file
            .persistent_prompt_cache_enabled
            .unwrap_or(true)
    }

    /// Resolves bounded hourly file logging for the supervisor or worker.
    pub fn logging(&self) -> Result<LoggingConfig, AstronomicalConfigError> {
        let home_directory = self
            .home_directory
            .as_ref()
            .ok_or(AstronomicalConfigError::DefaultLogDirectoryRequiresHome)?;
        let configured_logging = self.user_config_file.logging.as_ref();
        Ok(LoggingConfig::new(
            home_directory.join(CONFIG_DIRECTORY_NAME).join("logs"),
            configured_logging.map_or(LogLevel::Warn, |logging| logging.level),
            configured_logging
                .and_then(|logging| logging.retained_files)
                .unwrap_or(DEFAULT_RETAINED_LOG_FILES),
        ))
    }

    fn default_global_prompt_cache_root_directory(
        &self,
    ) -> Result<PathBuf, AstronomicalConfigError> {
        self.home_directory
            .as_ref()
            .map(|home_directory| home_directory.join(CONFIG_DIRECTORY_NAME).join("cache"))
            .ok_or(AstronomicalConfigError::DefaultPromptCacheDirectoryRequiresHome)
    }

    /// Resolves the optimizer state directory for persisting prefill chunk-size
    /// optimizer observations across restarts.
    ///
    /// Defaults to `~/.astronomical/optimizer/`. Always enabled; there is no
    /// config toggle to disable optimizer persistence.
    pub fn optimizer_directory(&self) -> Result<PathBuf, AstronomicalConfigError> {
        self.home_directory
            .as_ref()
            .map(|home_directory| home_directory.join(CONFIG_DIRECTORY_NAME).join("optimizer"))
            .ok_or(AstronomicalConfigError::DefaultOptimizerDirectoryRequiresHome)
    }

    /// Returns the per-request output-token ceiling.
    ///
    /// Defaults to 20,480 when not explicitly configured.
    #[must_use]
    pub fn max_output_tokens(&self) -> u32 {
        self.user_config_file.max_output_tokens.unwrap_or(20_480)
    }

    /// Returns whether detailed critical-path performance attribution is enabled.
    ///
    /// Attribution defaults to disabled so normal inference avoids timing and
    /// serialization work unless the user explicitly requests diagnostics.
    #[must_use]
    pub fn performance_attribution_enabled(&self) -> bool {
        self.user_config_file
            .performance_attribution_enabled
            .unwrap_or(false)
    }

    /// Returns whether qualified Qwen multi-token prediction is enabled.
    ///
    /// MTP is enabled by default when the setting is omitted from the user configuration.
    #[must_use]
    pub fn mtp_enabled(&self) -> bool {
        self.user_config_file.mtp_enabled.unwrap_or(true)
    }

    /// Resolves the optional user MLX memory ceiling in decimal SI bytes.
    ///
    /// `None` means that startup or live runtime control should use the machine maximum.
    pub fn maximum_mlx_memory_bytes(&self) -> Result<Option<u64>, AstronomicalConfigError> {
        self.user_config_file
            .maximum_mlx_memory_gb
            .map(maximum_mlx_memory_gb_to_bytes)
            .transpose()
    }
}

/// Resolved Qwen3.5-MoE prompt-processing chunk selection at worker startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillChunckSizingPolicy {
    /// Select chunk sizes online from measured context-specific observations.
    Optimized,
    /// Process each normal full chunk with one configured token count.
    Fixed {
        /// Required positive token count for each fixed prefill chunk.
        fixed_prefill_chunck_tokens: u32,
    },
}

fn resolve_prefill_chunck_sizing_policy(
    configured_prefill_chunck_size_optimizer_enabled: Option<bool>,
    configured_fixed_prefill_chunck_tokens: Option<u32>,
) -> Result<PrefillChunckSizingPolicy, AstronomicalConfigError> {
    let fixed_prefill_chunck_tokens = match configured_prefill_chunck_size_optimizer_enabled {
        Some(false) => configured_fixed_prefill_chunck_tokens.ok_or(
            AstronomicalConfigError::FixedPrefillChunckTokensRequiredWhenOptimizerDisabled,
        )?,
        Some(true) | None => {
            // Any configured fixed size is intentionally ignored in explicit
            // optimized mode. The menu bar surfaces that override separately.
            return Ok(PrefillChunckSizingPolicy::Optimized);
        }
    };
    if fixed_prefill_chunck_tokens == 0 {
        return Err(AstronomicalConfigError::InvalidFixedPrefillChunckTokens);
    }
    Ok(PrefillChunckSizingPolicy::Fixed {
        fixed_prefill_chunck_tokens,
    })
}

/// Returns the `fixed_prefill_chunck_tokens` value the daemon ignores because the
/// optimizer is enabled, so the menu bar app can flash a callout warning the user.
fn resolve_ignored_fixed_prefill_chunck_tokens(
    configured_prefill_chunck_size_optimizer_enabled: Option<bool>,
    configured_fixed_prefill_chunck_tokens: Option<u32>,
) -> Option<u32> {
    if configured_prefill_chunck_size_optimizer_enabled == Some(true) {
        configured_fixed_prefill_chunck_tokens
    } else {
        None
    }
}

fn parse_loopback_bind_address(
    raw_bind_address: &str,
) -> Result<SocketAddr, AstronomicalConfigError> {
    let supervisor_bind_address = raw_bind_address.parse::<SocketAddr>().map_err(|source| {
        AstronomicalConfigError::ParseBindAddress {
            raw_bind_address: raw_bind_address.to_owned(),
            source,
        }
    })?;
    if !supervisor_bind_address.ip().is_loopback() {
        return Err(AstronomicalConfigError::NonLoopbackBindAddress {
            supervisor_bind_address,
        });
    }
    Ok(supervisor_bind_address)
}

fn prompt_cache_size_gb_to_bytes(
    prompt_cache_max_size_gb: u64,
) -> Result<u64, AstronomicalConfigError> {
    if prompt_cache_max_size_gb == 0 {
        return Err(AstronomicalConfigError::InvalidPromptCacheMaxSizeGb {
            description: "prompt-cache max size must be positive",
        });
    }
    prompt_cache_max_size_gb
        .checked_mul(BYTES_PER_CONFIGURED_GIGABYTE)
        .ok_or(AstronomicalConfigError::InvalidPromptCacheMaxSizeGb {
            description: "prompt-cache max size exceeds the byte range",
        })
}
