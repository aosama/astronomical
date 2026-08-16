#![forbid(unsafe_code)]

use std::{net::SocketAddr, path::PathBuf};

mod astronomical_runtime_instance;
mod chunking_config;
mod config_error;
mod config_file;
mod logging_config;
mod maximum_mlx_memory;
mod model_discovery;
mod model_discovery_huggingface_cache;
mod model_identity;
mod prompt_cache_config;
mod speculative_prefill_config;

pub use astronomical_runtime_instance::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
pub use chunking_config::{
    ChunkingConfig, DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
    DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
    DEFAULT_PROMPT_PROCESSING_CHUNK_SIZE_OPTIMIZER_MAXIMUM_RETAINED_MEASUREMENTS_PER_CANDIDATE_AND_CONTEXT,
    DEFAULT_PROMPT_PROCESSING_CHUNK_SIZE_OPTIMIZER_POSITION_RANGE_SIZE_TOKENS,
    DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
};
pub use config_error::AstronomicalConfigError;
pub use logging_config::{LogLevel, LoggingConfig};
pub use maximum_mlx_memory::{
    maximum_mlx_memory_gb_to_bytes, restore_config_file, write_maximum_mlx_memory_gb,
};
pub use model_discovery::{
    ClassifiedModelArtifact, DiscoveredModel, DiscoveredModelError, ModelDiscoveryDirectoryScan,
    ModelFamily, ModelFamilyClassificationError, classify_model_directory,
    discover_classified_model_artifacts, discover_models, requestable_model_id,
};
pub use model_identity::{
    decode_huggingface_cache_directory_name, leaf_model_id, resolve_model_id,
};
pub use prompt_cache_config::PromptCacheConfig;
pub use speculative_prefill_config::SpeculativePrefillConfig;

use config_file::{UserConfigFile, read_user_config_file};
use speculative_prefill_config::resolve_speculative_prefill_config;

pub const DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB: u64 = 50;
pub const DEFAULT_OPTIMIZER_PREFILL_CHUNCK_TOKEN_CANDIDATES: [u32; 4] =
    [1_024, 2_048, 4_096, 8_192];
const BYTES_PER_CONFIGURED_GIGABYTE: u64 = 1_000_000_000;
const DEFAULT_RETAINED_LOG_FILES: usize = 7;

/// User-level Astronomical runtime configuration loaded inside one isolated instance.
#[derive(Clone, Debug)]
pub struct AstronomicalConfig {
    instance_paths: AstronomicalInstancePaths,
    user_config_file: UserConfigFile,
}

impl AstronomicalConfig {
    /// Loads the Stable user configuration used by the installed application.
    pub fn load_from_default_location() -> Result<Self, AstronomicalConfigError> {
        Self::load_from_instance_paths(AstronomicalInstancePaths::for_current_user(
            AstronomicalRuntimeInstance::Stable,
        )?)
    }

    /// Loads the Development configuration used by direct repository workflows.
    pub fn load_from_development_location() -> Result<Self, AstronomicalConfigError> {
        Self::load_from_instance_paths(AstronomicalInstancePaths::for_current_user(
            AstronomicalRuntimeInstance::Development,
        )?)
    }

    /// Loads Development-shaped configuration beneath a supplied test home.
    pub fn load_from_development_home_directory(
        home_directory: impl Into<PathBuf>,
    ) -> Result<Self, AstronomicalConfigError> {
        Self::load_from_instance_paths(AstronomicalInstancePaths::for_home_directory(
            home_directory,
            AstronomicalRuntimeInstance::Development,
        ))
    }

    /// Loads strict configuration from an explicit, already-resolved instance boundary.
    pub fn load_from_instance_paths(
        instance_paths: AstronomicalInstancePaths,
    ) -> Result<Self, AstronomicalConfigError> {
        let user_config_file = read_user_config_file(instance_paths.config_file_path())?;
        Ok(Self {
            instance_paths,
            user_config_file,
        })
    }

    /// Loads a Stable-shaped config beneath a supplied home for qualification callers.
    pub fn load_from_home_directory(
        home_directory: impl Into<PathBuf>,
    ) -> Result<Self, AstronomicalConfigError> {
        Self::load_from_instance_paths(AstronomicalInstancePaths::for_home_directory(
            home_directory,
            AstronomicalRuntimeInstance::Stable,
        ))
    }

    #[must_use]
    pub const fn instance_paths(&self) -> &AstronomicalInstancePaths {
        &self.instance_paths
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
            discover_models(self.model_directories(), self.max_output_tokens())?;
        Ok(configured_model_directory_scans
            .into_iter()
            .flat_map(|configured_model_directory_scan| {
                configured_model_directory_scan.discovered_models
            })
            .find(|discovered_model| discovered_model.model_id == model_id)
            .map(|discovered_model| discovered_model.model_directory))
    }

    /// Resolves whether Qwen3.5-MoE uses adaptive or fixed prompt-processing chunks.
    pub fn prompt_processing_chunk_sizing_policy(
        &self,
    ) -> Result<PromptProcessingChunkSizingPolicy, AstronomicalConfigError> {
        Ok(self
            .chunking()?
            .prompt_processing_chunk_sizing_policy()
            .clone())
    }

    /// Returns the `chunking.fixed_prompt_processing_chunk_size_tokens` value the daemon ignored because
    /// `chunking.prompt_processing_chunk_size_optimizer_enabled` was `true`. The menu bar app flashes a
    /// callout when this is `Some` so the user knows the fixed value has no effect.
    #[must_use]
    pub fn ignored_fixed_prompt_processing_chunk_size_tokens(&self) -> Option<u32> {
        resolve_ignored_fixed_prompt_processing_chunk_size_tokens(
            self.user_config_file
                .chunking
                .prompt_processing_chunk_size_optimizer_enabled,
            self.user_config_file
                .chunking
                .fixed_prompt_processing_chunk_size_tokens,
        )
    }

    /// Returns every resolved model-serving work partition from the nested chunking policy.
    pub fn chunking(&self) -> Result<ChunkingConfig, AstronomicalConfigError> {
        ChunkingConfig::resolve(&self.user_config_file.chunking)
    }

    /// Resolves and validates the loopback-only HTTP bind address.
    pub fn supervisor_bind_address(&self) -> Result<SocketAddr, AstronomicalConfigError> {
        let raw_bind_address = self
            .user_config_file
            .supervisor
            .as_ref()
            .and_then(|supervisor_config| supervisor_config.bind_address.clone())
            .unwrap_or_else(|| self.instance_paths.default_bind_address().to_string());
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
        let configured_logging = self.user_config_file.logging.as_ref();
        Ok(LoggingConfig::new(
            self.instance_paths.logging_directory(),
            configured_logging.map_or(LogLevel::Warn, |logging| logging.level),
            configured_logging
                .and_then(|logging| logging.retained_files)
                .unwrap_or(DEFAULT_RETAINED_LOG_FILES),
        ))
    }

    fn default_global_prompt_cache_root_directory(
        &self,
    ) -> Result<PathBuf, AstronomicalConfigError> {
        Ok(self.instance_paths.prompt_cache_directory())
    }

    /// Resolves the optimizer state directory for persisting prefill chunk-size
    /// optimizer observations across restarts.
    ///
    /// Defaults to `~/.astronomical/optimizer/`. Always enabled; there is no
    /// config toggle to disable optimizer persistence.
    pub fn optimizer_directory(&self) -> Result<PathBuf, AstronomicalConfigError> {
        Ok(self.instance_paths.optimizer_directory())
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

    /// Returns the explicit fixed MTP draft depth, or `None` for artifact selection.
    #[must_use]
    pub fn mtp_draft_depth(&self) -> Option<u8> {
        self.user_config_file.mtp_draft_depth
    }

    /// Resolves the optional draft-assisted speculative-prefill policy.
    pub fn speculative_prefill(&self) -> Result<SpeculativePrefillConfig, AstronomicalConfigError> {
        resolve_speculative_prefill_config(&self.user_config_file.speculative_prefill)
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptProcessingChunkSizingPolicy {
    /// Select chunk sizes online from measured context-specific observations.
    Optimized {
        /// Strictly increasing token counts the optimizer may request.
        prompt_processing_chunk_size_optimizer_candidate_token_counts: Vec<u32>,
    },
    /// Process each normal full chunk with one configured token count.
    Fixed {
        /// Required positive token count for each fixed complete-resident prompt-processing chunk.
        fixed_prompt_processing_chunk_size_tokens: u32,
        /// Optional smaller fixed size used only while sparse experts stream from storage.
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    },
}

fn resolve_prompt_processing_chunk_sizing_policy(
    configured_prompt_processing_chunk_size_optimizer_enabled: Option<bool>,
    configured_fixed_prompt_processing_chunk_size_tokens: Option<u32>,
    configured_fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    configured_prompt_processing_chunk_size_optimizer_candidate_token_counts: Option<&[u32]>,
) -> Result<PromptProcessingChunkSizingPolicy, AstronomicalConfigError> {
    let fixed_prompt_processing_chunk_size_tokens = match configured_prompt_processing_chunk_size_optimizer_enabled {
        Some(false) => configured_fixed_prompt_processing_chunk_size_tokens.ok_or(
            AstronomicalConfigError::FixedPromptProcessingChunkSizeTokensRequiredWhenOptimizerDisabled,
        )?,
        Some(true) | None => {
            // Any configured fixed size is intentionally ignored in explicit
            // optimized mode. The menu bar surfaces that override separately.
            return Ok(PromptProcessingChunkSizingPolicy::Optimized {
                prompt_processing_chunk_size_optimizer_candidate_token_counts:
                    resolve_prompt_processing_chunk_size_optimizer_candidate_token_counts(
                        configured_prompt_processing_chunk_size_optimizer_candidate_token_counts,
                    )?,
            });
        }
    };
    if fixed_prompt_processing_chunk_size_tokens == 0 {
        return Err(AstronomicalConfigError::InvalidFixedPromptProcessingChunkSizeTokens);
    }
    let fixed_ssd_streaming_prompt_processing_chunk_size_tokens =
        match configured_fixed_ssd_streaming_prompt_processing_chunk_size_tokens {
            Some(0) => {
                return Err(AstronomicalConfigError::InvalidFixedSsdStreamingPromptProcessingChunkSizeTokens);
            }
            Some(fixed_ssd_streaming_prompt_processing_chunk_size_tokens) => {
                Some(fixed_ssd_streaming_prompt_processing_chunk_size_tokens)
            }
            None => None,
        };
    Ok(PromptProcessingChunkSizingPolicy::Fixed {
        fixed_prompt_processing_chunk_size_tokens,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
    })
}

fn resolve_prompt_processing_chunk_size_optimizer_candidate_token_counts(
    configured_prompt_processing_chunk_size_optimizer_candidate_token_counts: Option<&[u32]>,
) -> Result<Vec<u32>, AstronomicalConfigError> {
    let prompt_processing_chunk_size_optimizer_candidate_token_counts =
        configured_prompt_processing_chunk_size_optimizer_candidate_token_counts.map_or_else(
            || DEFAULT_OPTIMIZER_PREFILL_CHUNCK_TOKEN_CANDIDATES.to_vec(),
            <[u32]>::to_vec,
        );
    if prompt_processing_chunk_size_optimizer_candidate_token_counts.is_empty() {
        return Err(AstronomicalConfigError::OptimizerCandidateTokenCountsMustNotBeEmpty);
    }
    if prompt_processing_chunk_size_optimizer_candidate_token_counts.contains(&0) {
        return Err(AstronomicalConfigError::OptimizerCandidateTokenCountsMustBePositive);
    }
    if prompt_processing_chunk_size_optimizer_candidate_token_counts
        .windows(2)
        .any(|adjacent_candidates| adjacent_candidates[0] >= adjacent_candidates[1])
    {
        return Err(AstronomicalConfigError::OptimizerCandidateTokenCountsMustBeStrictlyIncreasing);
    }
    Ok(prompt_processing_chunk_size_optimizer_candidate_token_counts)
}

/// Returns the fixed prompt-processing chunk size the daemon ignores because the
/// optimizer is enabled, so the menu bar app can flash a callout warning the user.
fn resolve_ignored_fixed_prompt_processing_chunk_size_tokens(
    configured_prompt_processing_chunk_size_optimizer_enabled: Option<bool>,
    configured_fixed_prompt_processing_chunk_size_tokens: Option<u32>,
) -> Option<u32> {
    if configured_prompt_processing_chunk_size_optimizer_enabled == Some(true) {
        configured_fixed_prompt_processing_chunk_size_tokens
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
