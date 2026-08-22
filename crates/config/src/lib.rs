#![forbid(unsafe_code)]

use std::{net::SocketAddr, path::PathBuf};

mod astronomical_runtime_instance;
mod chunking_config;
mod config_document;
mod config_error;
mod config_file;
mod configuration_generation;
mod duplicate_key_json;
mod laguna_template_source;
mod legacy_config_migration;
mod logging_config;
mod maximum_mlx_memory;
mod model_discovery;
mod model_discovery_huggingface_cache;
mod model_identity;
mod prompt_cache_config;
mod resolved_model_config;
mod speculative_prefill_config;

pub use astronomical_runtime_instance::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
pub use chunking_config::{
    ChunkingConfig, ConfiguredChunkingFields,
    DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
    DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
    DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
    DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
    DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
};
pub use config_error::AstronomicalConfigError;
pub use laguna_template_source::{
    LagunaRootChatTemplateSelectionError, LagunaRootChatTemplateSource,
    LagunaStandaloneChatTemplateState, select_laguna_root_chat_template,
    validate_laguna_standalone_chat_template_role,
};
pub use logging_config::{LogLevel, LoggingConfig};
pub use maximum_mlx_memory::{
    MaximumMlxMemoryConfigUpdate, commit_maximum_mlx_memory_gb_update,
    maximum_mlx_memory_gb_to_bytes, prepare_maximum_mlx_memory_gb_update,
    write_maximum_mlx_memory_gb,
};
pub use model_discovery::{
    ChatModelCapabilities, ClassifiedModelArtifact, DiscoveredModel, DiscoveredModelError,
    Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError, ImageGenerationCapabilities,
    ModelCapabilities, ModelDiscoveryDiagnostic, ModelDiscoveryDirectoryScan, ModelDiscoveryReport,
    ModelFamily, ModelFamilyClassificationError, ModelLicense, classify_model_directory,
    discover_classified_model_artifacts, discover_models,
    discover_models_excluding_ambiguous_identities, requestable_model_id,
    verify_flux2_klein_model_directory,
};
pub use model_identity::{
    decode_huggingface_cache_directory_name, leaf_model_id, resolve_model_id,
};
pub use prompt_cache_config::PromptCacheConfig;
pub use resolved_model_config::{DEFAULT_MAXIMUM_OUTPUT_TOKENS, ResolvedModelConfig};
pub use speculative_prefill_config::SpeculativePrefillConfig;

use config_document::UserConfigFile;
use config_file::read_user_config_file;

pub const DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB: u64 = 50;
const BYTES_PER_CONFIGURED_GIGABYTE: u64 = 1_000_000_000;
const DEFAULT_RETAINED_LOG_FILES: usize = 7;

/// User-level Astronomical runtime configuration loaded inside one isolated instance.
#[derive(Clone, Debug)]
pub struct AstronomicalConfig {
    instance_paths: AstronomicalInstancePaths,
    user_config_file: UserConfigFile,
    configuration_generation: String,
}

impl AstronomicalConfig {
    /// Resolves one already-captured strict v1 document without rereading a mutable file.
    pub fn load_v1_bytes(
        instance_paths: AstronomicalInstancePaths,
        config_file_bytes: &[u8],
    ) -> Result<Self, AstronomicalConfigError> {
        let config_file_path = instance_paths.config_file_path();
        let config_json = duplicate_key_json::parse_json_rejecting_duplicates(
            &config_file_path,
            config_file_bytes,
        )?;
        let user_config_file = config_file::parse_and_validate_v1(&config_file_path, config_json)?;
        let configuration_generation = configuration_generation::configuration_generation(
            &config_file_path,
            &user_config_file,
        )?;
        Ok(Self {
            instance_paths,
            user_config_file,
            configuration_generation,
        })
    }

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
        let configuration_generation = configuration_generation::configuration_generation(
            &instance_paths.config_file_path(),
            &user_config_file,
        )?;
        Ok(Self {
            instance_paths,
            user_config_file,
            configuration_generation,
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

    /// SHA-256 identity of the validated semantic v1 document.
    ///
    /// Canonical typed serialization deliberately makes whitespace and JSON
    /// member ordering irrelevant. The lowercase hexadecimal digest is bounded
    /// and cannot disclose configured local paths.
    #[must_use]
    pub fn generation(&self) -> &str {
        &self.configuration_generation
    }

    /// Returns absolute directories to recursively scan for supported models.
    #[must_use]
    pub fn model_directories(&self) -> &[PathBuf] {
        &self.user_config_file.runtime.model_directories
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
        let configured_model_directory_scans = discover_models(self.model_directories())?;
        Ok(configured_model_directory_scans
            .into_iter()
            .flat_map(|configured_model_directory_scan| {
                configured_model_directory_scan.discovered_models
            })
            .find(|discovered_model| discovered_model.model_id == model_id)
            .map(|discovered_model| discovered_model.model_directory))
    }

    /// Returns every resolved model-serving work partition from the nested chunking policy.
    pub fn chunking(&self) -> Result<ChunkingConfig, AstronomicalConfigError> {
        let default_chunking = Default::default();
        let configured_chunking = self
            .user_config_file
            .chunking
            .as_ref()
            .unwrap_or(&default_chunking);
        ChunkingConfig::resolve(configured_chunking)
    }

    /// Resolves and validates the loopback-only HTTP bind address.
    pub fn supervisor_bind_address(&self) -> Result<SocketAddr, AstronomicalConfigError> {
        let raw_bind_address = self.instance_paths.default_bind_address().to_string();
        parse_loopback_bind_address(&raw_bind_address)
    }

    /// Resolves the always-enabled SSD-backed prompt cache policy.
    pub fn prompt_cache(&self) -> Result<PromptCacheConfig, AstronomicalConfigError> {
        let global_prompt_cache_root_directory =
            self.default_global_prompt_cache_root_directory()?;
        let global_prompt_cache_maximum_size_bytes = prompt_cache_size_gb_to_bytes(
            self.user_config_file
                .prompt_cache
                .as_ref()
                .and_then(|prompt_cache| prompt_cache.maximum_size_gb)
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
            .prompt_cache
            .as_ref()
            .and_then(|prompt_cache| prompt_cache.enabled)
            .unwrap_or(true)
    }

    /// Returns the authored cache preference without replacing absence with policy.
    #[must_use]
    pub fn configured_persistent_prompt_cache_enabled(&self) -> Option<bool> {
        self.user_config_file
            .prompt_cache
            .as_ref()
            .and_then(|prompt_cache| prompt_cache.enabled)
    }

    /// Returns the authored decimal-SI cache capacity without applying its default.
    #[must_use]
    pub fn configured_prompt_cache_maximum_size_bytes(
        &self,
    ) -> Result<Option<u64>, AstronomicalConfigError> {
        self.user_config_file
            .prompt_cache
            .as_ref()
            .and_then(|prompt_cache| prompt_cache.maximum_size_gb)
            .map(prompt_cache_size_gb_to_bytes)
            .transpose()
    }

    /// Resolves bounded hourly file logging for the supervisor or worker.
    pub fn logging(&self) -> Result<LoggingConfig, AstronomicalConfigError> {
        let configured_diagnostics = self.user_config_file.diagnostics.as_ref();
        Ok(LoggingConfig::new(
            self.instance_paths.logging_directory(),
            configured_diagnostics
                .and_then(|diagnostics| diagnostics.log_level)
                .unwrap_or(LogLevel::Warn),
            configured_diagnostics
                .and_then(|diagnostics| diagnostics.retained_log_files)
                .unwrap_or(DEFAULT_RETAINED_LOG_FILES),
        ))
    }

    fn default_global_prompt_cache_root_directory(
        &self,
    ) -> Result<PathBuf, AstronomicalConfigError> {
        Ok(self.instance_paths.prompt_cache_directory())
    }

    /// Returns whether detailed critical-path performance attribution is enabled.
    ///
    /// Attribution defaults to disabled so normal inference avoids timing and
    /// serialization work unless the user explicitly requests diagnostics.
    #[must_use]
    pub fn performance_attribution_enabled(&self) -> bool {
        self.user_config_file
            .diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.performance_attribution_enabled)
            .unwrap_or(false)
    }

    /// Resolves inherited policy for one canonical discovered model identity.
    pub fn resolved_model_config(
        &self,
        model_id: &str,
        artifact_maximum_context_tokens: u32,
    ) -> Result<ResolvedModelConfig, AstronomicalConfigError> {
        let global_chunking = self.user_config_file.chunking.clone().unwrap_or_default();
        ResolvedModelConfig::resolve(
            model_id,
            artifact_maximum_context_tokens,
            &global_chunking,
            self.user_config_file.models.get(model_id),
        )
    }

    /// Returns configured canonical model identities for catalog-level validation.
    #[must_use]
    pub fn configured_model_ids(&self) -> Vec<&str> {
        self.user_config_file
            .models
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Resolves the optional user MLX memory ceiling in decimal SI bytes.
    ///
    /// `None` means that startup or live runtime control should use the machine maximum.
    pub fn maximum_mlx_memory_bytes(&self) -> Result<Option<u64>, AstronomicalConfigError> {
        self.user_config_file
            .runtime
            .maximum_mlx_memory_gb
            .map(maximum_mlx_memory_gb_to_bytes)
            .transpose()
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
