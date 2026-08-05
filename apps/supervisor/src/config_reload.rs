//! Config-reload diff logic for `POST /v1/config/reload`.
//!
//! This module classifies the difference between the currently resolved
//! runtime config and a candidate config into one of three decisions:
//! no worker restart, worker restart, or full REST API restart.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, DiscoveredModel, DiscoveredModelError, LogLevel,
    LoggingConfig, PrefillChunckSizingPolicy, PromptCacheConfig, discover_qwen3_5_models,
};
use astronomical_ipc_protocol::{
    WorkerLogLevel, WorkerPrefillChunckSizingPolicy, WorkerStartupConfiguration,
};
use thiserror::Error;

const IGNORED_FIXED_PREFILL_CHUNCK_TOKENS_WARNING: &str = "found fixed_prefill_chunck_tokens defined while prefill_chunck_size_optimizer_enabled = true, will ignore fixed_prefill_chunck_token value";

/// Immutable snapshot of every resolved runtime value the supervisor needs
/// to decide whether a config reload requires a worker restart, a full REST
/// API restart, or only an in-place update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRuntimeConfig {
    /// Resolved worker executable path used to spawn or replace the worker.
    pub worker_executable_path: PathBuf,
    /// All Qwen3.5-MoE-family models discovered from the resolved directories.
    pub discovered_models: Vec<DiscoveredModel>,
    /// Configured discovery roots, including roots that currently contain no model.
    pub configured_model_directories: Vec<PathBuf>,
    /// Named model directories used for recursive discovery and worker hot-swap.
    pub model_directories: Arc<HashMap<String, PathBuf>>,
    /// Per-request output-token ceiling.
    pub max_output_tokens: u32,
    /// Optional user-configured MLX memory ceiling in exact decimal SI bytes.
    pub maximum_mlx_memory_bytes: Option<u64>,
    /// Startup config warning text surfaced through `/v1/status`.
    pub config_warning: Option<String>,
    /// Resolved prefill chunk sizing policy read by the worker at startup.
    pub prefill_chunck_sizing_policy: PrefillChunckSizingPolicy,
    /// Persistent prefill optimizer state resolved from the daemon config location.
    pub optimizer_state_directory: PathBuf,
    /// Performance attribution preference captured by the worker at startup.
    pub performance_attribution_enabled: bool,
    /// Whether the worker may read and write the persistent prompt cache.
    pub persistent_prompt_cache_enabled: bool,
    /// Explicit user choice surfaced while Qwen MTP runtime support is parked.
    pub mtp_enabled: bool,
    /// Resolved SSD-backed prompt-cache policy (worker-startup field).
    pub prompt_cache_config: PromptCacheConfig,
    /// Resolved supervisor bind address (REST API restart required to change).
    pub bind_address: String,
    /// Resolved logging policy (REST API restart required to change).
    pub logging_config: LoggingConfig,
}

/// Resolves startup and reload config from one user configuration directory.
#[derive(Clone, Debug)]
pub struct ResolvedRuntimeConfigResolver {
    config_home_directory: PathBuf,
    fallback_worker_executable_path: PathBuf,
}

impl ResolvedRuntimeConfigResolver {
    #[must_use]
    pub const fn new(
        config_home_directory: PathBuf,
        fallback_worker_executable_path: PathBuf,
    ) -> Self {
        Self {
            config_home_directory,
            fallback_worker_executable_path,
        }
    }

    #[must_use]
    pub fn config_home_directory(&self) -> &std::path::Path {
        &self.config_home_directory
    }

    /// Loads and resolves the current config file.
    pub fn load(&self) -> Result<ResolvedRuntimeConfig, ResolvedRuntimeConfigError> {
        let user_config =
            AstronomicalConfig::load_from_home_directory(&self.config_home_directory)?;
        self.resolve(&user_config)
    }

    /// Resolves one already-loaded config using startup-equivalent precedence.
    pub fn resolve(
        &self,
        user_config: &AstronomicalConfig,
    ) -> Result<ResolvedRuntimeConfig, ResolvedRuntimeConfigError> {
        let supervisor_bind_address = user_config.supervisor_bind_address()?;
        let configured_model_directories = user_config.model_directories().to_vec();
        let max_output_tokens = user_config.max_output_tokens();
        let discovered_models =
            discover_qwen3_5_models(&configured_model_directories, max_output_tokens)?
                .into_iter()
                .flat_map(|directory_scan| directory_scan.discovered_models)
                .collect::<Vec<_>>();
        let model_directories = Arc::new(
            discovered_models
                .iter()
                .map(|discovered_model| {
                    (
                        discovered_model.model_id.clone(),
                        discovered_model.model_directory.clone(),
                    )
                })
                .collect(),
        );
        let prompt_cache_config = user_config.prompt_cache()?;
        let logging_config = user_config.logging()?;

        Ok(ResolvedRuntimeConfig {
            worker_executable_path: self.fallback_worker_executable_path.clone(),
            discovered_models,
            configured_model_directories,
            model_directories,
            max_output_tokens,
            maximum_mlx_memory_bytes: user_config.maximum_mlx_memory_bytes()?,
            config_warning: user_config
                .ignored_fixed_prefill_chunck_tokens()
                .map(|_| IGNORED_FIXED_PREFILL_CHUNCK_TOKENS_WARNING.to_owned()),
            prefill_chunck_sizing_policy: user_config.prefill_chunck_sizing_policy()?,
            optimizer_state_directory: user_config.optimizer_directory()?,
            performance_attribution_enabled: user_config.performance_attribution_enabled(),
            persistent_prompt_cache_enabled: user_config.persistent_prompt_cache_enabled(),
            mtp_enabled: user_config.mtp_enabled(),
            prompt_cache_config,
            bind_address: supervisor_bind_address.to_string(),
            logging_config,
        })
    }
}

impl ResolvedRuntimeConfig {
    /// Converts supervisor-resolved worker settings into the IPC bootstrap DTO.
    #[must_use]
    pub fn worker_startup_configuration(&self) -> WorkerStartupConfiguration {
        WorkerStartupConfiguration {
            global_prompt_cache_root_directory: self
                .prompt_cache_config
                .global_prompt_cache_root_directory()
                .clone(),
            global_prompt_cache_maximum_size_bytes: self
                .prompt_cache_config
                .global_prompt_cache_maximum_size_bytes(),
            persistent_prompt_cache_enabled: self.persistent_prompt_cache_enabled,
            prefill_chunck_sizing_policy: match self.prefill_chunck_sizing_policy {
                PrefillChunckSizingPolicy::Optimized => WorkerPrefillChunckSizingPolicy::Optimized,
                PrefillChunckSizingPolicy::Fixed {
                    fixed_prefill_chunck_tokens,
                } => WorkerPrefillChunckSizingPolicy::Fixed {
                    fixed_prefill_chunck_tokens,
                },
            },
            optimizer_state_directory: Some(self.optimizer_state_directory.clone()),
            configured_maximum_mlx_memory_bytes: self.maximum_mlx_memory_bytes,
            mtp_enabled: self.mtp_enabled,
            performance_attribution_enabled: self.performance_attribution_enabled,
            logging_directory: self.logging_config.directory().to_path_buf(),
            logging_level: match self.logging_config.level() {
                LogLevel::Error => WorkerLogLevel::Error,
                LogLevel::Warn => WorkerLogLevel::Warn,
                LogLevel::Info => WorkerLogLevel::Info,
                LogLevel::Debug => WorkerLogLevel::Debug,
                LogLevel::Trace => WorkerLogLevel::Trace,
            },
            retained_log_file_count: self.logging_config.retained_files(),
        }
    }
}

/// Failure while loading or resolving runtime config for startup/reload.
#[derive(Debug, Error)]
pub enum ResolvedRuntimeConfigError {
    #[error("invalid Astronomical configuration")]
    Configuration(#[from] AstronomicalConfigError),
    #[error("failed to discover configured models")]
    ModelDiscovery(#[from] DiscoveredModelError),
}

/// Result of comparing the current resolved config with a candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigReloadDecision {
    /// All changed fields were applied in-place without restarting the worker.
    NoWorkerRestart {
        reloaded_fields: Vec<String>,
        discovered_model_count: usize,
    },
    /// At least one worker-startup field changed; the worker must be replaced.
    RestartWorker {
        reloaded_fields: Vec<String>,
        discovered_model_count: usize,
    },
    /// At least one REST-API-only field changed; a full restart is required.
    RestApiRestartRequired {
        reloaded_fields: Vec<String>,
        restart_required_fields: Vec<String>,
        discovered_model_count: usize,
    },
}

impl ConfigReloadDecision {
    #[must_use]
    pub fn discovered_model_count(&self) -> usize {
        match self {
            Self::NoWorkerRestart {
                discovered_model_count,
                ..
            }
            | Self::RestartWorker {
                discovered_model_count,
                ..
            }
            | Self::RestApiRestartRequired {
                discovered_model_count,
                ..
            } => *discovered_model_count,
        }
    }

    #[must_use]
    pub fn reloaded_fields(&self) -> &[String] {
        match self {
            Self::NoWorkerRestart {
                reloaded_fields, ..
            }
            | Self::RestartWorker {
                reloaded_fields, ..
            }
            | Self::RestApiRestartRequired {
                reloaded_fields, ..
            } => reloaded_fields,
        }
    }

    #[must_use]
    pub fn worker_restart_required(&self) -> bool {
        matches!(self, Self::RestartWorker { .. })
    }
}

/// Pure comparison between two resolved configs.
pub struct ConfigReloadDiff;

impl ConfigReloadDiff {
    /// Compares `current` against `candidate` and returns the reload decision.
    #[must_use]
    pub fn compare(
        current: &ResolvedRuntimeConfig,
        candidate: &ResolvedRuntimeConfig,
    ) -> ConfigReloadDecision {
        let mut in_place_reloaded_fields = Vec::new();
        let mut worker_restart_reloaded_fields = Vec::new();
        let mut restart_required_fields = Vec::new();
        let mut worker_restart_required = false;

        if current.config_warning != candidate.config_warning {
            in_place_reloaded_fields.push("config_warning".to_owned());
        }
        if current.configured_model_directories != candidate.configured_model_directories
            || current.model_directories != candidate.model_directories
        {
            worker_restart_reloaded_fields.push("model_directories".to_owned());
            worker_restart_required = true;
        }
        if current.max_output_tokens != candidate.max_output_tokens {
            worker_restart_reloaded_fields.push("max_output_tokens".to_owned());
            worker_restart_required = true;
        }
        if current.maximum_mlx_memory_bytes != candidate.maximum_mlx_memory_bytes {
            in_place_reloaded_fields.push("maximum_mlx_memory_gb".to_owned());
        }
        if current.prefill_chunck_sizing_policy != candidate.prefill_chunck_sizing_policy {
            worker_restart_reloaded_fields.push("prefill_chunck_sizing_policy".to_owned());
            worker_restart_required = true;
        }
        if current.optimizer_state_directory != candidate.optimizer_state_directory {
            worker_restart_reloaded_fields.push("optimizer_state_directory".to_owned());
            worker_restart_required = true;
        }
        if current.performance_attribution_enabled != candidate.performance_attribution_enabled {
            worker_restart_reloaded_fields.push("performance_attribution_enabled".to_owned());
            worker_restart_required = true;
        }
        if current.persistent_prompt_cache_enabled != candidate.persistent_prompt_cache_enabled {
            worker_restart_reloaded_fields.push("persistent_prompt_cache_enabled".to_owned());
            worker_restart_required = true;
        }
        if current.mtp_enabled != candidate.mtp_enabled {
            worker_restart_reloaded_fields.push("mtp_enabled".to_owned());
            worker_restart_required = true;
        }
        if current.prompt_cache_config != candidate.prompt_cache_config {
            worker_restart_reloaded_fields.push("prompt_cache".to_owned());
            worker_restart_required = true;
        }
        if current.bind_address != candidate.bind_address {
            restart_required_fields.push("supervisor.bind_address".to_owned());
        }
        if current.logging_config != candidate.logging_config {
            restart_required_fields.push("logging".to_owned());
        }

        let discovered_model_count = candidate.discovered_models.len();

        if !restart_required_fields.is_empty() {
            return ConfigReloadDecision::RestApiRestartRequired {
                reloaded_fields: in_place_reloaded_fields,
                restart_required_fields,
                discovered_model_count,
            };
        }
        if worker_restart_required {
            in_place_reloaded_fields.extend(worker_restart_reloaded_fields);
            return ConfigReloadDecision::RestartWorker {
                reloaded_fields: in_place_reloaded_fields,
                discovered_model_count,
            };
        }
        ConfigReloadDecision::NoWorkerRestart {
            reloaded_fields: in_place_reloaded_fields,
            discovered_model_count,
        }
    }
}
