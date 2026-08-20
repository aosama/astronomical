//! Config-reload diff logic for `POST /v1/config/reload`.
//!
//! This module classifies the difference between the currently resolved
//! runtime config and a candidate config into one of three decisions:
//! no worker restart, worker restart, or full REST API restart.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, AstronomicalInstancePaths,
    AstronomicalRuntimeInstance, DiscoveredModel, DiscoveredModelError, LogLevel, LoggingConfig,
    ModelCapabilities, PromptCacheConfig, ResolvedModelConfig, discover_models,
};
use astronomical_ipc_protocol::{WorkerLogLevel, WorkerStartupConfiguration};
use thiserror::Error;

use crate::RuntimeModelPolicy;
use crate::resolved_model_policy_catalog::ResolvedModelPolicyCatalog;

/// Immutable snapshot of every resolved runtime value the supervisor needs
/// to decide whether a config reload requires a worker restart, a full REST
/// API restart, or only an in-place update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRuntimeConfig {
    /// Privacy-safe identity of the accepted semantic configuration document.
    pub configuration_generation: String,
    /// Resolved worker executable path used to spawn or replace the worker.
    pub worker_executable_path: PathBuf,
    /// All Qwen3.5-MoE-family models discovered from the resolved directories.
    pub discovered_models: Vec<DiscoveredModel>,
    /// Configured discovery roots, including roots that currently contain no model.
    pub configured_model_directories: Vec<PathBuf>,
    /// Canonical requestable identity to resolved directory and execution policy.
    pub model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    /// Configured preferences retained while their canonical target is absent.
    pub unmatched_model_config_ids: Vec<String>,
    /// Optional user-configured MLX memory ceiling in exact decimal SI bytes.
    pub maximum_mlx_memory_bytes: Option<u64>,
    /// Performance attribution preference captured by the worker at startup.
    pub performance_attribution_enabled: bool,
    /// Whether the worker may read and write the persistent prompt cache.
    pub persistent_prompt_cache_enabled: bool,
    /// Authored cache toggle before the enabled-by-default policy is applied.
    pub configured_persistent_prompt_cache_enabled: Option<bool>,
    /// Authored cache capacity before the 50 GB default is applied.
    pub configured_prompt_cache_maximum_size_bytes: Option<u64>,
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
    instance_paths: AstronomicalInstancePaths,
    fallback_worker_executable_path: PathBuf,
}

impl ResolvedRuntimeConfigResolver {
    #[must_use]
    /// Resolves temporary or user-selected Development-shaped test state.
    pub fn for_development_home_directory(
        development_home_directory: PathBuf,
        fallback_worker_executable_path: PathBuf,
    ) -> Self {
        Self {
            instance_paths: AstronomicalInstancePaths::for_home_directory(
                development_home_directory,
                AstronomicalRuntimeInstance::Development,
            ),
            fallback_worker_executable_path,
        }
    }

    #[must_use]
    pub const fn for_instance(
        instance_paths: AstronomicalInstancePaths,
        fallback_worker_executable_path: PathBuf,
    ) -> Self {
        Self {
            instance_paths,
            fallback_worker_executable_path,
        }
    }

    #[must_use]
    pub fn state_directory(&self) -> &std::path::Path {
        self.instance_paths.state_directory()
    }

    #[must_use]
    pub const fn instance_paths(&self) -> &AstronomicalInstancePaths {
        &self.instance_paths
    }

    /// Loads and resolves the current config file.
    pub fn load(&self) -> Result<ResolvedRuntimeConfig, ResolvedRuntimeConfigError> {
        let user_config =
            AstronomicalConfig::load_from_instance_paths(self.instance_paths.clone())?;
        self.resolve(&user_config)
    }

    /// Resolves one already-loaded config using startup-equivalent precedence.
    pub fn resolve(
        &self,
        user_config: &AstronomicalConfig,
    ) -> Result<ResolvedRuntimeConfig, ResolvedRuntimeConfigError> {
        let supervisor_bind_address = self
            .instance_paths
            .validate_configured_bind_address(user_config.supervisor_bind_address()?)?;
        let configured_model_directories = user_config.model_directories().to_vec();
        let mut discovered_models = discover_models(&configured_model_directories)?
            .into_iter()
            .flat_map(|directory_scan| directory_scan.discovered_models)
            .collect::<Vec<_>>();
        let discovered_model_ids = discovered_models
            .iter()
            .map(|discovered_model| discovered_model.model_id.clone())
            .collect::<std::collections::HashSet<_>>();
        let artifact_context_windows = discovered_models
            .iter()
            .filter_map(|model| match &model.capabilities {
                ModelCapabilities::Chat(capabilities) => {
                    Some((model.model_id.clone(), capabilities.context_window))
                }
                ModelCapabilities::ImageGeneration(_) => None,
            })
            .collect::<HashMap<_, _>>();
        let unmatched_model_config_ids = user_config
            .configured_model_ids()
            .into_iter()
            .filter(|configured_model_id| !discovered_model_ids.contains(*configured_model_id))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for discovered_model in &mut discovered_models {
            if let ModelCapabilities::Chat(capabilities) = &discovered_model.capabilities {
                let resolved_model_config = user_config.resolved_model_config(
                    &discovered_model.model_id,
                    capabilities.context_window,
                )?;
                apply_effective_model_limits(discovered_model, &resolved_model_config);
            }
        }
        let model_policy_catalog = ResolvedModelPolicyCatalog::resolve(
            user_config,
            &discovered_models,
            &artifact_context_windows,
        )?;
        let prompt_cache_config = user_config.prompt_cache()?;
        let logging_config = user_config.logging()?;
        let configuration_generation = crate::ResolvedConfigurationGeneration::derive(
            user_config.generation(),
            &discovered_models,
            &model_policy_catalog,
            &unmatched_model_config_ids,
        )?;
        Ok(ResolvedRuntimeConfig {
            configuration_generation,
            worker_executable_path: self.fallback_worker_executable_path.clone(),
            discovered_models,
            configured_model_directories,
            model_policy_catalog,
            unmatched_model_config_ids,
            maximum_mlx_memory_bytes: user_config.maximum_mlx_memory_bytes()?,
            performance_attribution_enabled: user_config.performance_attribution_enabled(),
            persistent_prompt_cache_enabled: user_config.persistent_prompt_cache_enabled(),
            configured_persistent_prompt_cache_enabled: user_config
                .configured_persistent_prompt_cache_enabled(),
            configured_prompt_cache_maximum_size_bytes: user_config
                .configured_prompt_cache_maximum_size_bytes()?,
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
            configuration_generation: self.configuration_generation.clone(),
            global_prompt_cache_root_directory: self
                .prompt_cache_config
                .global_prompt_cache_root_directory()
                .clone(),
            global_prompt_cache_maximum_size_bytes: self
                .prompt_cache_config
                .global_prompt_cache_maximum_size_bytes(),
            persistent_prompt_cache_enabled: self.persistent_prompt_cache_enabled,
            configured_maximum_mlx_memory_bytes: self.maximum_mlx_memory_bytes,
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
    #[error("invalid Astronomical configuration: {0}")]
    Configuration(#[from] AstronomicalConfigError),
    #[error("failed to discover configured models")]
    ModelDiscovery(#[from] DiscoveredModelError),
    #[error("failed to derive the resolved configuration generation")]
    ResolvedGeneration(#[from] serde_json::Error),
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

        if current.configured_model_directories != candidate.configured_model_directories {
            worker_restart_reloaded_fields.push("model_directories".to_owned());
            worker_restart_required = true;
        }
        if current.discovered_models != candidate.discovered_models {
            worker_restart_reloaded_fields.push("discovered_model_artifacts".to_owned());
            worker_restart_required = true;
        }
        if current.model_policy_catalog != candidate.model_policy_catalog {
            worker_restart_reloaded_fields.push("model_policies".to_owned());
            worker_restart_required = true;
        }
        if current.unmatched_model_config_ids != candidate.unmatched_model_config_ids {
            // Replacement keeps the worker acknowledgement aligned with the complete resolved
            // generation while the unmatched policy itself remains dormant and non-blocking.
            worker_restart_reloaded_fields.push("dormant_model_policies".to_owned());
            worker_restart_required = true;
        }
        if current.maximum_mlx_memory_bytes != candidate.maximum_mlx_memory_bytes {
            in_place_reloaded_fields.push("maximum_mlx_memory_gb".to_owned());
        }
        if current.performance_attribution_enabled != candidate.performance_attribution_enabled {
            worker_restart_reloaded_fields.push("performance_attribution_enabled".to_owned());
            worker_restart_required = true;
        }
        if current.persistent_prompt_cache_enabled != candidate.persistent_prompt_cache_enabled {
            worker_restart_reloaded_fields.push("persistent_prompt_cache_enabled".to_owned());
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
        if current.configuration_generation != candidate.configuration_generation
            && in_place_reloaded_fields.is_empty()
            && worker_restart_reloaded_fields.is_empty()
            && restart_required_fields.is_empty()
        {
            worker_restart_reloaded_fields.push("resolved_configuration".to_owned());
            worker_restart_required = true;
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

fn apply_effective_model_limits(
    discovered_model: &mut DiscoveredModel,
    resolved_model_config: &ResolvedModelConfig,
) {
    let ModelCapabilities::Chat(capabilities) = &mut discovered_model.capabilities else {
        return;
    };
    let effective_maximum_context_tokens = resolved_model_config
        .maximum_context_tokens()
        .unwrap_or(capabilities.context_window);
    capabilities.context_window = effective_maximum_context_tokens;
    // Input and output are independent maxima; request admission enforces their combined context.
    capabilities.max_input_tokens = effective_maximum_context_tokens.saturating_sub(1);
    capabilities.max_output_tokens =
        u32::from(u16::MAX).min(effective_maximum_context_tokens.saturating_sub(1));
}
