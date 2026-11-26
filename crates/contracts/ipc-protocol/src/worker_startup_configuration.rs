use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::WorkerLoadedModelRuntimeConfiguration;

/// Logging verbosity supplied by the daemon when starting a worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Internal `astro_expert_cache` decode-policy arm selected only by controlled benchmarks.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerDecodeAstroExpertCachePolicy {
    #[default]
    Baseline,
    Candidate,
}

/// Worker-acknowledged feature settings safe to expose through local status.
///
/// This intentionally excludes startup paths and model locations. It proves the
/// effective policy of the worker process that will serve requests.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerRuntimeFeatureConfiguration {
    /// Semantic generation applied by this exact worker process.
    pub configuration_generation: String,
    /// Whether the worker will persist and restore ordinary prompt state.
    pub astro_ssd_prompt_cache_enabled: bool,
    /// Effective global `astro_ssd_prompt_cache` capacity, without disclosing its local directory.
    pub astro_ssd_prompt_cache_maximum_size_bytes: u64,
    /// Present only after a swap binds one concrete model policy.
    pub loaded_model: Option<WorkerLoadedModelRuntimeConfiguration>,
}

/// Fully resolved worker-owned startup settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerStartupConfiguration {
    pub configuration_generation: String,
    pub global_astro_ssd_prompt_cache_root_directory: PathBuf,
    pub global_astro_ssd_prompt_cache_maximum_size_bytes: u64,
    /// The declared home of compiled Neural Engine programs.
    pub neural_engine_program_directory: PathBuf,
    pub astro_ssd_prompt_cache_enabled: bool,
    pub configured_maximum_mlx_memory_bytes: Option<u64>,
    pub performance_attribution_enabled: bool,
    /// Benchmark-only policy selector; never read from the user configuration.
    #[serde(default)]
    pub decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    pub logging_directory: PathBuf,
    pub logging_level: WorkerLogLevel,
    pub retained_log_file_count: usize,
    /// Read by the daemon when it launches this worker, because MLX takes the
    /// setting from the environment when the Metal device is first created.
    pub metal_residency_refresh_enabled: bool,
}
