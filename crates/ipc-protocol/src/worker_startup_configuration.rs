use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::WorkerLoadedModelRuntimeConfiguration;

/// Logging verbosity supplied by the supervisor when starting a worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Resolved optional draft-assisted speculative-prefill settings supplied to the worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSpeculativePrefillConfiguration {
    pub enabled: bool,
    pub target_model_id: Option<String>,
    pub draft_model_id: Option<String>,
    pub draft_model_directory: Option<PathBuf>,
    pub minimum_prompt_tokens: u32,
    pub keep_percentage: u32,
    pub selection_chunk_token_count: u32,
    pub mandatory_trailing_token_count: u32,
    pub lookahead_token_count: u32,
    pub importance_pooling_kernel_token_count: u32,
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
    pub persistent_prompt_cache_enabled: bool,
    /// Effective global cache capacity, without disclosing its local directory.
    pub prompt_cache_maximum_size_bytes: u64,
    /// Present only after a swap binds one concrete model policy.
    pub loaded_model: Option<WorkerLoadedModelRuntimeConfiguration>,
}

/// Fully resolved worker-owned startup settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerStartupConfiguration {
    pub configuration_generation: String,
    pub global_prompt_cache_root_directory: PathBuf,
    pub global_prompt_cache_maximum_size_bytes: u64,
    pub persistent_prompt_cache_enabled: bool,
    pub configured_maximum_mlx_memory_bytes: Option<u64>,
    pub performance_attribution_enabled: bool,
    pub logging_directory: PathBuf,
    pub logging_level: WorkerLogLevel,
    pub retained_log_file_count: usize,
}
