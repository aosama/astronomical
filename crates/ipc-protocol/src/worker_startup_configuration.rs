use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::WorkerChunkingConfiguration;

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
    pub selection_chunck_token_count: u32,
    pub mandatory_trailing_token_count: u32,
    pub lookahead_token_count: u32,
    pub importance_pooling_kernel_token_count: u32,
}

impl WorkerSpeculativePrefillConfiguration {
    /// Returns this policy enabled only when the loaded model is its configured target.
    pub fn for_loaded_model(&self, loaded_model_id: &str) -> Self {
        let mut loaded_model_speculative_prefill_configuration = self.clone();
        loaded_model_speculative_prefill_configuration.enabled =
            self.enabled && self.target_model_id.as_deref() == Some(loaded_model_id);
        loaded_model_speculative_prefill_configuration
    }
}

/// Worker-acknowledged feature settings safe to expose through local status.
///
/// This intentionally excludes startup paths and model locations. It proves the
/// effective policy of the worker process that will serve requests.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerRuntimeFeatureConfiguration {
    /// Whether the worker will persist and restore ordinary prompt state.
    pub persistent_prompt_cache_enabled: bool,
    /// Whether the worker may activate multi-token prediction for a compatible model.
    pub mtp_enabled: bool,
    /// Explicit user depth, or `None` for the artifact default and then depth one.
    pub mtp_draft_depth: Option<u8>,
    /// Whether the currently bound target model may execute draft-assisted prefill.
    pub speculative_prefill_enabled: bool,
}

/// Fully resolved worker-owned startup settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerStartupConfiguration {
    pub global_prompt_cache_root_directory: PathBuf,
    pub global_prompt_cache_maximum_size_bytes: u64,
    pub persistent_prompt_cache_enabled: bool,
    pub chunking: WorkerChunkingConfiguration,
    pub optimizer_state_directory: Option<PathBuf>,
    pub configured_maximum_mlx_memory_bytes: Option<u64>,
    pub mtp_enabled: bool,
    pub mtp_draft_depth: Option<u8>,
    pub speculative_prefill: WorkerSpeculativePrefillConfiguration,
    pub performance_attribution_enabled: bool,
    pub logging_directory: PathBuf,
    pub logging_level: WorkerLogLevel,
    pub retained_log_file_count: usize,
}
