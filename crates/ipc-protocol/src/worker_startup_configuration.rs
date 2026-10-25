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

/// Internal decode-cache arm selected only by controlled benchmarks.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerDecodeExpertCachePolicy {
    #[default]
    Baseline,
    Candidate,
}

/// Qwen3.5 engine family forced by controlled diagnostic journeys only.
///
/// Startup normally picks the resident or the SSD-streaming engine from the
/// complete-residency fit at the configured ceiling, which couples two
/// variables a throughput investigation must separate: the memory ceiling and
/// the engine architecture. Forcing the engine lets one journey run the
/// streaming engine under a roomy ceiling (or the resident engine under a
/// tight one) so each variable's cost is measured on its own. The startup
/// placement is still computed and recorded unchanged.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerDiagnosticQwen3_5EngineOverride {
    #[default]
    Automatic,
    Resident,
    Streaming,
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
    /// Benchmark-only policy selector; never read from the user configuration.
    #[serde(default)]
    pub decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    /// Diagnostic-only engine override; never read from the user configuration.
    #[serde(default)]
    pub diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride,
    /// Diagnostic-only replacement for the resident Qwen3.5 engine's static
    /// three-layer prefill activation floor; `None` keeps the production floor.
    /// Never read from the user configuration.
    #[serde(default)]
    pub diagnostic_qwen3_5_resident_prefill_activation_floor_bytes: Option<u64>,
    pub logging_directory: PathBuf,
    pub logging_level: WorkerLogLevel,
    pub retained_log_file_count: usize,
}
