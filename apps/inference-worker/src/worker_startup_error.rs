use std::io;
use std::path::PathBuf;

use astronomical_ipc_protocol::ProtocolError;
use astronomical_model_serving::{
    Qwen3_5MoEArtifactValidationError, Qwen3_5MoEPrefillChunckSizerError, Qwen3_5MoETokenizerError,
    WorkerRuntimeError,
};
use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

const MAXIMUM_PUBLIC_MODEL_LOAD_FAILURE_REASON_CHARACTER_COUNT: usize = 512;

/// Failure while starting or running the configured worker.
#[derive(Debug, Error)]
pub enum WorkerProcessError {
    #[error("worker bootstrap failed: {0}")]
    Bootstrap(#[from] ProtocolError),
    #[error("worker startup failed: {0}")]
    Startup(#[source] WorkerStartupError),
    #[error("worker runtime failed: {0}")]
    Runtime(#[source] WorkerRuntimeError),
}

#[derive(Debug, Error)]
pub enum WorkerStartupError {
    #[error("failed to initialize worker tracing: {description}")]
    InitializeTracing { description: String },
    #[error("invalid iogpu.wired_limit_mb value: {description}")]
    InvalidGpuWiredMemoryLimit { description: &'static str },
    #[error("failed to sample iogpu.wired_limit_mb")]
    SampleGpuWiredMemoryLimit(#[source] io::Error),
    #[error("sampling iogpu.wired_limit_mb timed out")]
    GpuWiredMemoryLimitSampleTimedOut,
    #[error("sysctl could not read iogpu.wired_limit_mb")]
    GpuWiredMemoryLimitSampleFailed,
    #[error("failed to read MLX recommended GPU working set")]
    ReadMlxRecommendedGpuWorkingSet(#[source] MlxRuntimeError),
    #[error("failed to validate Qwen3.5-MoE artifact at {model_directory:?}")]
    Qwen3_5MoEArtifactValidation {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5MoEArtifactValidationError,
    },
    #[error("failed to initialize Qwen3.5-MoE processor at {model_directory:?}")]
    Qwen3_5MoEProcessorInitialization {
        model_directory: PathBuf,
        #[source]
        source: Qwen3_5MoETokenizerError,
    },
    #[error("failed to open performance-attribution log at {performance_attribution_log_path:?}")]
    OpenPerformanceAttributionLog {
        performance_attribution_log_path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to configure Qwen3.5-MoE prompt-processing chunks")]
    PrefillChunckSizing(#[source] Qwen3_5MoEPrefillChunckSizerError),
    #[error("failed to start Qwen3.5-MoE engine at {model_directory:?}")]
    Qwen3_5MoEEngineInitialization {
        model_directory: PathBuf,
        #[source]
        source: astronomical_model_serving::InferenceEngineError,
    },
}

impl WorkerStartupError {
    /// Describes a model-load failure with safe detail but no local paths.
    #[must_use]
    pub fn public_model_load_failure_reason(&self) -> String {
        let unbounded_public_model_load_failure_reason = match self {
            Self::Qwen3_5MoEArtifactValidation {
                source: Qwen3_5MoEArtifactValidationError::OptiQMetadata(metadata_error),
                ..
            } => {
                format!("Qwen3.5-MoE OptiQ metadata validation failed: {metadata_error}")
            }
            Self::Qwen3_5MoEArtifactValidation {
                source: Qwen3_5MoEArtifactValidationError::Config(config_error),
                ..
            } => {
                format!("Qwen3.5-MoE config validation failed: {config_error}")
            }
            Self::Qwen3_5MoEArtifactValidation {
                source: Qwen3_5MoEArtifactValidationError::Qwen3_5MoEShardIndex(shard_index_error),
                ..
            } => {
                format!("Qwen3.5-MoE shard-index validation failed: {shard_index_error}")
            }
            Self::Qwen3_5MoEArtifactValidation { .. } => {
                "Qwen3.5-MoE artifact validation failed".to_owned()
            }
            Self::Qwen3_5MoEProcessorInitialization { .. } => {
                "Qwen3.5-MoE processor initialization failed".to_owned()
            }
            Self::Qwen3_5MoEEngineInitialization { .. } => {
                "Qwen3.5-MoE engine initialization failed".to_owned()
            }
            _ => "model initialization failed".to_owned(),
        };
        bound_public_model_load_failure_reason(unbounded_public_model_load_failure_reason)
    }
}

fn bound_public_model_load_failure_reason(
    unbounded_public_model_load_failure_reason: String,
) -> String {
    let mut public_model_load_failure_reason_character_indices =
        unbounded_public_model_load_failure_reason.char_indices();
    let Some((truncation_start_byte_index, _)) = public_model_load_failure_reason_character_indices
        .nth(MAXIMUM_PUBLIC_MODEL_LOAD_FAILURE_REASON_CHARACTER_COUNT - 1)
    else {
        return unbounded_public_model_load_failure_reason;
    };
    if public_model_load_failure_reason_character_indices
        .next()
        .is_none()
    {
        return unbounded_public_model_load_failure_reason;
    }
    format!(
        "{}…",
        &unbounded_public_model_load_failure_reason[..truncation_start_byte_index]
    )
}
