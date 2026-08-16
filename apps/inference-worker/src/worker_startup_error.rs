use std::io;

use astronomical_ipc_protocol::ProtocolError;
use astronomical_model_serving::WorkerRuntimeError;
use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

pub(crate) const MAXIMUM_PUBLIC_MODEL_LOAD_FAILURE_REASON_CHARACTER_COUNT: usize = 512;

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
    #[error("failed to sample iogpu.wired_limit_mb: {0}")]
    SampleGpuWiredMemoryLimit(#[source] io::Error),
    #[error("sampling iogpu.wired_limit_mb timed out")]
    GpuWiredMemoryLimitSampleTimedOut,
    #[error("sysctl could not read iogpu.wired_limit_mb")]
    GpuWiredMemoryLimitSampleFailed,
    #[error("failed to read MLX recommended GPU working set: {0}")]
    ReadMlxRecommendedGpuWorkingSet(#[source] MlxRuntimeError),
}

/// Bounds family-owned model failure detail before it crosses the worker boundary.
pub(crate) fn bound_public_model_load_failure_reason(
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
