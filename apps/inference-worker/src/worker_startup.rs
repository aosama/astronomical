//! Worker process bootstrap: read the daemon's InitializeWorker command, wire
//! up diagnostics and the machine memory ceiling, claim Metal wired residency,
//! and hand an idle engine to the daemon-controlled decode loop.
//!
//! Each sub-concern lives alongside this orchestrator in the `worker_startup/`
//! folder: the startup and runtime error types (`error`), the machine MLX
//! memory-ceiling sampling (`gpu_memory`), rolling diagnostics (`logging`), and
//! the Metal wired-residency claim (`wired_claim`).

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
use astronomical_model_serving::effective_mlx_memory_ceiling_after_wired_claim;
use astronomical_model_serving::effective_mlx_memory_ceiling_bytes;
use tokio::io::{AsyncRead, AsyncWrite};

pub(crate) mod error;
mod gpu_memory;
mod logging;
mod wired_claim;

#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod gpu_memory_tests;
#[cfg(test)]
mod wired_claim_tests;

use crate::model_family_factory::{InferenceWorker, ModelFamilyFactory};

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";

pub use crate::worker_startup::error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup::gpu_memory::machine_mlx_memory_ceiling_bytes;
pub use crate::worker_startup::logging::initialize_tracing;

/// Starts an idle worker that loads a supported model only after an explicit
/// SwapModel command from the daemon.
pub async fn run_bootstrapped_worker<ReadTransport, WriteTransport>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    let mut command_reader = ProtocolReader::new(read_transport);
    let event_writer = ProtocolWriter::new(write_transport);
    let Some(WorkerCommand::InitializeWorker(worker_startup_configuration)) =
        command_reader.next_command().await?
    else {
        return Err(WorkerProcessError::Startup(
            WorkerStartupError::InitializeTracing {
                description: "expected InitializeWorker as the first worker command".to_owned(),
            },
        ));
    };
    let _worker_logging_guard =
        initialize_tracing(&worker_startup_configuration).map_err(|source| {
            WorkerProcessError::Startup(WorkerStartupError::InitializeTracing {
                description: source.to_string(),
            })
        })?;
    run_initialized_worker(worker_startup_configuration, command_reader, event_writer).await
}

async fn run_initialized_worker<ReadTransport, WriteTransport>(
    worker_startup_configuration: WorkerStartupConfiguration,
    command_reader: ProtocolReader<ReadTransport>,
    event_writer: ProtocolWriter<WriteTransport>,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    let prompt_cache_config = PromptCacheConfig::new(
        worker_startup_configuration
            .global_prompt_cache_root_directory
            .clone(),
        worker_startup_configuration.global_prompt_cache_maximum_size_bytes,
    );
    let performance_attribution_enabled =
        worker_startup_configuration.performance_attribution_enabled;
    let performance_attribution_log_path = worker_startup_configuration
        .logging_directory
        .join(PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME);
    let persistent_prompt_cache_enabled =
        worker_startup_configuration.persistent_prompt_cache_enabled;
    let worker_runtime_feature_configuration = WorkerRuntimeFeatureConfiguration {
        configuration_generation: worker_startup_configuration
            .configuration_generation
            .clone(),
        persistent_prompt_cache_enabled,
        prompt_cache_maximum_size_bytes: worker_startup_configuration
            .global_prompt_cache_maximum_size_bytes,
        loaded_model: None,
    };
    let machine_mlx_memory_ceiling_bytes = machine_mlx_memory_ceiling_bytes()
        .await
        .map_err(WorkerProcessError::Startup)?;
    let machine_mlx_memory_ceiling_bytes_u64 = u64::try_from(machine_mlx_memory_ceiling_bytes)
        .map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidPhysicalMemorySize {
                description: "machine MLX memory ceiling exceeds the u64 range",
            })
        })?;
    let configured_maximum_mlx_memory_bytes =
        worker_startup_configuration.configured_maximum_mlx_memory_bytes;
    let effective_mlx_memory_ceiling_bytes_u64 = effective_mlx_memory_ceiling_bytes(
        configured_maximum_mlx_memory_bytes,
        machine_mlx_memory_ceiling_bytes_u64,
    );
    if let Some(configured_maximum_mlx_memory_bytes) = configured_maximum_mlx_memory_bytes
        && configured_maximum_mlx_memory_bytes > effective_mlx_memory_ceiling_bytes_u64
    {
        tracing::warn!(
            configured_maximum_mlx_memory_bytes,
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes_u64,
            "configured MLX memory ceiling exceeds the machine bound and was clamped to the effective ceiling"
        );
    }
    let claimed_wired_residency_budget_bytes =
        wired_claim::claim_wired_residency_up_to(machine_mlx_memory_ceiling_bytes);
    let effective_ceiling_before_wired_claim_bytes = effective_mlx_memory_ceiling_bytes_u64;
    let effective_mlx_memory_ceiling_bytes_u64 = effective_mlx_memory_ceiling_after_wired_claim(
        effective_mlx_memory_ceiling_bytes_u64,
        claimed_wired_residency_budget_bytes
            .and_then(|budget_bytes| u64::try_from(budget_bytes).ok()),
    );
    if effective_mlx_memory_ceiling_bytes_u64 < effective_ceiling_before_wired_claim_bytes {
        tracing::warn!(
            effective_mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes_u64,
            claimed_wired_residency_budget_bytes,
            machine_mlx_memory_ceiling_bytes,
            "serving ceiling clamped to the plannable wired residency (wired budget minus the Metal working-set margin)"
        );
    }
    let effective_mlx_memory_ceiling_bytes =
        usize::try_from(effective_mlx_memory_ceiling_bytes_u64).map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidPhysicalMemorySize {
                description: "effective MLX memory ceiling exceeds the platform integer range",
            })
        })?;
    tracing::info!(
        global_prompt_cache_maximum_size_bytes = ?prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
        performance_attribution_enabled,
        decode_expert_cache_policy = ?worker_startup_configuration.decode_expert_cache_policy,
        diagnostic_qwen3_5_engine_override = ?worker_startup_configuration.diagnostic_qwen3_5_engine_override,
        diagnostic_qwen3_5_resident_prefill_activation_floor_bytes = ?worker_startup_configuration.diagnostic_qwen3_5_resident_prefill_activation_floor_bytes,
        machine_mlx_memory_ceiling_bytes,
        configured_maximum_mlx_memory_bytes,
        effective_mlx_memory_ceiling_bytes,
        persistent_prompt_cache_enabled,
        configuration_generation = %worker_startup_configuration.configuration_generation,
        "starting idle inference worker"
    );
    let model_factory = ModelFamilyFactory::new(
        effective_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        prompt_cache_config,
        performance_attribution_enabled,
        worker_startup_configuration.decode_expert_cache_policy,
        performance_attribution_log_path,
        persistent_prompt_cache_enabled,
    )
    .with_diagnostic_qwen3_5_engine_override(
        worker_startup_configuration.diagnostic_qwen3_5_engine_override,
    )
    .with_diagnostic_qwen3_5_resident_prefill_activation_floor_bytes(
        worker_startup_configuration.diagnostic_qwen3_5_resident_prefill_activation_floor_bytes,
    );
    let engine_worker: InferenceWorker =
        InferenceWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
            model_factory,
            machine_mlx_memory_ceiling_bytes_u64,
            effective_mlx_memory_ceiling_bytes_u64,
        )
        .with_wired_residency_budget_bytes(
            claimed_wired_residency_budget_bytes
                .and_then(|budget_bytes| u64::try_from(budget_bytes).ok()),
        )
        .with_worker_runtime_feature_configuration(worker_runtime_feature_configuration);
    engine_worker
        .run_with_protocol(command_reader, event_writer)
        .await
        .map_err(WorkerProcessError::Runtime)
}
