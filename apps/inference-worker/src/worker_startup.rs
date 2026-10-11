use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
use astronomical_model_serving::{
    effective_mlx_serving_memory_ceiling_bytes, recommended_gpu_serving_memory_ceiling_bytes,
};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::model_family_factory::{InferenceWorker, ModelFamilyFactory};
use crate::worker_startup_gpu_memory::sample_gpu_memory_limits;

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";

pub use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup_gpu_memory::{
    GpuWiredMemoryLimitSetting, derive_mlx_memory_limits_from_gpu_wired_limit,
    parse_iogpu_wired_limit_setting, sample_iogpu_wired_limit_bytes,
};
pub use crate::worker_startup_logging::initialize_tracing;

/// Rejects a zero serving envelope before it reaches the process-wide MLX memory limit.
pub fn derive_serving_mlx_memory_ceiling_bytes(
    recommended_gpu_working_set_bytes: u64,
) -> Result<u64, WorkerStartupError> {
    let serving_mlx_memory_ceiling_bytes =
        recommended_gpu_serving_memory_ceiling_bytes(recommended_gpu_working_set_bytes);
    if serving_mlx_memory_ceiling_bytes == 0 {
        return Err(WorkerStartupError::RecommendedGpuWorkingSetTooSmall);
    }
    Ok(serving_mlx_memory_ceiling_bytes)
}

/// Starts an idle worker that loads a supported model only after an explicit
/// SwapModel command from the supervisor.
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
    let sampled_gpu_memory_limits = sample_gpu_memory_limits()
        .await
        .map_err(WorkerProcessError::Startup)?;
    let machine_mlx_memory_ceiling_bytes =
        sampled_gpu_memory_limits.machine_mlx_memory_ceiling_bytes;
    let machine_mlx_memory_ceiling_bytes_u64 = u64::try_from(machine_mlx_memory_ceiling_bytes)
        .map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidGpuWiredMemoryLimit {
                description: "machine MLX memory ceiling exceeds the u64 range",
            })
        })?;
    let recommended_gpu_working_set_bytes = u64::try_from(
        sampled_gpu_memory_limits.recommended_gpu_working_set_bytes,
    )
    .map_err(|_| {
        WorkerProcessError::Startup(WorkerStartupError::InvalidGpuWiredMemoryLimit {
            description: "recommended GPU working set exceeds the u64 range",
        })
    })?;
    let serving_mlx_memory_ceiling_bytes =
        derive_serving_mlx_memory_ceiling_bytes(recommended_gpu_working_set_bytes)
            .map_err(WorkerProcessError::Startup)?;
    let configured_maximum_mlx_memory_bytes =
        worker_startup_configuration.configured_maximum_mlx_memory_bytes;
    let effective_mlx_memory_ceiling_bytes_u64 = effective_mlx_serving_memory_ceiling_bytes(
        configured_maximum_mlx_memory_bytes,
        machine_mlx_memory_ceiling_bytes_u64,
        serving_mlx_memory_ceiling_bytes,
    );
    if let Some(configured_maximum_mlx_memory_bytes) = configured_maximum_mlx_memory_bytes
        && configured_maximum_mlx_memory_bytes > serving_mlx_memory_ceiling_bytes
    {
        tracing::warn!(
            configured_maximum_mlx_memory_bytes,
            serving_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes_u64,
            "configured MLX memory ceiling exceeds the serving envelope and was clamped"
        );
    }
    let effective_mlx_memory_ceiling_bytes =
        usize::try_from(effective_mlx_memory_ceiling_bytes_u64).map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidGpuWiredMemoryLimit {
                description: "effective MLX memory ceiling exceeds the platform integer range",
            })
        })?;
    tracing::info!(
        global_prompt_cache_maximum_size_bytes = ?prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
        performance_attribution_enabled,
        machine_mlx_memory_ceiling_bytes,
        recommended_gpu_working_set_bytes,
        serving_mlx_memory_ceiling_bytes,
        configured_maximum_mlx_memory_bytes,
        effective_mlx_memory_ceiling_bytes,
        persistent_prompt_cache_enabled,
        configuration_generation = %worker_startup_configuration.configuration_generation,
        "starting idle inference worker"
    );
    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        derive_mlx_memory_limits_from_gpu_wired_limit(effective_mlx_memory_ceiling_bytes);
    let model_factory = ModelFamilyFactory::new(
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        prompt_cache_config,
        performance_attribution_enabled,
        performance_attribution_log_path,
        persistent_prompt_cache_enabled,
    );
    let engine_worker: InferenceWorker =
        InferenceWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
            model_factory,
            machine_mlx_memory_ceiling_bytes_u64,
            effective_mlx_memory_ceiling_bytes_u64,
        )
        .with_serving_mlx_memory_ceiling(serving_mlx_memory_ceiling_bytes)
        .with_worker_runtime_feature_configuration(worker_runtime_feature_configuration);
    engine_worker
        .run_with_protocol(command_reader, event_writer)
        .await
        .map_err(WorkerProcessError::Runtime)
}
