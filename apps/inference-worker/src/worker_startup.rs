use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
use astronomical_model_serving::{
    EngineBackedWorker, ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine,
    Qwen3_5PromptProcessingChunkSizer,
};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::model_family_factory::ModelFamilyFactory;

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";

pub use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup_gpu_memory::{
    GpuWiredMemoryLimitSetting, derive_mlx_memory_limits_from_gpu_wired_limit,
    parse_iogpu_wired_limit_setting, resolve_effective_mlx_memory_ceiling_bytes,
    sample_iogpu_wired_limit_bytes,
};
pub use crate::worker_startup_logging::{astronomical_log_rotation, initialize_tracing};

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
    run_bootstrapped_worker_with_prompt_processing_chunk_sizer_override(
        read_transport,
        write_transport,
        None,
    )
    .await
}

async fn run_bootstrapped_worker_with_prompt_processing_chunk_sizer_override<
    ReadTransport,
    WriteTransport,
>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
    prompt_processing_chunk_sizer_override: Option<Qwen3_5PromptProcessingChunkSizer>,
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
    run_initialized_worker(
        worker_startup_configuration,
        command_reader,
        event_writer,
        prompt_processing_chunk_sizer_override,
    )
    .await
}

/// Runs the configured worker with explicit Qwen prompt-processing chunks for benchmarks.
#[cfg(feature = "performance-measurement")]
pub async fn run_configured_worker_with_prompt_processing_chunk_sizer_override<
    ReadTransport,
    WriteTransport,
>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
    prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    run_bootstrapped_worker_with_prompt_processing_chunk_sizer_override(
        read_transport,
        write_transport,
        Some(prompt_processing_chunk_sizer),
    )
    .await
}

async fn run_initialized_worker<ReadTransport, WriteTransport>(
    worker_startup_configuration: WorkerStartupConfiguration,
    command_reader: ProtocolReader<ReadTransport>,
    event_writer: ProtocolWriter<WriteTransport>,
    prompt_processing_chunk_sizer_override: Option<Qwen3_5PromptProcessingChunkSizer>,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    // Destructure the one authoritative chunking contract before constructing
    // model ownership. The sizing policy and every supporting boundary crossed
    // the same validated IPC boundary and therefore cannot disagree.
    let chunking = worker_startup_configuration.chunking;
    let prompt_cache_config = PromptCacheConfig::new(
        worker_startup_configuration
            .global_prompt_cache_root_directory
            .clone(),
        worker_startup_configuration.global_prompt_cache_maximum_size_bytes,
    );
    let optimizer_state_directory = worker_startup_configuration
        .optimizer_state_directory
        .clone();
    let performance_attribution_enabled =
        worker_startup_configuration.performance_attribution_enabled;
    let performance_attribution_log_path = worker_startup_configuration
        .logging_directory
        .join(PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME);
    let mtp_enabled = worker_startup_configuration.mtp_enabled;
    let persistent_prompt_cache_enabled =
        worker_startup_configuration.persistent_prompt_cache_enabled;
    let worker_runtime_feature_configuration = WorkerRuntimeFeatureConfiguration {
        persistent_prompt_cache_enabled,
        mtp_enabled,
        speculative_prefill_enabled: worker_startup_configuration.speculative_prefill.enabled,
    };
    let machine_mlx_memory_ceiling_bytes = sample_iogpu_wired_limit_bytes()
        .await
        .map_err(WorkerProcessError::Startup)?;
    let configured_maximum_mlx_memory_bytes =
        worker_startup_configuration.configured_maximum_mlx_memory_bytes;
    let effective_mlx_memory_ceiling_bytes = resolve_effective_mlx_memory_ceiling_bytes(
        configured_maximum_mlx_memory_bytes,
        machine_mlx_memory_ceiling_bytes,
    );
    tracing::info!(
        global_prompt_cache_maximum_size_bytes = ?prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
        prompt_processing_chunk_sizing_policy = ?chunking.prompt_processing_chunk_sizing_policy,
        optimizer_state_directory = ?optimizer_state_directory
            .as_ref()
            .map(|optimizer_state_directory| optimizer_state_directory.display()),
        performance_attribution_enabled,
        machine_mlx_memory_ceiling_bytes,
        configured_maximum_mlx_memory_bytes,
        effective_mlx_memory_ceiling_bytes,
        mtp_enabled,
        speculative_prefill = ?worker_startup_configuration.speculative_prefill,
        persistent_prompt_cache_enabled,
        chunking = ?chunking,
        "starting idle inference worker"
    );
    let model_factory = ModelFamilyFactory {
        effective_mlx_memory_ceiling_bytes,
        prompt_cache_config,
        optimizer_state_directory,
        performance_attribution_enabled,
        performance_attribution_log_path,
        prompt_processing_chunk_sizer_override,
        mtp_enabled,
        speculative_prefill: worker_startup_configuration.speculative_prefill.clone(),
        persistent_prompt_cache_enabled,
        chunking,
    };
    let engine_worker: EngineBackedWorker<
        ModelFamilyGenerationProcessor,
        ModelFamilyInferenceEngine,
        ModelFamilyFactory,
    > = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        model_factory,
        u64::try_from(machine_mlx_memory_ceiling_bytes).map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidGpuWiredMemoryLimit {
                description: "machine MLX memory ceiling exceeds the u64 range",
            })
        })?,
        u64::try_from(effective_mlx_memory_ceiling_bytes).map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidGpuWiredMemoryLimit {
                description: "MLX memory ceiling exceeds the u64 range",
            })
        })?,
    )
    .with_worker_runtime_feature_configuration(worker_runtime_feature_configuration);
    engine_worker
        .run_with_protocol(command_reader, event_writer)
        .await
        .map_err(WorkerProcessError::Runtime)
}
