use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
use astronomical_mlx_c_rust::set_wired_limit;
use astronomical_model_serving::effective_mlx_memory_ceiling_bytes;
use astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::model_family_factory::{InferenceWorker, ModelFamilyFactory};

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";
const WIRED_CLAIM_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_MLX_WIRED_CLAIM";

pub use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup_gpu_memory::machine_mlx_memory_ceiling_bytes;
pub use crate::worker_startup_logging::initialize_tracing;

/// Decides whether the worker claims Metal wired residency at startup.
///
/// The claim is on by default: Astronomical is this machine's primary
/// model-serving process, and an unclaimed resident model loses every
/// memory fight. Measured twice on 2026-10-16 with a 34 GB unwired resident
/// model on a busy 48 GB machine (compressor holding ~25 GB, swap full,
/// kernel pressure level 2): the kernel's default wiring heuristic had
/// wired only ~3.8 GB of the model's pages, macOS compressed the rest, and
/// the first prefill after load faulted the whole working set back from
/// disk — 52.9 s for a 64-token chunk and 42.8 s for a 558-token chunk.
/// With the claim raised to the kernel's own recommended working set, MLX
/// allocations become pageable-no-more and under pressure macOS pages the
/// neighbors instead of the model. Explicit `ASTRONOMICAL_MLX_WIRED_CLAIM=0`
/// is the escape hatch for machines or workflows that must keep serving
/// without a wired pool; any other value (including unset) keeps the claim.
pub(crate) fn wired_residency_claim_is_enabled(
    environment_override: Option<&std::ffi::OsStr>,
) -> bool {
    environment_override.is_none_or(|claim_value| claim_value != "0")
}

/// Claims Metal wired residency up to the system recommendation for the
/// worker lifetime — the mlx-lm large-model pattern — so that under memory
/// pressure the operating system pages other processes instead of the
/// loaded model's weights. The API rejects claims above the recommendation
/// and needs no elevation; the kernel releases the claim when the worker
/// exits. A failed claim never blocks serving: the worker continues
/// unprotected and says so in the log.
fn claim_wired_residency_by_default() {
    if !wired_residency_claim_is_enabled(
        std::env::var_os(WIRED_CLAIM_ENVIRONMENT_VARIABLE).as_deref(),
    ) {
        tracing::info!("wired-residency claim disabled by environment; serving unprotected");
        return;
    }
    match maximum_recommended_gpu_working_set_size_bytes() {
        Ok(recommended_working_set_bytes) => match set_wired_limit(recommended_working_set_bytes) {
            Ok(previous_wired_limit_bytes) => tracing::info!(
                claimed_wired_limit_bytes = recommended_working_set_bytes,
                previous_wired_limit_bytes,
                "claimed Metal wired residency for the worker lifetime"
            ),
            Err(claim_error) => tracing::warn!(
                error = %claim_error,
                "wired-residency claim failed; serving continues unprotected"
            ),
        },
        Err(recommendation_error) => tracing::warn!(
            error = %recommendation_error,
            "could not read the recommended working set; wired claim skipped"
        ),
    }
}

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
    let effective_mlx_memory_ceiling_bytes =
        usize::try_from(effective_mlx_memory_ceiling_bytes_u64).map_err(|_| {
            WorkerProcessError::Startup(WorkerStartupError::InvalidPhysicalMemorySize {
                description: "effective MLX memory ceiling exceeds the platform integer range",
            })
        })?;
    claim_wired_residency_by_default();
    tracing::info!(
        global_prompt_cache_maximum_size_bytes = ?prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
        performance_attribution_enabled,
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
        performance_attribution_log_path,
        persistent_prompt_cache_enabled,
    );
    let engine_worker: InferenceWorker =
        InferenceWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
            model_factory,
            machine_mlx_memory_ceiling_bytes_u64,
            effective_mlx_memory_ceiling_bytes_u64,
        )
        .with_worker_runtime_feature_configuration(worker_runtime_feature_configuration);
    engine_worker
        .run_with_protocol(command_reader, event_writer)
        .await
        .map_err(WorkerProcessError::Runtime)
}
