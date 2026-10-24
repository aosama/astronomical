use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
use astronomical_model_serving::effective_mlx_memory_ceiling_after_wired_claim;
use astronomical_model_serving::effective_mlx_memory_ceiling_bytes;
use astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes;
use mlx_c_rust::set_wired_limit;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::model_family_factory::{InferenceWorker, ModelFamilyFactory};

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";

pub use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup_gpu_memory::machine_mlx_memory_ceiling_bytes;
pub use crate::worker_startup_logging::initialize_tracing;

/// The largest wired-residency budget MLX will grant for the machine.
///
/// MLX rejects claims above Metal's recommended working set (its
/// `set_wired_limit` throws `invalid_argument` above the recommendation),
/// so the recommendation is the absolute wiring ceiling even on machines
/// where the memory ceiling sits higher. The claim is sized to the machine
/// ceiling, not the configured serving ceiling: the wired limit only caps how
/// many allocated bytes may stay wired, and the serving ceiling already bounds
/// allocation, so a larger claim wires nothing extra. Sizing it to the
/// configured setting instead froze every later live raise at the startup
/// setting, because the claim lasts the worker lifetime and the menu stepper
/// offers nothing above the reported budget. The engine still clamps its
/// ceiling to the granted budget: a ceiling above it is the 2026-10-16 fatal
/// gap (issue #1297), where admission planned up to the 42,949,672,960-byte
/// machine ceiling while the driver could keep only 40,000,028,672 bytes
/// GPU-resident.
pub(crate) fn wired_residency_claim_budget_bytes(
    machine_mlx_memory_ceiling_bytes: usize,
    recommended_working_set_bytes: usize,
) -> usize {
    machine_mlx_memory_ceiling_bytes.min(recommended_working_set_bytes)
}

/// Claims Metal wired residency for the worker lifetime up to the machine
/// ceiling, capped at the system recommendation — the mlx-lm large-model
/// pattern — so that under memory pressure the operating system pages other
/// processes instead of the loaded model's weights. Returns the granted
/// budget the engine must clamp its ceiling to, or `None` when serving
/// continues without a wired pool (recommendation unreadable or the claim
/// rejected). A failed claim never blocks
/// serving: with no wired budget the allocations stay pageable, so the
/// ceiling keeps its original value; the kernel releases the claim when
/// the worker exits.
fn claim_wired_residency_up_to(machine_mlx_memory_ceiling_bytes: usize) -> Option<usize> {
    let recommended_working_set_bytes = match maximum_recommended_gpu_working_set_size_bytes() {
        Ok(recommended_working_set_bytes) => recommended_working_set_bytes,
        Err(recommendation_error) => {
            tracing::warn!(
                error = %recommendation_error,
                "could not read the recommended working set; wired claim skipped"
            );
            return None;
        }
    };
    let claim_budget_bytes = wired_residency_claim_budget_bytes(
        machine_mlx_memory_ceiling_bytes,
        recommended_working_set_bytes,
    );
    match set_wired_limit(claim_budget_bytes) {
        Ok(previous_wired_limit_bytes) => {
            tracing::info!(
                claimed_wired_limit_bytes = claim_budget_bytes,
                previous_wired_limit_bytes,
                machine_mlx_memory_ceiling_bytes,
                "claimed Metal wired residency for the worker lifetime"
            );
            Some(claim_budget_bytes)
        }
        Err(claim_error) => {
            tracing::warn!(
                error = %claim_error,
                "wired-residency claim failed; serving continues unprotected"
            );
            None
        }
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
    let claimed_wired_residency_budget_bytes =
        claim_wired_residency_up_to(machine_mlx_memory_ceiling_bytes);
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
            "serving ceiling clamped to the wired-residency budget the driver can honor"
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
