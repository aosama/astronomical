use std::path::PathBuf;

use astronomical_config::{PrefillChunckSizingPolicy, PromptCacheConfig};
use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerCommand, WorkerPrefillChunckSizingPolicy,
    WorkerStartupConfiguration,
};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, EngineBackedWorker, ModelFactory,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5GenerationProcessor, Qwen3_5PrefillChunckSizer,
};
use tokio::io::{AsyncRead, AsyncWrite};

const PERFORMANCE_ATTRIBUTION_LOG_FILE_NAME: &str = "performance-attribution.jsonl";

pub use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};
pub use crate::worker_startup_gpu_memory::{
    GpuWiredMemoryLimitSetting, derive_mlx_memory_limits_from_gpu_wired_limit,
    parse_iogpu_wired_limit_setting, resolve_effective_mlx_memory_ceiling_bytes,
    sample_iogpu_wired_limit_bytes,
};
pub use crate::worker_startup_logging::{astronomical_log_rotation, initialize_tracing};

/// Factory that creates a Qwen3.5 processor and engine when a REST request
/// selects a model directory.
struct Qwen3_5ModelFactory {
    effective_mlx_memory_ceiling_bytes: usize,
    prompt_cache_config: PromptCacheConfig,
    prefill_chunck_sizing_policy: PrefillChunckSizingPolicy,
    optimizer_state_directory: Option<PathBuf>,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
    prefill_chunck_sizer_override: Option<Qwen3_5PrefillChunckSizer>,
    mtp_enabled: bool,
    persistent_prompt_cache_enabled: bool,
}

impl ModelFactory<Qwen3_5GenerationProcessor, Qwen3_5Engine> for Qwen3_5ModelFactory {
    async fn create(
        &self,
        model_directory: &str,
        max_output_tokens: u32,
    ) -> Result<(Qwen3_5GenerationProcessor, Qwen3_5Engine), String> {
        let model_directory_path = PathBuf::from(model_directory);
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let prompt_cache_config = self.prompt_cache_config.clone();
        let prefill_chunck_sizing_policy = self.prefill_chunck_sizing_policy;
        let optimizer_state_directory = self.optimizer_state_directory.clone();
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let prefill_chunck_sizer_override = self.prefill_chunck_sizer_override.clone();
        let mtp_enabled = self.mtp_enabled;
        let persistent_prompt_cache_enabled = self.persistent_prompt_cache_enabled;
        // Model initialization involves blocking I/O (artifact validation, tokenizer
        // loading) and must run off the async runtime.
        tokio::task::spawn_blocking(move || {
            initialize_qwen3_5_model(
                model_directory_path,
                effective_mlx_memory_ceiling_bytes,
                prompt_cache_config,
                prefill_chunck_sizing_policy,
                prefill_chunck_sizer_override,
                optimizer_state_directory,
                max_output_tokens,
                mtp_enabled,
                persistent_prompt_cache_enabled,
                performance_attribution_enabled,
                performance_attribution_log_path,
            )
            .map_err(|startup_error| startup_error.public_model_load_failure_reason())
        })
        .await
        .map_err(|join_error| join_error.to_string())?
    }

    fn update_mlx_memory_ceiling_bytes(&mut self, effective_mlx_memory_ceiling_bytes: u64) {
        self.effective_mlx_memory_ceiling_bytes =
            usize::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX);
    }
}

/// Starts an idle worker that loads a Qwen3.5 model only after an explicit
/// `SwapModel` command from the supervisor.
pub async fn run_bootstrapped_worker<ReadTransport, WriteTransport>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    run_bootstrapped_worker_with_prefill_chunck_sizer_override(
        read_transport,
        write_transport,
        None,
    )
    .await
}

async fn run_bootstrapped_worker_with_prefill_chunck_sizer_override<ReadTransport, WriteTransport>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
    prefill_chunck_sizer_override: Option<Qwen3_5PrefillChunckSizer>,
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
        prefill_chunck_sizer_override,
    )
    .await
}

/// Runs the configured worker with explicit model-owned `prefill_chunck_tokens` for benchmarks.
#[cfg(feature = "performance-measurement")]
pub async fn run_configured_worker_with_prefill_chunck_sizer_override<
    ReadTransport,
    WriteTransport,
>(
    read_transport: ReadTransport,
    write_transport: WriteTransport,
    prefill_chunck_sizer: Qwen3_5PrefillChunckSizer,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    run_bootstrapped_worker_with_prefill_chunck_sizer_override(
        read_transport,
        write_transport,
        Some(prefill_chunck_sizer),
    )
    .await
}

async fn run_initialized_worker<ReadTransport, WriteTransport>(
    worker_startup_configuration: WorkerStartupConfiguration,
    command_reader: ProtocolReader<ReadTransport>,
    event_writer: ProtocolWriter<WriteTransport>,
    prefill_chunck_sizer_override: Option<Qwen3_5PrefillChunckSizer>,
) -> Result<(), WorkerProcessError>
where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    let prefill_chunck_sizing_policy =
        match worker_startup_configuration.prefill_chunck_sizing_policy {
            WorkerPrefillChunckSizingPolicy::Optimized => PrefillChunckSizingPolicy::Optimized,
            WorkerPrefillChunckSizingPolicy::Fixed {
                fixed_prefill_chunck_tokens,
            } => PrefillChunckSizingPolicy::Fixed {
                fixed_prefill_chunck_tokens,
            },
        };
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
        prefill_chunck_sizing_policy = ?prefill_chunck_sizing_policy,
        optimizer_state_directory = ?optimizer_state_directory
            .as_ref()
            .map(|optimizer_state_directory| optimizer_state_directory.display()),
        performance_attribution_enabled,
        machine_mlx_memory_ceiling_bytes,
        configured_maximum_mlx_memory_bytes,
        effective_mlx_memory_ceiling_bytes,
        mtp_enabled,
        persistent_prompt_cache_enabled,
        "starting idle inference worker"
    );
    let model_factory = Qwen3_5ModelFactory {
        effective_mlx_memory_ceiling_bytes,
        prompt_cache_config,
        prefill_chunck_sizing_policy,
        optimizer_state_directory,
        performance_attribution_enabled,
        performance_attribution_log_path,
        prefill_chunck_sizer_override,
        mtp_enabled,
        persistent_prompt_cache_enabled,
    };
    EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
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
    .run_with_protocol(command_reader, event_writer)
    .await
    .map_err(WorkerProcessError::Runtime)
}

#[allow(clippy::too_many_arguments)]
fn initialize_qwen3_5_model(
    model_directory_path: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    prompt_cache_config: PromptCacheConfig,
    prefill_chunck_sizing_policy: PrefillChunckSizingPolicy,
    prefill_chunck_sizer_override: Option<Qwen3_5PrefillChunckSizer>,
    optimizer_state_directory: Option<PathBuf>,
    max_output_tokens: u32,
    mtp_enabled: bool,
    persistent_prompt_cache_enabled: bool,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
) -> Result<(Qwen3_5GenerationProcessor, Qwen3_5Engine), WorkerStartupError> {
    let mut model_loading_performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let mut performance_attribution_log = PerformanceAttributionLog::open(
        &performance_attribution_log_path,
        performance_attribution_enabled,
    )
    .map_err(|source| WorkerStartupError::OpenPerformanceAttributionLog {
        performance_attribution_log_path: performance_attribution_log_path.clone(),
        source,
    })?;
    let validated_artifact = match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::ArtifactValidation,
        |_performance_attribution| {
            Qwen3_5ArtifactValidator::new().validate(&model_directory_path, max_output_tokens)
        },
    ) {
        Ok(validated_artifact) => validated_artifact,
        Err(source) => {
            tracing::warn!(
                error = %source,
                model_directory = ?model_directory_path,
                "Qwen3.5 artifact validation failed during model initialization"
            );
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                None,
                None,
                None,
                None,
                "artifact validation failed",
            );
            return Err(WorkerStartupError::Qwen3_5ArtifactValidation {
                model_directory: model_directory_path.clone(),
                source,
            });
        }
    };
    let artifact_model_id = validated_artifact.model_id().to_owned();
    let artifact_model_revision = validated_artifact.revision().to_owned();
    let artifact_payload_bytes = validated_artifact.total_payload_bytes();
    let artifact_shard_count = validated_artifact.shard_count();
    let generation_processor = match model_loading_performance_attribution.measure_operation(
        PerformanceOperation::TokenizerInitialization,
        |_performance_attribution| {
            Qwen3_5GenerationProcessor::from_validated_artifact(
                &validated_artifact,
                true,
                performance_attribution_enabled,
            )
        },
    ) {
        Ok(generation_processor) => generation_processor,
        Err(source) => {
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                Some(artifact_model_id),
                Some(artifact_model_revision),
                Some(artifact_payload_bytes),
                Some(artifact_shard_count),
                "tokenizer initialization failed",
            );
            return Err(WorkerStartupError::Qwen3_5ProcessorInitialization {
                model_directory: model_directory_path.clone(),
                source,
            });
        }
    };
    let think_end_token_id = generation_processor.think_end_token_id();
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let maximum_prefill_chunck_tokens = validated_artifact.config().maximum_position_count();
    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        derive_mlx_memory_limits_from_gpu_wired_limit(effective_mlx_memory_ceiling_bytes);
    let per_model_prompt_cache_config = prompt_cache_config.for_model(&model_id, &model_revision);
    let persistent_prompt_cache_disk_store_config = persistent_prompt_cache_enabled.then(|| {
        PersistentPromptCacheDiskStoreConfig::new(
            per_model_prompt_cache_config
                .active_model_prompt_cache_directory()
                .clone(),
            per_model_prompt_cache_config
                .global_prompt_cache_root_directory()
                .clone(),
            per_model_prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
        )
    });
    let prefill_chunck_sizer_result = match prefill_chunck_sizer_override {
        Some(prefill_chunck_sizer) => Ok(prefill_chunck_sizer),
        None => match prefill_chunck_sizing_policy {
            PrefillChunckSizingPolicy::Fixed {
                fixed_prefill_chunck_tokens,
            } => Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(
                fixed_prefill_chunck_tokens,
            ),
            PrefillChunckSizingPolicy::Optimized => match optimizer_state_directory {
                Some(optimizer_directory) => {
                    Qwen3_5PrefillChunckSizer::for_optimized_production_with_persisted_state(
                        maximum_prefill_chunck_tokens,
                        optimizer_directory,
                        model_id,
                        model_revision,
                    )
                }
                None => Qwen3_5PrefillChunckSizer::production(maximum_prefill_chunck_tokens),
            },
        },
    };
    let prefill_chunck_sizer = match prefill_chunck_sizer_result {
        Ok(prefill_chunck_sizer) => prefill_chunck_sizer,
        Err(prefill_chunck_sizer_error) => {
            record_failed_model_loading_performance_attribution(
                model_loading_performance_attribution,
                &mut performance_attribution_log,
                Some(artifact_model_id),
                Some(artifact_model_revision),
                Some(artifact_payload_bytes),
                Some(artifact_shard_count),
                "prompt-processing chunk configuration failed",
            );
            return Err(WorkerStartupError::PrefillChunckSizing(
                prefill_chunck_sizer_error,
            ));
        }
    };
    let qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_performance_attribution(
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        persistent_prompt_cache_disk_store_config,
        prefill_chunck_sizer,
        think_end_token_id,
        model_directory_path.clone(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
        mtp_enabled,
        model_loading_performance_attribution,
        performance_attribution_log,
    )
    .map_err(|source| WorkerStartupError::Qwen3_5EngineInitialization {
        model_directory: model_directory_path,
        source,
    })?;
    Ok((generation_processor, qwen3_5_engine))
}

#[allow(clippy::too_many_arguments)]
fn record_failed_model_loading_performance_attribution(
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: &mut PerformanceAttributionLog,
    model_id: Option<String>,
    model_revision: Option<String>,
    total_artifact_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
    failure_description: &'static str,
) {
    let Some(performance_attribution_report) = model_loading_performance_attribution
        .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
            outcome: PerformanceAttributionOutcome::Failed,
            model_id,
            model_revision,
            prefill_transient_observation_completed: false,
            prefill_observed_transient_high_water_bytes: 0,
            retained_complete_expert_layer_count: 0,
            total_artifact_payload_bytes,
            resident_model_payload_bytes: None,
            model_shard_count,
            mlx_active_memory_bytes: None,
            mlx_allocator_cache_memory_bytes: None,
            mlx_peak_memory_bytes: None,
            failure_description: Some(failure_description.to_owned()),
        })
    else {
        return;
    };
    if let Err(performance_attribution_write_error) =
        performance_attribution_log.record(&performance_attribution_report)
    {
        tracing::warn!(
            error = %performance_attribution_write_error,
            "failed to append model-loading performance attribution"
        );
    }
}
