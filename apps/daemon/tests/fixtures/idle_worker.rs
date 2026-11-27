//! Deterministic subprocess fixture for worker lifecycle and event-order tests.
//!
//! Model-directory suffixes select scripted protocol behavior. The fixture uses
//! the real framed standard-input/standard-output transport, so tests exercise
//! daemon queueing and event handling without loading MLX or local artifacts.

use std::{error::Error, process::ExitCode, time::Duration};

use ipc_protocol::{
    ChatGenerationCompletionReason, ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolReader,
    ProtocolWriter, RequestId, WorkerCommand, WorkerEvent, WorkerMlxMemorySnapshot,
    WorkerModelConfiguration, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};

// This smaller fixture reuses only capability and completion helpers from the shared scenario module.
#[allow(dead_code)]
mod scripted_worker_image;

mod fixture_chat_capabilities;
mod fixture_memory_limit_answers;
mod scripted_worker_decisions;
mod scripted_worker_single_shot;
mod scripted_worker_speech;

const DELAYED_COMPLETION_MODEL_ID: &str = "astronomical/delayed-completion-model";
const GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: &str =
    "astronomical/generation-event-before-swap-model";
const TELEMETRY_BEFORE_SWAP_MODEL_ID: &str = "astronomical/telemetry-before-swap-model";
const DELAYED_POLICY_ACK_MODEL_ID: &str = "astronomical/delayed-policy-ack-model";
const DELAYED_IMAGE_POLICY_ACK_MODEL_ID: &str = "astronomical/delayed-image-policy-ack-model";
const DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION: &str =
    "delayed-startup-runtime-configuration";
const OPTIONAL_STARTUP_CONFIGURATION_TIMEOUT: Duration = Duration::from_millis(100);

#[tokio::main]
async fn main() -> ExitCode {
    match run_fixture().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(fixture_error) => {
            eprintln!("idle worker fixture failed: {fixture_error}");
            ExitCode::FAILURE
        }
    }
}

type FixtureError = Box<dyn Error + Send + Sync>;
type WorkerEventWriter = ProtocolWriter<tokio::io::Stdout>;

/// What the fixture remembers between commands.
struct IdleWorkerState {
    loaded_model_id: Option<String>,
    acknowledged_runtime_configuration: Option<WorkerRuntimeFeatureConfiguration>,
}

async fn run_fixture() -> Result<(), FixtureError> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let mut worker_state = announce_idle_worker(&mut command_reader, &mut event_writer).await?;
    while let Some(worker_command) = command_reader.next_command().await? {
        answer_worker_command(&mut event_writer, &mut worker_state, worker_command).await?;
    }
    Ok(())
}

/// Reads the optional startup command, reports Idle with the ceiling it
/// configured and, when a startup configuration arrived, acknowledges it.
async fn announce_idle_worker(
    command_reader: &mut ProtocolReader<tokio::io::Stdin>,
    event_writer: &mut WorkerEventWriter,
) -> Result<IdleWorkerState, FixtureError> {
    let mut acknowledged_runtime_configuration: Option<WorkerRuntimeFeatureConfiguration> = None;
    // Production writes InitializeWorker immediately after spawning the worker,
    // before any acknowledgement can be read. The idle ceiling therefore comes
    // from the startup configuration exactly as the real worker computes it.
    let mut configured_effective_mlx_memory_ceiling_bytes = 40_000_000_000;
    let startup_command = match tokio::time::timeout(
        OPTIONAL_STARTUP_CONFIGURATION_TIMEOUT,
        command_reader.next_command(),
    )
    .await
    {
        Ok(startup_command) => startup_command?,
        Err(_) => None,
    };
    match startup_command {
        Some(WorkerCommand::InitializeWorker(worker_startup_configuration)) => {
            if let Some(configured_maximum_mlx_memory_bytes) =
                worker_startup_configuration.configured_maximum_mlx_memory_bytes
            {
                configured_effective_mlx_memory_ceiling_bytes = configured_maximum_mlx_memory_bytes;
            }
            acknowledged_runtime_configuration = Some(runtime_configuration_acknowledging(
                worker_startup_configuration,
            ));
        }
        Some(unexpected_startup_command) => {
            return Err(std::io::Error::other(format!(
                "unexpected worker command before initial readiness: {unexpected_startup_command:?}"
            ))
            .into());
        }
        None => {}
    }
    event_writer
        .send_event(&WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
            effective_mlx_memory_ceiling_bytes: configured_effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
        })
        .await?;
    if let Some(runtime_configuration) = acknowledged_runtime_configuration.as_ref() {
        if runtime_configuration.configuration_generation
            == DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION
        {
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        send_runtime_configuration_applied(event_writer, runtime_configuration.clone()).await?;
    }
    Ok(IdleWorkerState {
        loaded_model_id: None,
        acknowledged_runtime_configuration,
    })
}

/// The runtime configuration a worker acknowledges for this startup
/// configuration, with no model loaded yet.
fn runtime_configuration_acknowledging(
    worker_startup_configuration: WorkerStartupConfiguration,
) -> WorkerRuntimeFeatureConfiguration {
    WorkerRuntimeFeatureConfiguration {
        configuration_generation: worker_startup_configuration.configuration_generation,
        astro_ssd_prompt_cache_enabled: worker_startup_configuration.astro_ssd_prompt_cache_enabled,
        astro_ssd_prompt_cache_maximum_size_bytes: worker_startup_configuration
            .global_astro_ssd_prompt_cache_maximum_size_bytes,
        loaded_model: None,
    }
}

async fn send_runtime_configuration_applied(
    event_writer: &mut WorkerEventWriter,
    worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
) -> Result<(), ipc_protocol::ProtocolError> {
    event_writer
        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration,
        })
        .await
}

async fn answer_worker_command(
    event_writer: &mut WorkerEventWriter,
    worker_state: &mut IdleWorkerState,
    worker_command: WorkerCommand,
) -> Result<(), FixtureError> {
    match worker_command {
        WorkerCommand::InitializeWorker(worker_startup_configuration) => {
            answer_mid_run_initialization(event_writer, worker_state, worker_startup_configuration)
                .await?;
        }
        WorkerCommand::SwapModel {
            model_directory,
            model_configuration,
        } => {
            answer_swap_model(
                event_writer,
                worker_state,
                &model_directory,
                &model_configuration,
            )
            .await?;
        }
        single_shot_command @ (WorkerCommand::GenerateDecisions(_)
        | WorkerCommand::TrainDecisionModel(_)
        | WorkerCommand::GenerateEmbeddings(_)
        | WorkerCommand::GenerateSpeech(_)) => {
            scripted_worker_single_shot::answer_single_shot_command(
                event_writer,
                single_shot_command,
            )
            .await?;
        }
        WorkerCommand::GenerateImage(generation_command) => {
            answer_image_generation(event_writer, generation_command).await?;
        }
        // A count is answered like a generation: one Completed event.
        WorkerCommand::Generate(generation_command)
        | WorkerCommand::CountChatTokens(generation_command) => {
            if worker_state.loaded_model_id.as_deref() == Some(DELAYED_COMPLETION_MODEL_ID) {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            send_empty_completion(event_writer, generation_command.request_id).await?;
        }
        WorkerCommand::Cancel { .. } => {}
        WorkerCommand::SampleMlxMemory => {
            emit_memory_snapshot(event_writer, MlxMemorySnapshotSource::IdlePoll).await?;
        }
        WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes,
            configuration_generation,
        } => {
            answer_memory_limit_update(
                event_writer,
                worker_state,
                effective_mlx_memory_ceiling_bytes,
                configuration_generation,
            )
            .await?;
        }
        WorkerCommand::ClearAstroSsdPromptCache { model_id } => {
            answer_astro_ssd_prompt_cache_clear(event_writer, model_id).await?;
        }
    }
    Ok(())
}

async fn answer_mid_run_initialization(
    event_writer: &mut WorkerEventWriter,
    worker_state: &mut IdleWorkerState,
    worker_startup_configuration: WorkerStartupConfiguration,
) -> Result<(), FixtureError> {
    // `announce_idle_worker` consumes the launch-time InitializeWorker; this
    // branch stays for fixtures re-initialized mid-run, where the
    // post-Idle event order (acknowledgement, not Idle) applies.
    if worker_startup_configuration.configuration_generation
        == DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION
    {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }
    let runtime_configuration = runtime_configuration_acknowledging(worker_startup_configuration);
    send_runtime_configuration_applied(event_writer, runtime_configuration.clone()).await?;
    worker_state.acknowledged_runtime_configuration = Some(runtime_configuration);
    Ok(())
}

async fn answer_swap_model(
    event_writer: &mut WorkerEventWriter,
    worker_state: &mut IdleWorkerState,
    model_directory: &str,
    model_configuration: &WorkerModelConfiguration,
) -> Result<(), FixtureError> {
    if model_directory.ends_with("hanging-model") {
        return Ok(());
    }
    if model_directory.ends_with("invalid-model") {
        event_writer
            .send_event(&WorkerEvent::ModelSwapFailed {
                loaded_model_remains_ready: false,
                model_load_failure_reason: "model artifact validation failed: OptiQ metadata uses unsupported 2-bit quantization".to_owned(),
            })
            .await?;
        return Ok(());
    }
    complete_model_swap(event_writer, worker_state, model_configuration).await
}

async fn complete_model_swap(
    event_writer: &mut WorkerEventWriter,
    worker_state: &mut IdleWorkerState,
    model_configuration: &WorkerModelConfiguration,
) -> Result<(), FixtureError> {
    let replacement_model_id = model_configuration.model_id();
    // These two branches deliberately violate the old assumption
    // that ModelSwapped is always the next frame after SwapModel.
    if replacement_model_id == TELEMETRY_BEFORE_SWAP_MODEL_ID {
        emit_memory_snapshot(event_writer, MlxMemorySnapshotSource::IdlePoll).await?;
    } else if replacement_model_id == GENERATION_EVENT_BEFORE_SWAP_MODEL_ID {
        send_empty_completion(event_writer, RequestId::new(999)).await?;
    }
    event_writer
        .send_event(&WorkerEvent::ModelSwapped {
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            minimum_mlx_memory_ceiling_bytes: 3_000_000_000,
            model_id: replacement_model_id.to_owned(),
            capabilities: swapped_model_capabilities(model_configuration),
        })
        .await?;
    if let Some(mut runtime_configuration) = worker_state.acknowledged_runtime_configuration.clone()
    {
        if matches!(
            replacement_model_id,
            DELAYED_POLICY_ACK_MODEL_ID | DELAYED_IMAGE_POLICY_ACK_MODEL_ID
        ) {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        runtime_configuration.loaded_model = Some(model_configuration.runtime_configuration());
        send_runtime_configuration_applied(event_writer, runtime_configuration).await?;
    }
    worker_state.loaded_model_id = Some(replacement_model_id.to_owned());
    emit_memory_snapshot(event_writer, MlxMemorySnapshotSource::ModelLoaded).await?;
    Ok(())
}

async fn answer_image_generation(
    event_writer: &mut WorkerEventWriter,
    generation_command: ipc_protocol::ImageGenerationCommand,
) -> Result<(), FixtureError> {
    if generation_command.prompt == "must-not-dispatch-after-disconnect" {
        return Err(
            std::io::Error::other("image generation dispatched after client disconnect").into(),
        );
    }
    scripted_worker_image::send_completed_image(
        generation_command,
        std::time::Duration::ZERO,
        event_writer,
    )
    .await?;
    Ok(())
}

async fn answer_memory_limit_update(
    event_writer: &mut WorkerEventWriter,
    worker_state: &mut IdleWorkerState,
    effective_mlx_memory_ceiling_bytes: u64,
    configuration_generation: String,
) -> Result<(), FixtureError> {
    if effective_mlx_memory_ceiling_bytes == 30_000_000_000 {
        return Ok(());
    }
    if effective_mlx_memory_ceiling_bytes == 31 * 1_073_741_824 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        event_writer
            .send_event(&fixture_memory_limit_answers::refused_memory_limit(
                effective_mlx_memory_ceiling_bytes,
                1,
                "fixture rejected the requested limit",
            ))
            .await?;
        return Ok(());
    }
    event_writer
        .send_event(&fixture_memory_limit_answers::applied_memory_limit(
            effective_mlx_memory_ceiling_bytes,
        ))
        .await?;
    if let Some(runtime_configuration) = worker_state.acknowledged_runtime_configuration.as_mut() {
        runtime_configuration.configuration_generation = configuration_generation;
    }
    Ok(())
}

async fn answer_astro_ssd_prompt_cache_clear(
    event_writer: &mut WorkerEventWriter,
    model_id: Option<String>,
) -> Result<(), FixtureError> {
    let acknowledged_model_id =
        if model_id.as_deref() == Some("astronomical/mismatched-clear-model") {
            Some("astronomical/different-model".to_owned())
        } else {
            model_id
        };
    emit_astro_ssd_prompt_cache_stats_after_clear(event_writer).await?;
    event_writer
        .send_event(&WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: acknowledged_model_id,
            blocks_removed: 3,
            bytes_freed: 4_096,
        })
        .await?;
    Ok(())
}

/// The one Completed event a generation, a count and the generation event
/// raised before a swap are answered with: a one-token prompt, nothing
/// generated.
async fn send_empty_completion(
    event_writer: &mut WorkerEventWriter,
    request_id: RequestId,
) -> Result<(), ipc_protocol::ProtocolError> {
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            astro_ssd_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}

/// What the swapped-in model advertises, by its kind: chat for an
/// autoregressive model, speech for a speech model, and images otherwise.
fn swapped_model_capabilities(
    model_configuration: &ipc_protocol::WorkerModelConfiguration,
) -> ipc_protocol::WorkerModelCapabilities {
    if let Some(autoregressive_configuration) = model_configuration.autoregressive() {
        return fixture_chat_capabilities::swapped_chat_capabilities(autoregressive_configuration)
            .into();
    }
    if matches!(
        model_configuration,
        ipc_protocol::WorkerModelConfiguration::Speech(_)
    ) {
        return ipc_protocol::WorkerModelCapabilities::speech(
            ipc_protocol::WorkerSpeechCapabilities {
                sample_rate_hertz: 24_000,
                voice_names: vec!["af_heart".to_owned()],
                default_voice_name: "af_heart".to_owned(),
                supports_speed: true,
            },
        );
    }
    ipc_protocol::WorkerModelCapabilities::image_generation(
        scripted_worker_image::image_capabilities(),
    )
}

async fn emit_memory_snapshot<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    mlx_memory_snapshot_source: MlxMemorySnapshotSource,
) -> Result<(), ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: mlx_memory_snapshot_source,
                active_memory_bytes: 20_000_000_000,
                mlx_buffer_cache_bytes: 0,
                peak_memory_bytes: 20_000_000_000,
                expert_payload_bytes: 12_000_000_000,
                model_core_payload_bytes: 8_000_000_000,
                context_state_payload_bytes: 0,
                memory_ceiling_utilization: None,
                memory_ledger: None,
            }),
            expert_residency: None,
        })
        .await
}

async fn emit_astro_ssd_prompt_cache_stats_after_clear<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::AstroSsdPromptCacheStats {
            astro_ssd_prompt_cache_hits: 0,
            astro_ssd_prompt_cache_misses: 0,
            astro_ssd_prompt_cache_tokens_saved: 0,
            astro_ssd_prompt_cache_block_token_count: 2_048,
            astro_ssd_prompt_cache_sequence_state_block_count: 0,
            astro_ssd_prompt_cache_boundary_state_snapshot_count: 0,
            astro_ssd_prompt_cache_visual_embedding_count: 0,
            astro_ssd_prompt_cache_total_size_bytes: 0,
            astro_ssd_prompt_cache_visual_embedding_total_size_bytes: 0,
            astro_ssd_prompt_cache_maximum_size_bytes: 50_000_000_000,
            astro_ssd_prompt_cache_visual_embedding_hits: 0,
            astro_ssd_prompt_cache_visual_embedding_misses: 0,
            astro_ssd_prompt_cache_visual_embedding_rows_loaded: 0,
            astro_ssd_prompt_cache_partial_tail_hits: 0,
        })
        .await
}
