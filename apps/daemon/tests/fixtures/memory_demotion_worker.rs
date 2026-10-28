//! Deterministic worker fixture that records its own process identity and the
//! liveness of every sibling worker already recorded in its logging directory.
//!
//! The memory-demotion contract is that the trusted worker has exited and been
//! reaped before any replacement process is launched. Asserting that from the
//! daemon's own control flow would only restate the implementation, so this
//! fixture instead lets the *replacement* report what it observed at the instant
//! it started: if the trusted worker were still alive then, the overlap is
//! proven regardless of what the daemon believes.
//!
//! Configuration generations select the scripted failure branches. The fixture
//! speaks the real framed standard-input/standard-output transport, so no MLX or
//! local artifact is involved.

#![forbid(unsafe_code)]

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    process::ExitCode,
};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, ExpertMemoryMode,
    ImageGenerationCapabilities, MlxMemorySnapshotSource, ProtocolError, ProtocolReader,
    ProtocolWriter, WorkerAutoregressiveModelConfiguration, WorkerCommand, WorkerEvent,
    WorkerLoadedModelRuntimeConfiguration, WorkerMlxMemorySnapshot, WorkerModelCapabilities,
    WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};

/// How long the fixture waits for the daemon's guaranteed-first
/// `InitializeWorker` command. Every scenario using this fixture launches the
/// worker through `WorkerHandle`, which writes that command immediately after
/// spawning, so a long wait is free on an idle host; the window only has to be
/// wide enough to absorb a heavily loaded continuous-integration runner, where
/// a shorter timeout silently skipped the launch observation and the runtime
/// acknowledgement and diverged every role that depends on them.
const STARTUP_COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const WORKER_PROCESS_IDENTIFIER_PREFIX: &str = "demotion-worker-";
const LAUNCH_ORDER_LOG_FILE: &str = "demotion-launch-order.log";
const MACHINE_MEMORY_CEILING_BYTES: u64 = 40_000_000_000;
const MINIMUM_MEMORY_CEILING_BYTES: u64 = 1;

/// A candidate that never acknowledges its runtime configuration, forcing the
/// daemon's candidate-acknowledgement timeout to reject it.
const SILENT_ACKNOWLEDGEMENT_GENERATION: &str = "silent-acknowledgement-generation";
/// A candidate that acknowledges a configuration generation the daemon did
/// not ask for, which identity binding must reject.
const MISMATCHED_ACKNOWLEDGEMENT_GENERATION: &str = "mismatched-acknowledgement-generation";
const UNREQUESTED_ACKNOWLEDGED_GENERATION: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A model directory suffix that makes restoration fail after the restart, so
/// the daemon must not publish readiness for a model it never loaded.
const RESTORE_FAILURE_MODEL_DIRECTORY_SUFFIX: &str = "restore-failure-model";
/// The replacement role, which is the only role allowed to script a restoration
/// failure; the trusted worker must still be able to select the same model so
/// the journey starts from a genuinely loaded one.
const REPLACEMENT_CONFIGURATION_GENERATION: &str = "replacement-generation";

/// The trusted role's configuration generation.
const TRUSTED_CONFIGURATION_GENERATION: &str = "trusted-generation";

/// How long the trusted worker lingers after its input closes before exiting.
///
/// The ordering evidence is observed by the *replacement* process probing its
/// predecessor's liveness, and a replacement boots far more slowly than an
/// orderly shutdown completes. Without this pause a candidate-first bug could
/// close that window before the probe ran, and the acceptance test would pass
/// against exactly the overlap it exists to catch. Lingering well inside the
/// daemon's three-second shutdown timeout widens the window without changing
/// the ordering being measured.
const TRUSTED_EXIT_LINGER: std::time::Duration = std::time::Duration::from_millis(1_500);

fn main() -> ExitCode {
    match run_fixture() {
        Ok(()) => ExitCode::SUCCESS,
        Err(fixture_error) => {
            eprintln!("memory-demotion worker fixture failed: {fixture_error}");
            ExitCode::FAILURE
        }
    }
}

fn run_fixture() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_fixture_async())
}

#[expect(clippy::too_many_lines)]
async fn run_fixture_async() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let startup_configuration = read_startup_configuration(&mut command_reader).await?;

    record_launch_observation(
        &startup_configuration.logging_directory,
        &startup_configuration.configuration_generation,
    )?;

    let mut effective_memory_ceiling_bytes = MACHINE_MEMORY_CEILING_BYTES;
    if let Some(configured_maximum_mlx_memory_bytes) =
        startup_configuration.configured_maximum_mlx_memory_bytes
    {
        effective_memory_ceiling_bytes = configured_maximum_mlx_memory_bytes;
    }
    // The silent branch models a candidate that starts and then stops
    // talking: it must neither acknowledge nor exit, so the daemon's
    // acknowledgement timeout is what rejects it.
    if startup_configuration.configuration_generation == SILENT_ACKNOWLEDGEMENT_GENERATION {
        while command_reader.next_command().await?.is_some() {}
        return Ok(());
    }

    event_writer
        .send_event(&WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: MACHINE_MEMORY_CEILING_BYTES,
            effective_mlx_memory_ceiling_bytes: effective_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: MINIMUM_MEMORY_CEILING_BYTES,
            wired_residency_budget_bytes: None,
        })
        .await?;

    // In-place live ceiling updates rename the owning generation without a
    // restart; every later acknowledgement must carry the current generation
    // so the daemon's identity binding keeps holding after an update.
    let mut acknowledged_configuration_generation =
        startup_configuration.configuration_generation.clone();

    let acknowledged_generation = if startup_configuration.configuration_generation
        == MISMATCHED_ACKNOWLEDGEMENT_GENERATION
    {
        UNREQUESTED_ACKNOWLEDGED_GENERATION
    } else {
        startup_configuration.configuration_generation.as_str()
    };
    event_writer
        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration: runtime_configuration(
                &startup_configuration,
                acknowledged_generation,
                None,
            ),
        })
        .await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        match worker_command {
            WorkerCommand::SwapModel {
                model_directory,
                model_configuration,
            } => {
                if startup_configuration.configuration_generation
                    == REPLACEMENT_CONFIGURATION_GENERATION
                    && model_directory.ends_with(RESTORE_FAILURE_MODEL_DIRECTORY_SUFFIX)
                {
                    event_writer
                        .send_event(&WorkerEvent::ModelSwapFailed {
                            loaded_model_remains_ready: false,
                            model_load_failure_reason:
                                "scripted restoration failure for the memory-demotion journey"
                                    .to_owned(),
                        })
                        .await?;
                    continue;
                }
                event_writer
                    .send_event(&WorkerEvent::ModelSwapped {
                        expert_memory_mode: Some(ExpertMemoryMode::Hybrid),
                        minimum_mlx_memory_ceiling_bytes: MINIMUM_MEMORY_CEILING_BYTES,
                        model_id: model_configuration.model_id().to_owned(),
                        capabilities: chat_capabilities(model_configuration.autoregressive()),
                    })
                    .await?;
                event_writer
                    .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                        worker_runtime_feature_configuration: runtime_configuration(
                            &startup_configuration,
                            &acknowledged_configuration_generation,
                            Some(model_configuration.runtime_configuration()),
                        ),
                    })
                    .await?;
                emit_memory_snapshot(&mut event_writer, MlxMemorySnapshotSource::ModelLoaded)
                    .await?;
            }
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
                configuration_generation,
            } => {
                event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: MINIMUM_MEMORY_CEILING_BYTES,
                        expert_memory_mode: ExpertMemoryMode::Hybrid,
                        mlx_memory_snapshot: None,
                        expert_residency: None,
                    })
                    .await?;
                acknowledged_configuration_generation = configuration_generation;
            }
            WorkerCommand::Generate(generation_command) => {
                // Finalization precedes completion, matching the real worker's
                // emission order (engine_backed_worker output.rs) and the
                // scripted worker fixture: the daemon's completion handler
                // takes the active request, so a finalize emitted after it is a
                // protocol violation that force-terminates the worker mid-test
                // and races the demotion command against the containment.
                event_writer
                    .send_event(&WorkerEvent::GenerationFinalized {
                        request_id: generation_command.request_id,
                        expert_memory_mode: Some(ExpertMemoryMode::Hybrid),
                        mlx_memory_snapshot: None,
                        expert_residency: None,
                    })
                    .await?;
                event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id: generation_command.request_id,
                        prompt_token_count: 1,
                        generated_token_count: 1,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        persistent_prompt_cache_diagnostics: None,
                        reason: ChatGenerationCompletionReason::EndOfSequence,
                    })
                    .await?;
            }
            WorkerCommand::InitializeWorker(worker_startup_configuration) => {
                event_writer
                    .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                        worker_runtime_feature_configuration: runtime_configuration(
                            &worker_startup_configuration,
                            &worker_startup_configuration.configuration_generation,
                            None,
                        ),
                    })
                    .await?;
            }
            WorkerCommand::SampleMlxMemory => {
                // The daemon's idle sampler polls ready workers on an
                // interval with a bounded acknowledgement timeout; ignoring the
                // sample would let the wait expire and contain the worker.
                emit_memory_snapshot(&mut event_writer, MlxMemorySnapshotSource::IdlePoll).await?;
            }
            WorkerCommand::Cancel { .. } => {}
            _ => {}
        }
    }
    if startup_configuration.configuration_generation == TRUSTED_CONFIGURATION_GENERATION {
        tokio::time::sleep(TRUSTED_EXIT_LINGER).await;
    }
    Ok(())
}

async fn read_startup_configuration(
    command_reader: &mut ProtocolReader<tokio::io::Stdin>,
) -> Result<WorkerStartupConfiguration, Box<dyn std::error::Error + Send + Sync>> {
    match tokio::time::timeout(STARTUP_COMMAND_TIMEOUT, command_reader.next_command()).await {
        Ok(Ok(Some(WorkerCommand::InitializeWorker(startup_configuration)))) => {
            Ok(startup_configuration)
        }
        Ok(Ok(Some(unexpected_first_command))) => Err(std::io::Error::other(format!(
            "unexpected first worker command: {unexpected_first_command:?}"
        ))
        .into()),
        // Unlike fixtures a bare test daemon may launch, this fixture cannot
        // role-play without its launch configuration: the generation selects
        // the scripted role and the logging directory carries the ordering
        // evidence. Diverging silently would strain every downstream branch.
        Ok(Ok(None)) => {
            Err("the daemon closed the worker input before sending InitializeWorker".into())
        }
        Ok(Err(protocol_error)) => Err(protocol_error.into()),
        Err(_) => Err(std::io::Error::other(format!(
            "the daemon did not send InitializeWorker within {STARTUP_COMMAND_TIMEOUT:?}"
        ))
        .into()),
    }
}

fn runtime_configuration(
    startup_configuration: &WorkerStartupConfiguration,
    acknowledged_generation: &str,
    loaded_model: Option<WorkerLoadedModelRuntimeConfiguration>,
) -> WorkerRuntimeFeatureConfiguration {
    WorkerRuntimeFeatureConfiguration {
        configuration_generation: acknowledged_generation.to_owned(),
        persistent_prompt_cache_enabled: startup_configuration.persistent_prompt_cache_enabled,
        prompt_cache_maximum_size_bytes: startup_configuration
            .global_prompt_cache_maximum_size_bytes,
        loaded_model,
    }
}

fn chat_capabilities(
    autoregressive_configuration: Option<&WorkerAutoregressiveModelConfiguration>,
) -> WorkerModelCapabilities {
    match autoregressive_configuration {
        Some(autoregressive_configuration) => ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: false,
            max_input_tokens: autoregressive_configuration
                .maximum_context_tokens
                .saturating_sub(1),
            max_output_tokens: autoregressive_configuration.maximum_output_tokens,
            context_window: autoregressive_configuration.maximum_context_tokens,
        }
        .into(),
        None => WorkerModelCapabilities::image_generation(ImageGenerationCapabilities {
            minimum_width_pixels: 64,
            maximum_width_pixels: 1_024,
            minimum_height_pixels: 64,
            maximum_height_pixels: 1_024,
            dimension_multiple_pixels: 16,
            maximum_steps: 50,
            maximum_guidance_thousandths: 10_000,
            output_mime_types: vec!["image/png".to_owned()],
        }),
    }
}

async fn emit_memory_snapshot<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    source: MlxMemorySnapshotSource,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source,
                active_memory_bytes: 1_000_000_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 1_000_000_000,
                expert_payload_bytes: 0,
                model_core_payload_bytes: 0,
                context_state_payload_bytes: 0,
                memory_ceiling_utilization: None,
            }),
            expert_residency: None,
        })
        .await
}

/// Records this process's identity and what it observed about every sibling
/// worker already recorded in the shared logging directory.
///
/// A stop-first replacement observes its predecessor as gone. A candidate-first
/// replacement observes it as alive, which is the overlap the demotion contract
/// exists to forbid.
fn record_launch_observation(
    logging_directory: &Path,
    configuration_generation: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let own_identifier_file = logging_directory.join(format!(
        "{WORKER_PROCESS_IDENTIFIER_PREFIX}{configuration_generation}.pid"
    ));
    fs::write(&own_identifier_file, std::process::id().to_string())?;

    let mut observations = Vec::new();
    for entry in fs::read_dir(logging_directory)? {
        let entry = entry?;
        let entry_name = entry.file_name().to_string_lossy().into_owned();
        let Some(observed_generation) = entry_name
            .strip_prefix(WORKER_PROCESS_IDENTIFIER_PREFIX)
            .and_then(|name| name.strip_suffix(".pid"))
        else {
            continue;
        };
        if observed_generation == configuration_generation {
            continue;
        }
        let observed_identifier = fs::read_to_string(entry.path())?;
        observations.push(format!(
            "{observed_generation}={}",
            if process_is_running(observed_identifier.trim()) {
                "alive"
            } else {
                "gone"
            }
        ));
    }
    observations.sort();

    let mut log_entry = format!("{configuration_generation} observed");
    if observations.is_empty() {
        log_entry.push_str(" no-sibling");
    } else {
        for observation in observations {
            log_entry.push(' ');
            log_entry.push_str(&observation);
        }
    }
    log_entry.push('\n');

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logging_directory.join(LAUNCH_ORDER_LOG_FILE))?;
    log_file.write_all(log_entry.as_bytes())?;
    log_file.flush()?;
    Ok(())
}

fn process_is_running(process_identifier: &str) -> bool {
    if process_identifier.is_empty() {
        return false;
    }
    std::process::Command::new("kill")
        .args(["-0", process_identifier])
        .output()
        .is_ok_and(|process_probe| process_probe.status.success())
}
