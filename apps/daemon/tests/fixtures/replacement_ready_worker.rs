#![forbid(unsafe_code)]

use ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, ImageGenerationCapabilities,
    ProtocolReader, ProtocolWriter, RequestId, WorkerEvent,
    WorkerImageGenerationModelConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerModelCapabilities, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};

const CONFIGURATION_BEFORE_READY_GENERATION: &str =
    "4444444444444444444444444444444444444444444444444444444444444444";
const GENERATION_EVENT_GENERATION: &str =
    "3333333333333333333333333333333333333333333333333333333333333333";
const INCONSISTENT_READY_GENERATION: &str =
    "5555555555555555555555555555555555555555555555555555555555555555";
const MATCHING_FLUX_GENERATION: &str =
    "6666666666666666666666666666666666666666666666666666666666666666";
const FLUX_MODEL_ID: &str = "FLUX.2-klein-4B";
const FLUX_REVISION: &str = "reviewed-revision";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let startup_configuration = read_optional_startup_configuration(&mut command_reader).await?;
    if let Some(startup_configuration) = startup_configuration.as_ref() {
        std::fs::write(
            startup_configuration
                .logging_directory
                .join("replacement-candidate.pid"),
            std::process::id().to_string(),
        )?;
        if startup_configuration.configuration_generation == CONFIGURATION_BEFORE_READY_GENERATION {
            send_runtime_configuration(
                &mut event_writer,
                startup_configuration,
                CONFIGURATION_BEFORE_READY_GENERATION,
            )
            .await?;
        }
    }
    send_readiness_event(&mut event_writer, startup_configuration.as_ref()).await?;
    if let Some(startup_configuration) = startup_configuration.as_ref() {
        send_acknowledgement_after_readiness(&mut event_writer, startup_configuration).await?;
    }

    while command_reader.next_command().await?.is_some() {}
    Ok(())
}

async fn read_optional_startup_configuration(
    command_reader: &mut ProtocolReader<tokio::io::Stdin>,
) -> Result<Option<WorkerStartupConfiguration>, Box<dyn std::error::Error + Send + Sync>> {
    match tokio::time::timeout(
        std::time::Duration::from_millis(100),
        command_reader.next_command(),
    )
    .await
    {
        Ok(Ok(Some(ipc_protocol::WorkerCommand::InitializeWorker(configuration)))) => {
            Ok(Some(configuration))
        }
        Ok(Ok(Some(_))) => Err("unexpected first worker command".into()),
        Ok(Err(protocol_error)) => Err(protocol_error.into()),
        Ok(Ok(None)) | Err(_) => Ok(None),
    }
}

/// Whether the startup configuration the worker received carries this
/// configuration generation.
fn has_configuration_generation(
    startup_configuration: Option<&WorkerStartupConfiguration>,
    configuration_generation: &str,
) -> bool {
    matches!(
        startup_configuration,
        Some(configuration) if configuration.configuration_generation == configuration_generation
    )
}

/// The first readiness event, chosen by the generation the replacement
/// candidate was started with: a FLUX model, a chat model whose readiness
/// disagrees with its acknowledgement, or plain idleness.
async fn send_readiness_event(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    startup_configuration: Option<&WorkerStartupConfiguration>,
) -> Result<(), ipc_protocol::ProtocolError> {
    if has_configuration_generation(startup_configuration, MATCHING_FLUX_GENERATION) {
        event_writer
            .send_event(&WorkerEvent::Ready {
                model_id: FLUX_MODEL_ID.to_owned(),
                capabilities: WorkerModelCapabilities::image_generation(
                    ImageGenerationCapabilities {
                        minimum_width_pixels: 64,
                        maximum_width_pixels: 1_024,
                        minimum_height_pixels: 64,
                        maximum_height_pixels: 1_024,
                        dimension_multiple_pixels: 16,
                        maximum_steps: 50,
                        maximum_guidance_thousandths: 10_000,
                        output_mime_types: vec!["image/png".to_owned()],
                    },
                ),
            })
            .await
    } else if has_configuration_generation(startup_configuration, INCONSISTENT_READY_GENERATION) {
        event_writer
            .send_event(&WorkerEvent::Ready {
                model_id: "astronomical/unacknowledged-ready-model".to_owned(),
                capabilities: ChatModelCapabilities {
                    supports_reasoning: false,
                    supports_tool_calls: false,
                    has_vision: false,
                    max_input_tokens: 1,
                    max_output_tokens: 1,
                    context_window: 2,
                }
                .into(),
            })
            .await
    } else {
        event_writer
            .send_event(&WorkerEvent::Idle {
                machine_mlx_memory_ceiling_bytes: 40_000_000_000,
                effective_mlx_memory_ceiling_bytes: 40_000_000_000,
                minimum_mlx_memory_ceiling_bytes: 1,
                wired_residency_budget_bytes: None,
            })
            .await
    }
}

/// The event that follows readiness, chosen by the generation the
/// replacement candidate was started with.
async fn send_acknowledgement_after_readiness(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    startup_configuration: &WorkerStartupConfiguration,
) -> Result<(), ipc_protocol::ProtocolError> {
    if startup_configuration.configuration_generation == GENERATION_EVENT_GENERATION {
        event_writer
            .send_event(&WorkerEvent::Completed {
                request_id: RequestId::new(1),
                prompt_token_count: 1,
                generated_token_count: 0,
                reasoning_token_count: 0,
                astro_ssd_prompt_cache_restored_token_count: 0,
                astro_ssd_prompt_cache_diagnostics: None,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            })
            .await
    } else if startup_configuration.configuration_generation == MATCHING_FLUX_GENERATION {
        event_writer
            .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
                    configuration_generation: MATCHING_FLUX_GENERATION.to_owned(),
                    astro_ssd_prompt_cache_enabled: startup_configuration
                        .astro_ssd_prompt_cache_enabled,
                    astro_ssd_prompt_cache_maximum_size_bytes: startup_configuration
                        .global_astro_ssd_prompt_cache_maximum_size_bytes,
                    loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::ImageGeneration(
                        WorkerImageGenerationModelConfiguration {
                            model_id: FLUX_MODEL_ID.to_owned(),
                            model_family: "flux2_klein".to_owned(),
                            artifact_revision: FLUX_REVISION.to_owned(),
                        },
                    )),
                },
            })
            .await
    } else if startup_configuration.configuration_generation == INCONSISTENT_READY_GENERATION {
        send_runtime_configuration(
            event_writer,
            startup_configuration,
            INCONSISTENT_READY_GENERATION,
        )
        .await
    } else if startup_configuration.configuration_generation
        != CONFIGURATION_BEFORE_READY_GENERATION
    {
        send_runtime_configuration(
            event_writer,
            startup_configuration,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
        .await
    } else {
        Ok(())
    }
}

async fn send_runtime_configuration<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    startup_configuration: &ipc_protocol::WorkerStartupConfiguration,
    acknowledged_generation: &str,
) -> Result<(), ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
                // This fixture proves replacement acknowledgement is identity-bound rather
                // than inferred from process readiness or otherwise matching policy fields.
                configuration_generation: acknowledged_generation.to_owned(),
                astro_ssd_prompt_cache_enabled: startup_configuration
                    .astro_ssd_prompt_cache_enabled,
                astro_ssd_prompt_cache_maximum_size_bytes: startup_configuration
                    .global_astro_ssd_prompt_cache_maximum_size_bytes,
                loaded_model: None,
            },
        })
        .await
}
