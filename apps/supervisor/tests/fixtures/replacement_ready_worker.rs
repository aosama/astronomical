#![forbid(unsafe_code)]

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, MtpRuntimeState, ProtocolReader,
    ProtocolWriter, RequestId, SpeculativePrefillRuntimeState, WorkerEvent,
    WorkerRuntimeFeatureConfiguration,
};

const CONFIGURATION_BEFORE_READY_GENERATION: &str =
    "4444444444444444444444444444444444444444444444444444444444444444";
const GENERATION_EVENT_GENERATION: &str =
    "3333333333333333333333333333333333333333333333333333333333333333";
const INCONSISTENT_READY_GENERATION: &str =
    "5555555555555555555555555555555555555555555555555555555555555555";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let startup_configuration = match tokio::time::timeout(
        std::time::Duration::from_millis(100),
        command_reader.next_command(),
    )
    .await
    {
        Ok(Ok(Some(astronomical_ipc_protocol::WorkerCommand::InitializeWorker(configuration)))) => {
            Some(configuration)
        }
        Ok(Ok(Some(_))) => return Err("unexpected first worker command".into()),
        Ok(Err(protocol_error)) => return Err(protocol_error.into()),
        Ok(Ok(None)) | Err(_) => None,
    };
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
    if startup_configuration.as_ref().is_some_and(|configuration| {
        configuration.configuration_generation == INCONSISTENT_READY_GENERATION
    }) {
        event_writer
            .send_event(&WorkerEvent::Ready {
                mtp_runtime_state: MtpRuntimeState::Disabled,
                mtp_unavailable_reason: None,
                mtp_depth_status: Default::default(),
                speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
                speculative_prefill_unavailable_reason: None,
                speculative_prefill_draft_model_id: None,
                speculative_prefill_draft_model_revision: None,
                model_id: "astronomical/unacknowledged-ready-model".to_owned(),
                capabilities: ChatModelCapabilities {
                    supports_reasoning: false,
                    supports_tool_calls: false,
                    has_vision: false,
                    max_input_tokens: 1,
                    max_output_tokens: 1,
                    context_window: 2,
                },
            })
            .await?;
    } else {
        event_writer
            .send_event(&WorkerEvent::Idle {
                machine_mlx_memory_ceiling_bytes: 40_000_000_000,
                effective_mlx_memory_ceiling_bytes: 40_000_000_000,
                minimum_mlx_memory_ceiling_bytes: 1,
            })
            .await?;
    }
    if let Some(startup_configuration) = startup_configuration.as_ref() {
        if startup_configuration.configuration_generation == GENERATION_EVENT_GENERATION {
            event_writer
                .send_event(&WorkerEvent::Completed {
                    request_id: RequestId::new(1),
                    prompt_token_count: 1,
                    generated_token_count: 0,
                    reasoning_token_count: 0,
                    cached_token_count: 0,
                    persistent_prompt_cache_diagnostics: None,
                    reason: ChatGenerationCompletionReason::EndOfSequence,
                })
                .await?;
        } else if startup_configuration.configuration_generation == INCONSISTENT_READY_GENERATION {
            send_runtime_configuration(
                &mut event_writer,
                startup_configuration,
                INCONSISTENT_READY_GENERATION,
            )
            .await?;
        } else if startup_configuration.configuration_generation
            != CONFIGURATION_BEFORE_READY_GENERATION
        {
            send_runtime_configuration(
                &mut event_writer,
                startup_configuration,
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            )
            .await?;
        }
    }

    while command_reader.next_command().await?.is_some() {}
    Ok(())
}

async fn send_runtime_configuration<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    startup_configuration: &astronomical_ipc_protocol::WorkerStartupConfiguration,
    acknowledged_generation: &str,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
                // This fixture proves replacement acknowledgement is identity-bound rather
                // than inferred from process readiness or otherwise matching policy fields.
                configuration_generation: acknowledged_generation.to_owned(),
                persistent_prompt_cache_enabled: startup_configuration
                    .persistent_prompt_cache_enabled,
                prompt_cache_maximum_size_bytes: startup_configuration
                    .global_prompt_cache_maximum_size_bytes,
                loaded_model: None,
            },
        })
        .await
}
