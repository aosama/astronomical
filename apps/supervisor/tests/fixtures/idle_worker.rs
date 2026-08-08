use std::{error::Error, process::ExitCode};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, MlxMemorySnapshotSource,
    MtpRuntimeState, ProtocolReader, ProtocolWriter, SpeculativePrefillRuntimeState, WorkerCommand,
    WorkerEvent, WorkerMlxMemorySnapshot,
};

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

async fn run_fixture() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    event_writer
        .send_event(&WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            minimum_mlx_memory_ceiling_bytes: 1,
        })
        .await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        match worker_command {
            WorkerCommand::InitializeWorker(_) => {}
            WorkerCommand::SwapModel {
                model_directory, ..
            } => {
                if model_directory.ends_with("hanging-model") {
                    continue;
                } else if model_directory.ends_with("invalid-model") {
                    event_writer
                        .send_event(&WorkerEvent::ModelSwapFailed {
                            loaded_model_remains_ready: false,
                            model_load_failure_reason: "model artifact validation failed: OptiQ metadata uses unsupported 2-bit quantization".to_owned(),
                        })
                        .await?;
                } else {
                    event_writer
                        .send_event(&WorkerEvent::ModelSwapped {
                            minimum_mlx_memory_ceiling_bytes: 3_000_000_000,
                            mtp_runtime_state: MtpRuntimeState::Disabled,
                            mtp_unavailable_reason: None,
                            speculative_prefill_runtime_state:
                                SpeculativePrefillRuntimeState::Disabled,
                            speculative_prefill_unavailable_reason: None,
                            speculative_prefill_draft_model_id: None,
                            speculative_prefill_draft_model_revision: None,
                            model_id: "astronomical/requested-model".to_owned(),
                            capabilities: ChatModelCapabilities {
                                supports_reasoning: true,
                                supports_tool_calls: true,
                                has_vision: false,
                                max_input_tokens: 1_024,
                                max_output_tokens: 128,
                                context_window: 2_048,
                            },
                        })
                        .await?;
                    emit_model_loaded_memory_snapshot(&mut event_writer).await?;
                }
            }
            WorkerCommand::Generate(generation_command) => {
                event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id: generation_command.request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        reason: ChatGenerationCompletionReason::EndOfSequence,
                    })
                    .await?;
            }
            WorkerCommand::Cancel { .. } => {}
            WorkerCommand::SampleMlxMemory => {
                event_writer
                    .send_event(&WorkerEvent::MlxMemorySample {
                        mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                            source: MlxMemorySnapshotSource::IdlePoll,
                            active_memory_bytes: 20_000_000_000,
                            allocator_cache_memory_bytes: 0,
                            peak_memory_bytes: 20_000_000_000,
                            expert_payload_bytes: 12_000_000_000,
                            model_core_payload_bytes: 8_000_000_000,
                            context_state_payload_bytes: 0,
                        }),
                    })
                    .await?;
            }
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
            } => {
                if effective_mlx_memory_ceiling_bytes == 30_000_000_000 {
                    continue;
                }
                if effective_mlx_memory_ceiling_bytes == 31_000_000_000 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    event_writer
                        .send_event(&WorkerEvent::MlxMemoryLimitRejected {
                            requested_mlx_memory_ceiling_bytes: effective_mlx_memory_ceiling_bytes,
                            minimum_mlx_memory_ceiling_bytes: 1,
                            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
                            reason: "fixture rejected the requested limit".to_owned(),
                        })
                        .await?;
                    continue;
                }
                event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: 1,
                        expert_memory_mode: astronomical_ipc_protocol::ExpertMemoryMode::Resident,
                        mlx_memory_snapshot: None,
                    })
                    .await?;
            }
        }
    }
    Ok(())
}

async fn emit_model_loaded_memory_snapshot<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::ModelLoaded,
                active_memory_bytes: 20_000_000_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 20_000_000_000,
                expert_payload_bytes: 12_000_000_000,
                model_core_payload_bytes: 8_000_000_000,
                context_state_payload_bytes: 0,
            }),
        })
        .await
}
