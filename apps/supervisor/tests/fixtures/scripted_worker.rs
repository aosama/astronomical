use std::{error::Error, process::ExitCode, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatModelCapabilities, ExpertMemoryMode, MlxMemorySnapshotSource, MtpRuntimeState,
    ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent, WorkerMlxMemorySnapshot,
};

const READY_MODEL_ID_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID";
const DEFAULT_READY_MODEL_ID: &str = "astronomical/test-worker";

#[tokio::main]
async fn main() -> ExitCode {
    match run_fixture().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(fixture_error) => {
            eprintln!("test worker failed: {fixture_error}");
            ExitCode::FAILURE
        }
    }
}

async fn run_fixture() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let mut active_request_id = None;
    let mut cancellation_acknowledgement_delay = Duration::ZERO;
    let mut should_acknowledge_cancellation = true;
    let ready_model_id = std::env::var(READY_MODEL_ID_ENVIRONMENT_VARIABLE)
        .unwrap_or_else(|_| DEFAULT_READY_MODEL_ID.to_owned());
    event_writer
        .send_event(&WorkerEvent::Ready {
            expert_storage_format:
                astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            model_id: ready_model_id,
            capabilities: ChatModelCapabilities {
                supports_reasoning: true,
                supports_tool_calls: true,
                has_vision: true,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
        })
        .await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        match worker_command {
            WorkerCommand::InitializeWorker(_) => {}
            WorkerCommand::Generate(generation_command) => {
                let request_id = generation_command.request_id;
                match generation_command.model.as_str() {
                    "astronomical/accepted-chat-fixture" => {
                        send_accepted_chat(request_id, &mut event_writer).await?;
                    }
                    "astronomical/malformed-output-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Failed {
                                request_id,
                                reason: ChatGenerationFailureReason::MalformedModelOutput,
                            })
                            .await?;
                    }
                    "astronomical/delayed-malformed-output-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::GenerationProgress {
                                request_id,
                                generated_token_count: 3,
                                maximum_output_tokens: generation_command
                                    .settings
                                    .max_output_tokens,
                                elapsed_millis: 250,
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                        tokio::time::sleep(Duration::from_millis(750)).await;
                        event_writer
                            .send_event(&WorkerEvent::Failed {
                                request_id,
                                reason: ChatGenerationFailureReason::MalformedModelOutput,
                            })
                            .await?;
                    }
                    "astronomical/deadline-expiring-chat-fixture"
                    | "astronomical/delayed-fragment-chat-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                    }
                    "astronomical/unacknowledged-cancellation-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = false;
                    }
                    "astronomical/delayed-cancellation-acknowledgement-fixture" => {
                        active_request_id = Some(request_id);
                        cancellation_acknowledgement_delay = Duration::from_secs(4);
                        should_acknowledge_cancellation = true;
                    }
                    "astronomical/activity-transition-fixture" => {
                        send_activity_transition(request_id, &mut event_writer).await?;
                    }
                    "astronomical/out-of-order-chat-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Output {
                                request_id,
                                sequence_number: 1,
                                generated_token_count: 1,
                                outputs: vec![ChatGenerationOutput::Text {
                                    text: "out of order".to_owned(),
                                }],
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                    }
                    "astronomical/empty-output-batch-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Output {
                                request_id,
                                sequence_number: 0,
                                generated_token_count: 1,
                                outputs: Vec::new(),
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id,
                                prompt_token_count: 1,
                                generated_token_count: 1,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                reason: ChatGenerationCompletionReason::EndOfSequence,
                            })
                            .await?;
                    }
                    "astronomical/invalid-output-batch-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Output {
                                request_id,
                                sequence_number: 0,
                                generated_token_count: 1,
                                outputs: vec![
                                    ChatGenerationOutput::Text {
                                        text: "must not escape malformed batch".to_owned(),
                                    },
                                    ChatGenerationOutput::ToolCall {
                                        tool_call_index: 1,
                                        function_name: "read".to_owned(),
                                        arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
                                    },
                                ],
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                    }
                    "astronomical/over-budget-tool-completion-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Output {
                                request_id,
                                sequence_number: 0,
                                generated_token_count: 1,
                                outputs: vec![ChatGenerationOutput::ToolCall {
                                    tool_call_index: 0,
                                    function_name: "read".to_owned(),
                                    arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
                                }],
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id,
                                prompt_token_count: 1,
                                generated_token_count: 17,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                reason: ChatGenerationCompletionReason::ToolCalls,
                            })
                            .await?;
                    }
                    "astronomical/unsolicited-cancellation-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id,
                                prompt_token_count: 1,
                                generated_token_count: 0,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                reason: ChatGenerationCompletionReason::Cancelled,
                            })
                            .await?;
                    }
                    "astronomical/backpressure-fixture" => {
                        for sequence_number in 0..64_u16 {
                            event_writer
                                .send_event(&WorkerEvent::Output {
                                    request_id,
                                    sequence_number,
                                    generated_token_count: 1,
                                    outputs: vec![ChatGenerationOutput::Text {
                                        text: "fragment".to_owned(),
                                    }],
                                    mlx_memory_snapshot: None,
                                })
                                .await?;
                        }
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id,
                                prompt_token_count: 1,
                                generated_token_count: 1,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                reason: ChatGenerationCompletionReason::MaximumOutputTokens,
                            })
                            .await?;
                    }
                    "astronomical/prefill-progress-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::PrefillProgress {
                                request_id,
                                processed_tokens: 2_048,
                                total_tokens: 50_000,
                                elapsed_millis: 1_500,
                                forward_prefill_chunck_elapsed_millis: Some(1_400),
                                completed_prefill_chunck_tokens: Some(2_048),
                                mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                                    source: MlxMemorySnapshotSource::Prefill,
                                    active_memory_bytes: 11_000,
                                    allocator_cache_memory_bytes: 12_000,
                                    peak_memory_bytes: 13_000,
                                    expert_payload_bytes: 4_000,
                                    model_core_payload_bytes: 3_000,
                                    context_state_payload_bytes: 2_000,
                                }),
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::Output {
                                request_id,
                                sequence_number: 0,
                                generated_token_count: 1,
                                outputs: vec![ChatGenerationOutput::Text {
                                    text: "done".to_owned(),
                                }],
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::GenerationFinalized {
                                request_id,
                                expert_memory_mode: Some(ExpertMemoryMode::Resident),
                                mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                                    source: MlxMemorySnapshotSource::Finalized,
                                    active_memory_bytes: 24_000,
                                    allocator_cache_memory_bytes: 0,
                                    peak_memory_bytes: 25_000,
                                    expert_payload_bytes: 19_000,
                                    model_core_payload_bytes: 3_000,
                                    context_state_payload_bytes: 0,
                                }),
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id,
                                prompt_token_count: 50_000,
                                generated_token_count: 1,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                reason: ChatGenerationCompletionReason::EndOfSequence,
                            })
                            .await?;
                    }
                    "astronomical/exit-after-chat-admission-fixture" => return Ok(()),
                    _ => send_simple_completion(request_id, &mut event_writer).await?,
                }
            }
            WorkerCommand::Cancel { request_id } => {
                if active_request_id == Some(request_id) {
                    active_request_id = None;
                }
                if !should_acknowledge_cancellation {
                    continue;
                }
                tokio::time::sleep(cancellation_acknowledgement_delay).await;
                cancellation_acknowledgement_delay = Duration::ZERO;
                event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        reason: ChatGenerationCompletionReason::Cancelled,
                    })
                    .await?;
            }
            WorkerCommand::SwapModel {
                model_directory, ..
            } => {
                tracing::warn!(
                    model_directory,
                    "test scripted worker received SwapModel; ignoring"
                );
            }
            WorkerCommand::SampleMlxMemory => {}
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
            } => {
                event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: 1,
                        expert_memory_mode: ExpertMemoryMode::Resident,
                        mlx_memory_snapshot: None,
                    })
                    .await?;
            }
        }
    }
    Ok(())
}

async fn send_accepted_chat<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    for (sequence_number, generated_token_count, outputs) in [
        (
            0,
            1,
            vec![ChatGenerationOutput::Reasoning {
                text: "accepted chat reasoning".to_owned(),
            }],
        ),
        (
            1,
            1,
            vec![ChatGenerationOutput::Text {
                text: "accepted chat text".to_owned(),
            }],
        ),
        (
            2,
            3,
            vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 0,
                function_name: "read".to_owned(),
                arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
            }],
        ),
        (
            3,
            4,
            vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 1,
                function_name: "glob".to_owned(),
                arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
            }],
        ),
    ] {
        event_writer
            .send_event(&WorkerEvent::Output {
                request_id,
                sequence_number,
                generated_token_count,
                outputs,
                mlx_memory_snapshot: None,
            })
            .await?;
    }
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 2,
            generated_token_count: 4,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        })
        .await
}

async fn send_simple_completion<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}

async fn send_activity_transition<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: "activity transition".to_owned(),
            }],
            mlx_memory_snapshot: None,
        })
        .await?;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}
