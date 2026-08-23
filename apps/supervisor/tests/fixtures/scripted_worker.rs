use std::{error::Error, process::ExitCode, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatModelCapabilities, ExpertMemoryMode, MlxMemorySnapshotSource, MtpRuntimeState,
    ProtocolReader, ProtocolWriter, RequestId, SpeculativePrefillRuntimeState, WorkerCommand,
    WorkerEvent, WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase,
};

mod scripted_worker_chat;
mod scripted_worker_image;

use scripted_worker_chat::{send_accepted_chat, send_activity_transition, send_simple_completion};
use scripted_worker_image::{
    ScriptedImageCommandOutcome, handle_image_command, image_capabilities, send_failed_image,
};

const READY_MODEL_ID_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID";
const DEFAULT_READY_MODEL_ID: &str = "astronomical/test-worker";

/// Selects the memory telemetry emitted before a cancellation acknowledgement.
enum CancellationMlxTelemetry {
    None,
    Publish,
    Clear,
}

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
    let mut should_emit_unexpected_cancellation_event = false;
    let mut should_emit_cache_stats_before_cancellation_acknowledgement = false;
    let mut cancellation_mlx_telemetry = CancellationMlxTelemetry::None;
    let ready_model_id = std::env::var(READY_MODEL_ID_ENVIRONMENT_VARIABLE)
        .unwrap_or_else(|_| DEFAULT_READY_MODEL_ID.to_owned());
    event_writer
        .send_event(&WorkerEvent::Ready {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            mtp_depth_status: Default::default(),
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            model_id: ready_model_id,
            capabilities: astronomical_ipc_protocol::WorkerModelCapabilities::chat_and_image(
                ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                },
                image_capabilities(),
            ),
        })
        .await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        match worker_command {
            WorkerCommand::InitializeWorker(_) => {}
            WorkerCommand::GenerateImage(generation_command) => {
                let request_id = generation_command.request_id;
                if let ScriptedImageCommandOutcome::CancellationPending { should_acknowledge } =
                    handle_image_command(generation_command, &mut event_writer).await?
                {
                    active_request_id = Some(request_id);
                    should_acknowledge_cancellation = should_acknowledge;
                }
            }
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
                    "astronomical/unexpected-cancellation-event-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                        should_emit_unexpected_cancellation_event = true;
                    }
                    "astronomical/cache-stats-during-cancellation-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                        should_emit_cache_stats_before_cancellation_acknowledgement = true;
                    }
                    "astronomical/mlx-memory-during-cancellation-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                        cancellation_mlx_telemetry = CancellationMlxTelemetry::Publish;
                    }
                    "astronomical/mlx-memory-clear-during-cancellation-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                        // Seed a visible snapshot so the cancellation event can
                        // prove that `None` clears previously published memory.
                        event_writer
                            .send_event(&WorkerEvent::MlxMemorySample {
                                mlx_memory_snapshot: Some(cancellation_memory_snapshot(33_000)),
                            })
                            .await?;
                        cancellation_mlx_telemetry = CancellationMlxTelemetry::Clear;
                    }
                    "astronomical/activity-transition-fixture" => {
                        send_activity_transition(request_id, &mut event_writer).await?;
                    }
                    "astronomical/duplicate-generation-preparation-fixture" => {
                        let generation_preparation_event =
                            WorkerEvent::GenerationPreparationStarted {
                                request_id,
                                total_layer_count: 40,
                                complete_layer_count: 25,
                                complete_layer_payload_bytes: 22_649_241_600,
                                partial_layer_count: 15,
                                partial_layer_payload_bytes: 424_673_280,
                            };
                        event_writer
                            .send_event(&generation_preparation_event)
                            .await?;
                        event_writer
                            .send_event(&generation_preparation_event)
                            .await?;
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
                                persistent_prompt_cache_diagnostics: None,
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
                                persistent_prompt_cache_diagnostics: None,
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
                                persistent_prompt_cache_diagnostics: None,
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
                                persistent_prompt_cache_diagnostics: None,
                                reason: ChatGenerationCompletionReason::MaximumOutputTokens,
                            })
                            .await?;
                    }
                    "astronomical/prefill-progress-fixture" => {
                        event_writer
                            .send_event(&WorkerEvent::PrefillProgress {
                                request_id,
                                prompt_processing_phase: WorkerPromptProcessingPhase::Target,
                                processed_tokens: 2_048,
                                total_tokens: 50_000,
                                elapsed_millis: 1_500,
                                forward_prefill_chunk_elapsed_millis: Some(1_400),
                                completed_prefill_chunk_tokens: Some(2_048),
                                mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                                    source: MlxMemorySnapshotSource::Prefill,
                                    active_memory_bytes: 11_000,
                                    allocator_cache_memory_bytes: 12_000,
                                    peak_memory_bytes: 13_000,
                                    expert_payload_bytes: 4_000,
                                    model_core_payload_bytes: 3_000,
                                    context_state_payload_bytes: 2_000,
                                    speculative_prefill_draft_memory_bytes: 0,
                                }),
                                expert_residency: None,
                                speculative_prefill_draft_memory_snapshot: None,
                            })
                            .await?;
                        event_writer
                            .send_event(&WorkerEvent::FirstDecodeCompleted {
                                request_id,
                                elapsed_millis: 321,
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
                                expert_residency: None,
                                mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                                    source: MlxMemorySnapshotSource::Finalized,
                                    active_memory_bytes: 24_000,
                                    allocator_cache_memory_bytes: 0,
                                    peak_memory_bytes: 25_000,
                                    expert_payload_bytes: 19_000,
                                    model_core_payload_bytes: 3_000,
                                    context_state_payload_bytes: 0,
                                    speculative_prefill_draft_memory_bytes: 0,
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
                                persistent_prompt_cache_diagnostics: None,
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
                if should_emit_unexpected_cancellation_event {
                    should_emit_unexpected_cancellation_event = false;
                    event_writer
                        .send_event(&WorkerEvent::Completed {
                            request_id: RequestId::new(2),
                            prompt_token_count: 1,
                            generated_token_count: 0,
                            reasoning_token_count: 0,
                            cached_token_count: 0,
                            persistent_prompt_cache_diagnostics: None,
                            reason: ChatGenerationCompletionReason::Cancelled,
                        })
                        .await?;
                    continue;
                }
                if should_emit_cache_stats_before_cancellation_acknowledgement {
                    should_emit_cache_stats_before_cancellation_acknowledgement = false;
                    event_writer
                        .send_event(&WorkerEvent::PersistentPromptCacheStats {
                            persistent_prompt_cache_hits: 1,
                            persistent_prompt_cache_misses: 0,
                            persistent_prompt_cache_tokens_saved: 2_048,
                            persistent_prompt_cache_block_token_count: 2_048,
                            persistent_prompt_cache_sequence_state_block_count: 1,
                            persistent_prompt_cache_boundary_state_snapshot_count: 1,
                            persistent_prompt_cache_visual_embedding_count: 0,
                            persistent_prompt_cache_total_size_bytes: 4_096,
                            persistent_prompt_cache_visual_embedding_total_size_bytes: 0,
                            persistent_prompt_cache_maximum_size_bytes: 50_000_000_000,
                            persistent_prompt_cache_visual_embedding_hits: 0,
                            persistent_prompt_cache_visual_embedding_misses: 0,
                            persistent_prompt_cache_visual_embedding_rows_loaded: 0,
                        })
                        .await?;
                }
                match std::mem::replace(
                    &mut cancellation_mlx_telemetry,
                    CancellationMlxTelemetry::None,
                ) {
                    CancellationMlxTelemetry::None => {}
                    CancellationMlxTelemetry::Publish => {
                        event_writer
                            .send_event(&WorkerEvent::MlxMemorySample {
                                mlx_memory_snapshot: Some(cancellation_memory_snapshot(44_000)),
                            })
                            .await?;
                    }
                    CancellationMlxTelemetry::Clear => {
                        event_writer
                            .send_event(&WorkerEvent::MlxMemorySample {
                                mlx_memory_snapshot: None,
                            })
                            .await?;
                    }
                }
                if should_acknowledge_cancellation && request_id.value() >= 100 {
                    send_failed_image(request_id, &mut event_writer).await?;
                    continue;
                }
                event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        persistent_prompt_cache_diagnostics: None,
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
                configuration_generation: _,
            } => {
                event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: 1,
                        expert_memory_mode: ExpertMemoryMode::Resident,
                        mlx_memory_snapshot: None,
                        expert_residency: None,
                    })
                    .await?;
            }
            WorkerCommand::ClearPromptCache { model_id } => {
                event_writer
                    .send_event(&WorkerEvent::PromptCacheCleared {
                        model_id,
                        blocks_removed: 0,
                        bytes_freed: 0,
                    })
                    .await?;
            }
        }
    }
    Ok(())
}

/// Builds deterministic memory telemetry for cancellation-containment tests.
fn cancellation_memory_snapshot(active_memory_bytes: u64) -> WorkerMlxMemorySnapshot {
    WorkerMlxMemorySnapshot {
        source: MlxMemorySnapshotSource::Finalized,
        active_memory_bytes,
        allocator_cache_memory_bytes: 2_000,
        peak_memory_bytes: active_memory_bytes + 1_000,
        expert_payload_bytes: 20_000,
        model_core_payload_bytes: 10_000,
        context_state_payload_bytes: 5_000,
        speculative_prefill_draft_memory_bytes: 0,
    }
}
