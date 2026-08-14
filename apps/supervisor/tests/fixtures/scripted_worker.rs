use std::{error::Error, process::ExitCode, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatModelCapabilities, ExpertMemoryMode, MlxMemorySnapshotSource, MtpRuntimeState,
    ProtocolReader, ProtocolWriter, RequestId, SpeculativePrefillRuntimeState, WorkerCommand,
    WorkerEvent, WorkerMlxMemorySnapshot, WorkerPrefillOptimizerCandidateEvidence,
    WorkerPrefillOptimizerContext, WorkerPrefillOptimizerDecisionReason,
    WorkerPrefillOptimizerInsight, WorkerPromptProcessingPhase,
};

mod scripted_worker_chat;

use scripted_worker_chat::{send_accepted_chat, send_activity_transition, send_simple_completion};

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
    let mut should_emit_unexpected_cancellation_event = false;
    let ready_model_id = std::env::var(READY_MODEL_ID_ENVIRONMENT_VARIABLE)
        .unwrap_or_else(|_| DEFAULT_READY_MODEL_ID.to_owned());
    event_writer
        .send_event(&WorkerEvent::Ready {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
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
                    "astronomical/unexpected-cancellation-event-fixture" => {
                        active_request_id = Some(request_id);
                        should_acknowledge_cancellation = true;
                        should_emit_unexpected_cancellation_event = true;
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
                                forward_prefill_chunck_elapsed_millis: Some(1_400),
                                completed_prefill_chunck_tokens: Some(2_048),
                                prefill_optimizer_insight: Some(WorkerPrefillOptimizerInsight {
                                    requested_prefill_chunck_tokens: 4_096,
                                    actual_prefill_chunck_tokens: 2_048,
                                    elapsed_millis: 1_500,
                                    decision_reason:
                                        WorkerPrefillOptimizerDecisionReason::InitialExploration,
                                    has_observed_prefill_capacity_constraint: true,
                                    has_observations_for_every_candidate: false,
                                    context: WorkerPrefillOptimizerContext {
                                        prompt_position_tokens: 0,
                                        has_restored_prefix: false,
                                        is_first_chunck_after_restore: false,
                                        has_visual_embeddings: false,
                                        is_mtp_active: false,
                                        are_sparse_experts_paged: true,
                                        is_prompt_cache_capture_eligible: true,
                                        has_prior_capacity_reduction: false,
                                    },
                                    candidate_evidence: vec![
                                        WorkerPrefillOptimizerCandidateEvidence {
                                            candidate_prefill_chunck_tokens: 4_096,
                                            observation_count: 1,
                                            average_actual_prefill_chunck_tokens: 2_048,
                                            average_elapsed_millis: 1_500,
                                            decisions_since_last_observation: Some(0),
                                        },
                                    ],
                                }),
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
                                speculative_prefill_draft_memory_snapshot: None,
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
