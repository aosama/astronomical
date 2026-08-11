//! Deterministic subprocess fixture for worker lifecycle and event-order tests.
//!
//! Model-directory suffixes select scripted protocol behavior. The fixture uses
//! the real framed standard-input/standard-output transport, so tests exercise
//! supervisor queueing and event handling without loading MLX or local artifacts.

use std::{error::Error, process::ExitCode};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, ExpertMemoryMode,
    MlxMemorySnapshotSource, MtpRuntimeState, ProtocolReader, ProtocolWriter, RequestId,
    SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent, WorkerMlxMemorySnapshot,
    WorkerRuntimeFeatureConfiguration,
};

const DELAYED_COMPLETION_MODEL_ID: &str = "astronomical/delayed-completion-model";
const GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: &str =
    "astronomical/generation-event-before-swap-model";
const REQUESTED_MODEL_ID: &str = "astronomical/requested-model";
const TELEMETRY_BEFORE_SWAP_MODEL_ID: &str = "astronomical/telemetry-before-swap-model";

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
    let mut loaded_model_id: Option<String> = None;
    event_writer
        .send_event(&WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            minimum_mlx_memory_ceiling_bytes: 1,
        })
        .await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        match worker_command {
            WorkerCommand::InitializeWorker(worker_startup_configuration) => {
                event_writer
                    .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
                            persistent_prompt_cache_enabled: worker_startup_configuration
                                .persistent_prompt_cache_enabled,
                            mtp_enabled: worker_startup_configuration.mtp_enabled,
                            speculative_prefill_enabled: worker_startup_configuration
                                .speculative_prefill
                                .enabled,
                        },
                    })
                    .await?;
            }
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
                    let replacement_model_id = model_id_for_directory(&model_directory);
                    // These two branches deliberately violate the old assumption
                    // that ModelSwapped is always the next frame after SwapModel.
                    if replacement_model_id == TELEMETRY_BEFORE_SWAP_MODEL_ID {
                        emit_memory_snapshot(&mut event_writer, MlxMemorySnapshotSource::IdlePoll)
                            .await?;
                    } else if replacement_model_id == GENERATION_EVENT_BEFORE_SWAP_MODEL_ID {
                        event_writer
                            .send_event(&WorkerEvent::Completed {
                                request_id: RequestId::new(999),
                                prompt_token_count: 1,
                                generated_token_count: 0,
                                reasoning_token_count: 0,
                                cached_token_count: 0,
                                persistent_prompt_cache_diagnostics: None,
                                reason: ChatGenerationCompletionReason::EndOfSequence,
                            })
                            .await?;
                    }
                    event_writer
                        .send_event(&WorkerEvent::ModelSwapped {
                            expert_memory_mode: Some(ExpertMemoryMode::Resident),
                            minimum_mlx_memory_ceiling_bytes: 3_000_000_000,
                            mtp_runtime_state: MtpRuntimeState::Disabled,
                            mtp_unavailable_reason: None,
                            speculative_prefill_runtime_state:
                                SpeculativePrefillRuntimeState::Disabled,
                            speculative_prefill_unavailable_reason: None,
                            speculative_prefill_draft_model_id: None,
                            speculative_prefill_draft_model_revision: None,
                            model_id: replacement_model_id.to_owned(),
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
                    loaded_model_id = Some(replacement_model_id.to_owned());
                    emit_memory_snapshot(&mut event_writer, MlxMemorySnapshotSource::ModelLoaded)
                        .await?;
                }
            }
            WorkerCommand::Generate(generation_command) => {
                if loaded_model_id.as_deref() == Some(DELAYED_COMPLETION_MODEL_ID) {
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
                event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id: generation_command.request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        persistent_prompt_cache_diagnostics: None,
                        reason: ChatGenerationCompletionReason::EndOfSequence,
                    })
                    .await?;
            }
            WorkerCommand::Cancel { .. } => {}
            WorkerCommand::SampleMlxMemory => {
                emit_memory_snapshot(&mut event_writer, MlxMemorySnapshotSource::IdlePoll).await?;
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

fn model_id_for_directory(model_directory: &str) -> &'static str {
    if model_directory.ends_with("delayed-completion-model") {
        DELAYED_COMPLETION_MODEL_ID
    } else if model_directory.ends_with("telemetry-before-swap-model") {
        TELEMETRY_BEFORE_SWAP_MODEL_ID
    } else if model_directory.ends_with("generation-event-before-swap-model") {
        GENERATION_EVENT_BEFORE_SWAP_MODEL_ID
    } else {
        REQUESTED_MODEL_ID
    }
}

async fn emit_memory_snapshot<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    mlx_memory_snapshot_source: MlxMemorySnapshotSource,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: mlx_memory_snapshot_source,
                active_memory_bytes: 20_000_000_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 20_000_000_000,
                expert_payload_bytes: 12_000_000_000,
                model_core_payload_bytes: 8_000_000_000,
                context_state_payload_bytes: 0,
                speculative_prefill_draft_memory_bytes: 0,
            }),
        })
        .await
}
