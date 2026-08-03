#![forbid(unsafe_code)]

use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, MtpRuntimeState, ProtocolReader,
    ProtocolWriter, WorkerCommand, WorkerEvent,
};

#[tokio::main]
async fn main() {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    tokio::time::sleep(Duration::from_millis(500)).await;
    if event_writer
        .send_event(&WorkerEvent::Ready {
            expert_storage_format:
                astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            model_id: "astronomical/test-worker".to_owned(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: false,
                supports_tool_calls: false,
                has_vision: false,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
        })
        .await
        .is_err()
    {
        return;
    }
    while let Ok(Some(worker_command)) = command_reader.next_command().await {
        match worker_command {
            WorkerCommand::InitializeWorker(_) => {}
            WorkerCommand::Generate(generation_command) => {
                let _send_outcome = event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id: generation_command.request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        reason: ChatGenerationCompletionReason::EndOfSequence,
                    })
                    .await;
            }
            WorkerCommand::Cancel { request_id } => {
                let _send_outcome = event_writer
                    .send_event(&WorkerEvent::Completed {
                        request_id,
                        prompt_token_count: 1,
                        generated_token_count: 0,
                        reasoning_token_count: 0,
                        cached_token_count: 0,
                        reason: ChatGenerationCompletionReason::Cancelled,
                    })
                    .await;
            }
            WorkerCommand::SwapModel {
                model_directory, ..
            } => {
                eprintln!("test worker received SwapModel for {model_directory}; ignoring");
            }
            WorkerCommand::SampleMlxMemory => {}
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
            } => {
                let _send_outcome = event_writer
                    .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                        effective_mlx_memory_ceiling_bytes,
                        minimum_mlx_memory_ceiling_bytes: 1,
                        expert_memory_mode: astronomical_ipc_protocol::ExpertMemoryMode::Resident,
                        mlx_memory_snapshot: None,
                    })
                    .await;
            }
        }
    }
}
