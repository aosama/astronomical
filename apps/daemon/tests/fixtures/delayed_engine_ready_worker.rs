#![forbid(unsafe_code)]

use std::time::Duration;

use ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, ProtocolError, ProtocolReader,
    ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};

type WorkerEventWriter = ProtocolWriter<tokio::io::Stdout>;

#[tokio::main]
async fn main() {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    tokio::time::sleep(Duration::from_millis(500)).await;
    if send_ready_event(&mut event_writer).await.is_err() {
        return;
    }
    while let Ok(Some(worker_command)) = command_reader.next_command().await {
        answer_worker_command(&mut event_writer, worker_command).await;
    }
}

async fn send_ready_event(event_writer: &mut WorkerEventWriter) -> Result<(), ProtocolError> {
    event_writer
        .send_event(&WorkerEvent::Ready {
            model_id: "astronomical/test-worker".to_owned(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: false,
                supports_tool_calls: false,
                has_vision: false,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            }
            .into(),
        })
        .await
}

async fn answer_worker_command(
    event_writer: &mut WorkerEventWriter,
    worker_command: WorkerCommand,
) {
    match worker_command {
        WorkerCommand::InitializeWorker(_) => {}
        WorkerCommand::GenerateImage(_) => {
            panic!("the chat-only fixture must not receive image generation")
        }
        WorkerCommand::GenerateEmbeddings(_)
        | WorkerCommand::GenerateDecisions(_)
        | WorkerCommand::TrainDecisionModel(_)
        | WorkerCommand::GenerateSpeech(_) => {
            panic!(
                "the chat-only fixture must not receive an embeddings, decision or speech request"
            )
        }
        // A count is answered like a generation: one Completed event.
        WorkerCommand::Generate(generation_command)
        | WorkerCommand::CountChatTokens(generation_command) => {
            send_empty_completion(
                event_writer,
                generation_command.request_id,
                ChatGenerationCompletionReason::EndOfSequence,
            )
            .await;
        }
        WorkerCommand::Cancel { request_id } => {
            send_empty_completion(
                event_writer,
                request_id,
                ChatGenerationCompletionReason::Cancelled,
            )
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
            configuration_generation: _,
        } => {
            let _send_outcome = event_writer
                .send_event(&WorkerEvent::MlxMemoryLimitChanged {
                    effective_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes: 1,
                    expert_memory_mode: ipc_protocol::ExpertMemoryMode::Resident,
                    mlx_memory_snapshot: None,
                    expert_residency: None,
                })
                .await;
        }
        WorkerCommand::ClearAstroSsdPromptCache { model_id } => {
            let _send_outcome = event_writer
                .send_event(&WorkerEvent::AstroSsdPromptCacheCleared {
                    model_id,
                    blocks_removed: 0,
                    bytes_freed: 0,
                })
                .await;
        }
    }
}

/// The one Completed event both a generation and a cancel are answered with:
/// a one-token prompt, nothing generated.
async fn send_empty_completion(
    event_writer: &mut WorkerEventWriter,
    request_id: RequestId,
    reason: ChatGenerationCompletionReason,
) {
    let _send_outcome = event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            astro_ssd_prompt_cache_diagnostics: None,
            reason,
        })
        .await;
}
