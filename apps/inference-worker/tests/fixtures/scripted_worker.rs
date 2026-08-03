use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, MtpRuntimeState, ProtocolReader,
    ProtocolWriter, WorkerCommand, WorkerEvent,
};

#[tokio::main]
async fn main() {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    if event_writer
        .send_event(&WorkerEvent::Ready {
            expert_storage_format:
                astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            model_id: "astronomical/scripted-worker".to_owned(),
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
        let (request_id, reason) = match worker_command {
            WorkerCommand::InitializeWorker(_) => continue,
            WorkerCommand::Generate(generation_command) => (
                generation_command.request_id,
                ChatGenerationCompletionReason::EndOfSequence,
            ),
            WorkerCommand::Cancel { request_id } => {
                (request_id, ChatGenerationCompletionReason::Cancelled)
            }
            WorkerCommand::SwapModel { .. } => continue,
            WorkerCommand::SampleMlxMemory => continue,
            WorkerCommand::UpdateMlxMemoryLimit { .. } => continue,
        };
        if event_writer
            .send_event(&WorkerEvent::Completed {
                request_id,
                prompt_token_count: 1,
                generated_token_count: 0,
                reasoning_token_count: 0,
                cached_token_count: 0,
                reason,
            })
            .await
            .is_err()
        {
            return;
        }
    }
}
