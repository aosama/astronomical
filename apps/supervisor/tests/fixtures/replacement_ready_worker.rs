#![forbid(unsafe_code)]

use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpRuntimeState, ProtocolReader, ProtocolWriter, WorkerEvent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    event_writer
        .send_event(&WorkerEvent::Ready {
            expert_storage_format:
                astronomical_ipc_protocol::ExpertStorageFormat::StandardSafetensors,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            model_id: "astronomical/replacement-worker".to_owned(),
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

    while command_reader.next_command().await?.is_some() {}
    Ok(())
}
