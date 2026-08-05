#![forbid(unsafe_code)]

use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpRuntimeState, ProtocolWriter, WorkerEvent,
};

#[tokio::main]
async fn main() {
    let _send_outcome = ProtocolWriter::new(tokio::io::stdout())
        .send_event(&WorkerEvent::Ready {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            model_id: "astronomical/wrong-model".to_owned(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: false,
                supports_tool_calls: false,
                has_vision: false,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
        })
        .await;
}
