#![forbid(unsafe_code)]

use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpRuntimeState, ProtocolReader, ProtocolWriter,
    SpeculativePrefillRuntimeState, WorkerEvent,
};

#[tokio::main]
async fn main() {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let _initialization_command = command_reader.next_command().await;
    let _send_outcome = ProtocolWriter::new(tokio::io::stdout())
        .send_event(&WorkerEvent::Ready {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            mtp_depth_status: Default::default(),
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
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
