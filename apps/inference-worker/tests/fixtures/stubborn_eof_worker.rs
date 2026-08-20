use std::time::Duration;

use astronomical_inference_worker::worker_process_runtime;
use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpRuntimeState, ProtocolWriter, SpeculativePrefillRuntimeState,
    WorkerEvent,
};
use tokio::io::AsyncReadExt;

fn main() {
    let should_finish_with_blocked_stdin =
        std::env::args().any(|argument| argument == "--finish-with-blocked-stdin");
    worker_process_runtime::run_worker_future_with_bounded_runtime_shutdown(|| async move {
        if should_finish_with_blocked_stdin {
            let _blocked_stdin_read_task = tokio::spawn(async {
                let mut one_byte = [0_u8; 1];
                let _read_outcome = tokio::io::stdin().read(&mut one_byte).await;
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
            return;
        }

        if ProtocolWriter::new(tokio::io::stdout())
            .send_event(&WorkerEvent::Ready {
                mtp_runtime_state: MtpRuntimeState::Disabled,
                mtp_unavailable_reason: None,
                mtp_depth_status: Default::default(),
                speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
                speculative_prefill_unavailable_reason: None,
                speculative_prefill_draft_model_id: None,
                speculative_prefill_draft_model_revision: None,
                model_id: "astronomical/stubborn-worker".to_owned(),
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
            .is_ok()
        {
            std::future::pending::<()>().await;
        }
    })
    .expect("the stubborn-worker fixture runtime should initialize");
}
