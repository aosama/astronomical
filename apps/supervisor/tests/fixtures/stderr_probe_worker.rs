#![forbid(unsafe_code)]

use std::{error::Error, os::unix::fs::MetadataExt, process::ExitCode};

use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpRuntimeState, ProtocolWriter, SpeculativePrefillRuntimeState,
    WorkerEvent,
};

#[tokio::main]
async fn main() -> ExitCode {
    match run_stderr_probe_worker().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(worker_error) => {
            eprintln!("stderr-probe worker failed: {worker_error}");
            ExitCode::FAILURE
        }
    }
}

async fn run_stderr_probe_worker() -> Result<(), Box<dyn Error + Send + Sync>> {
    if stderr_points_to_dev_null()? {
        return Ok(());
    }
    eprintln!("stderr-probe worker observed visible stderr");
    ProtocolWriter::new(tokio::io::stdout())
        .send_event(&WorkerEvent::Ready {
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
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
        .await?;
    Ok(())
}

fn stderr_points_to_dev_null() -> Result<bool, Box<dyn Error + Send + Sync>> {
    let stderr_metadata = std::fs::metadata("/dev/fd/2")?;
    let dev_null_metadata = std::fs::metadata("/dev/null")?;
    Ok(stderr_metadata.dev() == dev_null_metadata.dev()
        && stderr_metadata.ino() == dev_null_metadata.ino())
}
