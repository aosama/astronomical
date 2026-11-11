use astronomical_ipc_protocol::{
    ChatGenerationFailureReason, ProtocolWriter, RequestId, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::support;
use super::support::WorkerRuntimeError;
use crate::InferenceEngineError;

pub(crate) async fn report_fatal_engine_error<WriteTransport, ActiveGeneration>(
    request_id: RequestId,
    engine_error: InferenceEngineError,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<Option<ActiveGeneration>, WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    let fatal_worker_error = support::engine_generation_error(engine_error);
    let public_failure_reason = public_fatal_failure_reason(&fatal_worker_error.to_string());
    tracing::error!(
        request_id = request_id.value(),
        error = %fatal_worker_error,
        public_failure_reason = ?public_failure_reason,
        "fatal model execution failed; reporting bounded reason before worker exit"
    );
    event_writer
        .send_event(&WorkerEvent::Failed {
            request_id,
            reason: public_failure_reason,
        })
        .await?;
    Err(fatal_worker_error)
}

/// Whether rendered error text is Metal's report of a command buffer that ran
/// out of GPU memory. The worker holds only the text of a fatal error here,
/// and this module is built without the MLX runtime crate, so the two
/// fragments are matched again rather than shared with
/// `MlxRuntimeError::is_recoverable_graphics_processor_out_of_memory`; a test
/// built with the runtime holds the two to the same answer. Both fragments are
/// required so another subsystem's "Insufficient Memory" is never blamed on
/// the graphics processor.
fn describes_graphics_processor_out_of_memory(rendered_error: &str) -> bool {
    rendered_error.contains("[METAL] Command buffer execution failed: Insufficient Memory")
        && rendered_error.contains("kIOGPUCommandBufferCallbackErrorOutOfMemory")
}

/// The bounded reason a fatal failure is reported under. Running out of
/// graphics memory is named on its own because it is the one fatal failure the
/// person can do something about; every other one keeps native details in the
/// log and says only that execution failed.
pub(crate) fn public_fatal_failure_reason(
    unbounded_failure_reason: &str,
) -> ChatGenerationFailureReason {
    if describes_graphics_processor_out_of_memory(unbounded_failure_reason) {
        return ChatGenerationFailureReason::OutOfGraphicsMemory;
    }
    let bounded_reason = if unbounded_failure_reason.contains("[metal::malloc]")
        && unbounded_failure_reason.contains("maximum allowed buffer size")
    {
        "GPU allocation exceeded the platform buffer limit while evaluating the model; reduce the prompt size or configured prefill chunk size"
    } else {
        "model execution failed inside the local worker; inspect the worker log for the request-specific native error"
    };
    ChatGenerationFailureReason::FatalExecution {
        reason: bounded_reason.to_owned(),
    }
}

#[cfg(test)]
#[path = "fatal_tests.rs"]
mod fatal_tests;
