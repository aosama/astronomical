use astronomical_ipc_protocol::{
    ChatGenerationFailureReason, ProtocolWriter, RequestId, WorkerEvent,
};
use tokio::io::AsyncWrite;

use crate::InferenceEngineError;
use crate::engine_backed_worker_support::{WorkerRuntimeError, engine_generation_error};

pub(crate) async fn report_fatal_engine_error<WriteTransport, ActiveGeneration>(
    request_id: RequestId,
    engine_error: InferenceEngineError,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<Option<ActiveGeneration>, WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    let fatal_worker_error = engine_generation_error(engine_error);
    let public_failure_reason = public_fatal_execution_reason(&fatal_worker_error.to_string());
    tracing::error!(
        request_id = request_id.value(),
        error = %fatal_worker_error,
        public_failure_reason,
        "fatal model execution failed; reporting bounded reason before worker exit"
    );
    event_writer
        .send_event(&WorkerEvent::Failed {
            request_id,
            reason: ChatGenerationFailureReason::FatalExecution {
                reason: public_failure_reason.to_owned(),
            },
        })
        .await?;
    Err(fatal_worker_error)
}

fn public_fatal_execution_reason(unbounded_failure_reason: &str) -> &'static str {
    if unbounded_failure_reason.contains("[metal::malloc]")
        && unbounded_failure_reason.contains("maximum allowed buffer size")
    {
        return "GPU allocation exceeded the platform buffer limit while evaluating the model; reduce the prompt size or configured prefill chunk size";
    }
    "model execution failed inside the local worker; inspect the worker log for the request-specific native error"
}
