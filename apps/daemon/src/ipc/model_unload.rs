//! `ModelUnload` over IPC: the CLI's `astronomical unload`. Same rule as
//! `POST /v1/control/unload-model`, which this answers through the same
//! worker command, so both surfaces free memory the same way.

use ipc_protocol::{
    DaemonModelUnloadStatus, DaemonResponse, DaemonTransportError, StreamingResponseWriter,
};

use crate::ModelUnloadOutcome;

use super::models::send_request_rejection;
use super::service::{DaemonIpcGenerationContext, send_terminal_streaming_response};

/// Unloads the loaded model when no request is running or waiting.
pub(super) async fn handle_model_unload(
    generation_context: &DaemonIpcGenerationContext,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let Some(worker_handle) = generation_context.worker_control.as_ref() else {
        return send_terminal_streaming_response(
            streaming_response_writer,
            &model_unloaded_response(ModelUnloadOutcome::Unavailable),
        )
        .await;
    };
    match worker_handle.unload_model().await {
        Ok(outcome) => {
            send_terminal_streaming_response(
                streaming_response_writer,
                &model_unloaded_response(outcome),
            )
            .await
        }
        Err(worker_control_error) => {
            tracing::error!(error = %worker_control_error, "model unload failed");
            send_request_rejection(
                streaming_response_writer,
                "the model could not be unloaded".to_owned(),
            )
            .await
        }
    }
}

/// The IPC frame for what the unload did.
pub(super) fn model_unloaded_response(outcome: ModelUnloadOutcome) -> DaemonResponse {
    let (status, model_id) = match outcome {
        ModelUnloadOutcome::Unloaded { model_id } => {
            (DaemonModelUnloadStatus::Unloaded, Some(model_id))
        }
        ModelUnloadOutcome::NothingLoaded => (DaemonModelUnloadStatus::NothingLoaded, None),
        ModelUnloadOutcome::Busy => (DaemonModelUnloadStatus::Busy, None),
        ModelUnloadOutcome::Unavailable => (DaemonModelUnloadStatus::Unavailable, None),
        ModelUnloadOutcome::UnloadedWithoutReadyWorker { model_id } => (
            DaemonModelUnloadStatus::UnloadedEngineRestarting,
            Some(model_id),
        ),
    };
    DaemonResponse::ModelUnloaded { status, model_id }
}

#[cfg(test)]
#[path = "model_unload_tests.rs"]
mod tests;
