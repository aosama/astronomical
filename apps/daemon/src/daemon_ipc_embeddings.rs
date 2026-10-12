//! Embeddings routing for the ephemeral daemon IPC service: resolves the
//! target model, dispatches one embeddings command to the worker executor,
//! and answers with a single terminal frame.

use astronomical_ipc_protocol::{
    DaemonResponse, DaemonTransportError, EmbeddingEncodingFormat, EmbeddingsCommand,
    EmbeddingsFailureReason, RequestId, StreamingResponseWriter,
};

use crate::{
    DaemonIpcGenerationContext, EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
    SupervisorPerformanceOperation, application::allocate_chat_request_id,
    daemon_ipc::generation_start_rejection_reason, worker_health::WorkerHealthSnapshot,
};

/// Runs one IPC embeddings batch: resolves the resident-or-requested model,
/// lets the worker swap models when they differ, and answers with exactly
/// one terminal frame.
pub(crate) async fn run_embeddings_generation(
    generation_context: &DaemonIpcGenerationContext,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    model: Option<String>,
    inputs: Vec<String>,
    dimensions: Option<u32>,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcEmbedGenerate,
            || async {
                let terminal_response = resolve_embeddings_terminal_response(
                    generation_context,
                    model.as_deref(),
                    inputs,
                    dimensions,
                )
                .await;
                send_embeddings_terminal_response(streaming_response_writer, &terminal_response)
                    .await
            },
            |embeddings_outcome| {
                if embeddings_outcome.is_ok() {
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
                }
            },
        )
        .await
}

/// Produces the single terminal frame for one embeddings batch: a rejection
/// before admission, or the worker's completed/failed outcome after.
async fn resolve_embeddings_terminal_response(
    generation_context: &DaemonIpcGenerationContext,
    model: Option<&str>,
    inputs: Vec<String>,
    dimensions: Option<u32>,
) -> DaemonResponse {
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    let Some(resolved_model_id) = resolve_embeddings_model(&health_snapshot, model) else {
        return DaemonResponse::GenerationRejected {
            reason: "no model is resident; load a model or pass --model".to_owned(),
        };
    };
    let Some(request_id_raw) = allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        return DaemonResponse::GenerationRejected {
            reason: "the local request identifier space is exhausted".to_owned(),
        };
    };
    let embeddings_command = EmbeddingsCommand {
        request_id: RequestId::new(request_id_raw),
        model: resolved_model_id.clone(),
        inputs,
        encoding_format: EmbeddingEncodingFormat::Float,
        dimensions,
    };
    if let Err(validation_error) = embeddings_command.validate() {
        return DaemonResponse::GenerationRejected {
            reason: validation_error.to_string(),
        };
    }
    let mut embeddings_result_receiver = match generation_context
        .executor
        .start_embeddings_generation(embeddings_command)
        .await
    {
        Ok(embeddings_result_receiver) => embeddings_result_receiver,
        Err(start_error) => {
            return DaemonResponse::GenerationRejected {
                reason: generation_start_rejection_reason(start_error),
            };
        }
    };
    let embeddings_outcome = match embeddings_result_receiver.recv().await {
        Some(embeddings_outcome) => embeddings_outcome,
        None => {
            return DaemonResponse::EmbeddingsFailed {
                reason: EmbeddingsFailureReason::FatalExecution {
                    reason: "the worker stream ended before the embeddings batch completed"
                        .to_owned(),
                },
            };
        }
    };
    embeddings_outcome_to_daemon_response(embeddings_outcome, resolved_model_id)
}

/// Embeddings target the resident model unless the caller names one. Unlike
/// chat generation, a requested model that differs from the resident model is
/// not a rejection: the worker swaps models for embeddings itself.
fn resolve_embeddings_model(
    health_snapshot: &WorkerHealthSnapshot,
    requested_model_id: Option<&str>,
) -> Option<String> {
    requested_model_id
        .map(str::to_owned)
        .or_else(|| health_snapshot.ready_model_id.clone())
}

fn embeddings_outcome_to_daemon_response(
    embeddings_outcome: Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    resolved_model_id: String,
) -> DaemonResponse {
    match embeddings_outcome {
        Ok(embeddings_output) => DaemonResponse::EmbeddingsCompleted {
            model: resolved_model_id,
            vectors: embeddings_output.embeddings,
            input_token_counts: embeddings_output.input_token_counts,
        },
        Err(EmbeddingsExecutionError::WorkerFailure(reason)) => {
            DaemonResponse::EmbeddingsFailed { reason }
        }
        Err(EmbeddingsExecutionError::WorkerUnavailable) => DaemonResponse::GenerationRejected {
            reason: generation_start_rejection_reason(GenerationStartError::WorkerUnavailable),
        },
    }
}

async fn send_embeddings_terminal_response(
    mut streaming_response_writer: StreamingResponseWriter,
    daemon_response: &DaemonResponse,
) -> Result<(), DaemonTransportError> {
    streaming_response_writer
        .send_response(daemon_response)
        .await?;
    streaming_response_writer.close().await
}
