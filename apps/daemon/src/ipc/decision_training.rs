//! Training routing for the daemon IPC service: resolves the decision
//! model, queues the training like any decisions request, and streams its
//! progress frames, then exactly one terminal frame.

use decision_contract::DecisionFailure;
use decision_contract::decision_training_job::DecisionTrainingJob;
use ipc_protocol::{
    DaemonResponse, DaemonTransportError, DecisionTrainingCommand, RequestId,
    StreamingResponseWriter,
};

use crate::{
    DaemonIpcGenerationContext, DecisionsExecutionError, DecisionsOutput, GenerationStartError,
    ipc::decisions::resolved_decisions_target, ipc::service::generation_start_rejection_reason,
};

/// Runs one IPC training request.
pub(crate) async fn run_decision_training(
    generation_context: &DaemonIpcGenerationContext,
    requested_model: Option<String>,
    job: DecisionTrainingJob,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let (resolved_model_id, request_identifier) =
        match resolved_decisions_target(generation_context, requested_model.as_deref()) {
            Ok(training_target) => training_target,
            Err(rejection_reason) => {
                streaming_response_writer
                    .send_response(&rejected(rejection_reason))
                    .await?;
                return streaming_response_writer.close().await;
            }
        };
    let training_command = DecisionTrainingCommand {
        request_id: RequestId::new(request_identifier),
        model: resolved_model_id.clone(),
        job,
    };
    let mut training_receiver = match generation_context
        .executor
        .start_training(training_command)
        .await
    {
        Ok(training_receiver) => training_receiver,
        Err(start_error) => {
            streaming_response_writer
                .send_response(&rejected(generation_start_rejection_reason(start_error)))
                .await?;
            return streaming_response_writer.close().await;
        }
    };
    let terminal_response = loop {
        match training_receiver.recv().await {
            Some(Ok(DecisionsOutput::TrainingProgress {
                stage,
                completed,
                total,
            })) => {
                streaming_response_writer
                    .send_response(&DaemonResponse::DecisionTrainingProgress {
                        stage,
                        completed,
                        total,
                    })
                    .await?;
            }
            worker_result => break trained_response(worker_result, resolved_model_id),
        }
    };
    streaming_response_writer
        .send_response(&terminal_response)
        .await?;
    streaming_response_writer.close().await
}

/// The terminal frame for what ended a training's stream.
fn trained_response(
    worker_result: Option<Result<DecisionsOutput, DecisionsExecutionError>>,
    resolved_model_id: String,
) -> DaemonResponse {
    match worker_result {
        Some(Ok(DecisionsOutput::ModelTrained { outcome, .. })) => {
            DaemonResponse::DecisionModelTrained {
                model: resolved_model_id,
                outcome,
            }
        }
        Some(Ok(DecisionsOutput::Answered { .. })) => {
            failed("the worker answered a training with decisions")
        }
        // The stream loop forwards every progress frame itself; one that
        // reaches here would mean the loop changed without this match.
        Some(Ok(DecisionsOutput::TrainingProgress { .. })) => {
            failed("the worker sent progress where the training's result belongs")
        }
        Some(Err(DecisionsExecutionError::WorkerFailure(reason))) => {
            DaemonResponse::DecisionsFailed { reason }
        }
        Some(Err(DecisionsExecutionError::WorkerUnavailable)) => rejected(
            generation_start_rejection_reason(GenerationStartError::WorkerUnavailable),
        ),
        None => failed("the worker stream ended before the training finished"),
    }
}

fn failed(reason: &str) -> DaemonResponse {
    DaemonResponse::DecisionsFailed {
        reason: DecisionFailure::fatal_execution(reason),
    }
}

fn rejected(reason: String) -> DaemonResponse {
    DaemonResponse::GenerationRejected { reason }
}
