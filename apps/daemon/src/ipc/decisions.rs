//! Decisions routing for the ephemeral daemon IPC service: resolves the
//! target model, dispatches one decisions command to the worker executor,
//! and answers with a single terminal frame.

use astronomical_decision_contract::{DecisionFailure, OrderedJsonValue};
use astronomical_ipc_protocol::{DaemonResponse, DaemonTransportError, StreamingResponseWriter};

use crate::{
    DaemonIpcGenerationContext, DecisionsExecutionError, GenerationStartError,
    application::allocate_chat_request_id,
    ipc::service::generation_start_rejection_reason,
    serving::decisions_service::{build_decisions_command, resolve_decisions_model},
};

/// Runs one IPC decisions request and answers with exactly one terminal frame.
pub(crate) async fn run_decisions_generation(
    generation_context: &DaemonIpcGenerationContext,
    model: Option<String>,
    state: OrderedJsonValue,
    questions: OrderedJsonValue,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let terminal_response =
        resolve_decisions_terminal_response(generation_context, model.as_deref(), state, questions)
            .await;
    streaming_response_writer
        .send_response(&terminal_response)
        .await?;
    streaming_response_writer.close().await
}

/// Produces the single terminal frame for one decisions request: a rejection
/// before admission, or the worker's completed or failed outcome after.
async fn resolve_decisions_terminal_response(
    generation_context: &DaemonIpcGenerationContext,
    requested_model: Option<&str>,
    state: OrderedJsonValue,
    questions: OrderedJsonValue,
) -> DaemonResponse {
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    let discovered_models = match generation_context.reloadable_config.as_ref() {
        Some(reloadable_config) => match reloadable_config.read() {
            Ok(resolved_runtime_config) => resolved_runtime_config.discovered_models.clone(),
            Err(_poisoned_lock) => Vec::new(),
        },
        None => Vec::new(),
    };
    let resolved_model_id = match resolve_decisions_model(
        &discovered_models,
        health_snapshot.ready_model_id.as_deref(),
        requested_model,
    ) {
        Ok(resolved_model_id) => resolved_model_id,
        Err(resolution_error) => return rejected(resolution_error.to_string()),
    };
    let Some(request_identifier) =
        allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        return rejected("the local request identifier space is exhausted".to_owned());
    };
    let decisions_command =
        match build_decisions_command(request_identifier, &resolved_model_id, state, questions) {
            Ok(decisions_command) => decisions_command,
            Err(question_error) => return rejected(question_error.to_string()),
        };
    let mut decisions_result_receiver = match generation_context
        .executor
        .start_decisions_generation(decisions_command)
        .await
    {
        Ok(decisions_result_receiver) => decisions_result_receiver,
        Err(start_error) => return rejected(generation_start_rejection_reason(start_error)),
    };
    match decisions_result_receiver.recv().await {
        Some(Ok(decisions_output)) => DaemonResponse::DecisionsCompleted {
            model: resolved_model_id,
            question_outcomes: decisions_output.question_outcomes,
            input_token_count: decisions_output.input_token_count,
            elapsed_millis: decisions_output.elapsed_millis,
        },
        Some(Err(DecisionsExecutionError::WorkerFailure(reason))) => {
            DaemonResponse::DecisionsFailed { reason }
        }
        Some(Err(DecisionsExecutionError::WorkerUnavailable)) => rejected(
            generation_start_rejection_reason(GenerationStartError::WorkerUnavailable),
        ),
        None => DaemonResponse::DecisionsFailed {
            reason: DecisionFailure::fatal_execution(
                "the worker stream ended before the decision request completed",
            ),
        },
    }
}

fn rejected(reason: String) -> DaemonResponse {
    DaemonResponse::GenerationRejected { reason }
}
