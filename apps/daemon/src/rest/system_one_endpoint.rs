//! The decisions endpoint, `POST /v1/systemone`, with complete validation
//! before shared queue admission.
//!
//! System One is the request shape existing decision-model clients send: a
//! `state` to decide about and a `questions` object of typed questions. The
//! response carries one answer per question with a probability for every
//! allowed answer. Fields this endpoint does not know are ignored, as those
//! clients expect.

use astronomical_decision_contract::{DecisionFailure, OrderedJsonValue, system_one_response};
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

use crate::{
    DecisionsExecutionError, GenerationStartError,
    application::{ApplicationState, allocate_chat_request_id},
    rest::single_shot_rejections::{
        generation_start_error_response, invalid_request_response,
        request_identifier_exhausted_response, worker_unavailable_response,
    },
    serving::decisions_service::{
        DecisionsModelResolutionError, build_decisions_command, resolve_decisions_model,
    },
};

/// The validated top-level fields of one System One request body.
struct SystemOneRequest {
    requested_model: Option<String>,
    state: OrderedJsonValue,
    questions: OrderedJsonValue,
}

pub(crate) async fn create_decisions(
    State(application_state): State<ApplicationState>,
    request_body_bytes: axum::body::Bytes,
) -> Response {
    let system_one_request = match parse_system_one_request(request_body_bytes.as_ref()) {
        Ok(system_one_request) => system_one_request,
        Err(rejection_response) => return rejection_response,
    };
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return worker_unavailable_response();
    }
    let resolved_model_id = match resolve_decisions_model(
        &application_state.discovered_models_snapshot(),
        worker_health_snapshot.ready_model_id.as_deref(),
        system_one_request.requested_model.as_deref(),
    ) {
        Ok(resolved_model_id) => resolved_model_id,
        Err(resolution_error) => return model_resolution_rejection_response(&resolution_error),
    };
    let Some(request_identifier) =
        allocate_chat_request_id(&application_state.next_chat_request_id)
    else {
        return request_identifier_exhausted_response();
    };
    let decisions_command = match build_decisions_command(
        request_identifier,
        &resolved_model_id,
        system_one_request.state,
        system_one_request.questions,
    ) {
        Ok(decisions_command) => decisions_command,
        Err(question_error) => {
            return invalid_request_response(
                question_error.to_string(),
                Some("questions"),
                "invalid_questions",
            );
        }
    };
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut decisions_start_future = application_state
        .generation_executor
        .start_decisions_generation_with_admission_signal(decisions_command, admission_sender);
    // The configuration lock is held only until the request owns its place
    // in the queue, so a reload cannot slip between resolution and admission.
    let decisions_start_result = tokio::select! {
        admission_result = &mut admission_receiver => {
            drop(configuration_transition_guard);
            match admission_result {
                Ok(()) => decisions_start_future.await,
                Err(_) => Err(GenerationStartError::WorkerUnavailable),
            }
        }
        decisions_start_result = &mut decisions_start_future => {
            drop(configuration_transition_guard);
            decisions_start_result
        }
    };
    let mut decisions_result_receiver = match decisions_start_result {
        Ok(decisions_result_receiver) => decisions_result_receiver,
        Err(start_error) => return generation_start_error_response(start_error),
    };
    match decisions_result_receiver.recv().await {
        Some(Ok(decisions_output)) => Json(system_one_response(
            &resolved_model_id,
            &decisions_output.question_outcomes,
            decisions_output.input_token_count,
        ))
        .into_response(),
        Some(Err(decisions_error)) => {
            decisions_execution_error_response(request_identifier, decisions_error)
        }
        None => worker_unavailable_response(),
    }
}

/// Reads the request body's three fields, rejecting anything malformed
/// before any state is touched.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_system_one_request(request_body_bytes: &[u8]) -> Result<SystemOneRequest, Response> {
    let request_body: OrderedJsonValue = match serde_json::from_slice(request_body_bytes) {
        Ok(request_body) => request_body,
        Err(json_error) => {
            return Err(invalid_request_response(
                format!("request body is not valid JSON: {json_error}"),
                None,
                "invalid_json",
            ));
        }
    };
    if request_body.as_object().is_none() {
        return Err(invalid_request_response(
            "request body must be a JSON object",
            None,
            "invalid_request",
        ));
    }
    let requested_model = match request_body.member("model") {
        None | Some(OrderedJsonValue::Null) => None,
        Some(OrderedJsonValue::Text(model)) if !model.trim().is_empty() => Some(model.clone()),
        Some(_) => {
            return Err(invalid_request_response(
                "model must be a non-empty string when given",
                Some("model"),
                "invalid_request",
            ));
        }
    };
    let Some(state) = request_body.member("state") else {
        return Err(invalid_request_response(
            "state is required: the text or JSON to decide about",
            Some("state"),
            "invalid_request",
        ));
    };
    let Some(questions) = request_body.member("questions") else {
        return Err(invalid_request_response(
            "questions is required: an object of typed questions",
            Some("questions"),
            "invalid_request",
        ));
    };
    Ok(SystemOneRequest {
        requested_model,
        state: state.clone(),
        questions: questions.clone(),
    })
}

fn model_resolution_rejection_response(
    resolution_error: &DecisionsModelResolutionError,
) -> Response {
    let error_code = match resolution_error {
        DecisionsModelResolutionError::ModelNotFound => "model_not_found",
        DecisionsModelResolutionError::ModelCannotDecide => "model_capability_mismatch",
        DecisionsModelResolutionError::NoDecisionModelInstalled => "no_decision_model",
        DecisionsModelResolutionError::SeveralDecisionModelsInstalled { .. } => {
            "decision_model_ambiguous"
        }
    };
    invalid_request_response(resolution_error.to_string(), Some("model"), error_code)
}

fn decisions_execution_error_response(
    request_identifier: u64,
    decisions_error: DecisionsExecutionError,
) -> Response {
    match decisions_error {
        // The reason names the question and the fix, and is already bounded
        // and free of local paths, so it is returned as written.
        DecisionsExecutionError::WorkerFailure(DecisionFailure::InvalidRequest { reason }) => {
            invalid_request_response(reason, Some("questions"), "invalid_decision_request")
        }
        DecisionsExecutionError::WorkerFailure(DecisionFailure::EngineBusy) => (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::capacity_unavailable(
                    "the decision engine is busy",
                ),
            ),
        )
            .into_response(),
        DecisionsExecutionError::WorkerFailure(DecisionFailure::FatalExecution { reason }) => {
            tracing::error!(
                request_identifier,
                reason = %reason,
                "decisions worker execution failed"
            );
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                        "the decision request failed in the local worker",
                        Some("decisions_failed"),
                    ),
                ),
            )
                .into_response()
        }
        DecisionsExecutionError::WorkerUnavailable => worker_unavailable_response(),
    }
}

#[cfg(test)]
#[path = "system_one_endpoint_tests.rs"]
mod system_one_endpoint_tests;
