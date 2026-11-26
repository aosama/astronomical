//! Unit contracts for reading a System One request body before any model,
//! lock or queue is touched.

use axum::http::StatusCode;
use decision_contract::OrderedJsonValue;

use super::{decisions_reply_response, parse_system_one_request};
use crate::{DecisionsExecutionError, DecisionsOutput};
use decision_contract::DecisionFailure;
use decision_contract::decision_training_job::DecisionTrainingStage;

fn rejection_status(request_body: &str) -> StatusCode {
    match parse_system_one_request(request_body.as_bytes()) {
        Ok(_) => panic!("the body should have been rejected: {request_body}"),
        Err(rejection_response) => rejection_response.status(),
    }
}

#[test]
fn should_read_the_state_questions_and_model_of_a_complete_body() {
    let system_one_request = parse_system_one_request(
        br#"{"model": "decider", "state": {"b": 1, "a": 2}, "questions": {"q": {"type": "noul"}}}"#,
    )
    .unwrap_or_else(|_| panic!("a complete body should parse"));

    assert_eq!(
        system_one_request.requested_model.as_deref(),
        Some("decider")
    );
    let state_keys: Vec<&str> = system_one_request
        .state
        .as_object()
        .expect("the state should stay an object")
        .iter()
        .map(|state_member| state_member.key.as_str())
        .collect();
    assert_eq!(
        state_keys,
        ["b", "a"],
        "the state keeps the member order the model will read"
    );
    assert!(system_one_request.questions.member("q").is_some());
}

#[test]
fn should_treat_a_missing_or_null_model_as_no_model_named() {
    let without_model = parse_system_one_request(br#"{"state": "s", "questions": {}}"#)
        .unwrap_or_else(|_| panic!("a body without a model should parse"));
    let with_null_model =
        parse_system_one_request(br#"{"model": null, "state": "s", "questions": {}}"#)
            .unwrap_or_else(|_| panic!("a body with a null model should parse"));

    assert_eq!(without_model.requested_model, None);
    assert_eq!(with_null_model.requested_model, None);
}

#[test]
fn should_ignore_fields_it_does_not_know() {
    let system_one_request = parse_system_one_request(
        br#"{"state": "s", "questions": {}, "temperature": 0.2, "stream": false}"#,
    )
    .unwrap_or_else(|_| panic!("unknown fields should be ignored"));

    assert_eq!(
        system_one_request.state,
        OrderedJsonValue::Text("s".to_owned())
    );
}

#[test]
fn should_accept_an_empty_text_state_and_leave_judging_it_to_the_model() {
    let system_one_request = parse_system_one_request(br#"{"state": "", "questions": {}}"#)
        .unwrap_or_else(|_| panic!("an empty state is still a state"));

    assert_eq!(
        system_one_request.state,
        OrderedJsonValue::Text(String::new())
    );
}

#[test]
fn should_reject_a_body_that_is_not_json() {
    assert_eq!(rejection_status("{not json"), StatusCode::BAD_REQUEST);
}

#[test]
fn should_reject_a_body_that_is_not_an_object() {
    assert_eq!(rejection_status(r#"["state"]"#), StatusCode::BAD_REQUEST);
}

#[test]
fn should_reject_a_body_without_a_state() {
    assert_eq!(
        rejection_status(r#"{"questions": {}}"#),
        StatusCode::BAD_REQUEST
    );
}

#[test]
fn should_reject_a_body_without_questions() {
    assert_eq!(
        rejection_status(r#"{"state": "s"}"#),
        StatusCode::BAD_REQUEST
    );
}

#[test]
fn should_reject_a_model_that_is_not_a_non_empty_string() {
    assert_eq!(
        rejection_status(r#"{"model": 7, "state": "s", "questions": {}}"#),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        rejection_status(r#"{"model": "  ", "state": "s", "questions": {}}"#),
        StatusCode::BAD_REQUEST
    );
}

#[test]
fn should_answer_with_a_system_one_body_when_the_worker_answered() {
    let reply_response = decisions_reply_response(
        "decider",
        7,
        Some(Ok(DecisionsOutput::Answered {
            question_outcomes: Vec::new(),
            input_token_count: 12,
            elapsed_millis: 3,
        })),
    );

    assert_eq!(reply_response.status(), StatusCode::OK);
}

#[test]
fn should_report_the_worker_unavailable_when_the_reply_channel_closed_silently() {
    let reply_response = decisions_reply_response("decider", 7, None);

    assert_eq!(reply_response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn should_report_the_worker_unavailable_when_an_answer_request_got_training_progress() {
    let reply_response = decisions_reply_response(
        "decider",
        7,
        Some(Ok(DecisionsOutput::TrainingProgress {
            stage: DecisionTrainingStage::Reading,
            completed: 1,
            total: 4,
        })),
    );

    assert_eq!(reply_response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn should_map_worker_failures_to_their_http_statuses() {
    let busy_response = decisions_reply_response(
        "decider",
        7,
        Some(Err(DecisionsExecutionError::WorkerFailure(
            DecisionFailure::EngineBusy,
        ))),
    );
    let invalid_response = decisions_reply_response(
        "decider",
        7,
        Some(Err(DecisionsExecutionError::WorkerFailure(
            DecisionFailure::InvalidRequest {
                reason: "question q has no options".to_owned(),
            },
        ))),
    );

    assert_eq!(busy_response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(invalid_response.status(), StatusCode::BAD_REQUEST);
}
