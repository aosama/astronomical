//! Acceptance coverage for the decisions HTTP journey, `POST /v1/systemone`.

use std::path::PathBuf;

use astronomical_config::{
    DecisionModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily, ModelLicense,
};
use astronomical_daemon::{DecisionsExecutionError, DecisionsOutput};
use astronomical_decision_contract::{
    ChoiceAnswer, DecisionAnswer, DecisionFailure, OptionProbability, OrderedJsonValue,
    QuestionInputCoverage, QuestionOutcome,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const DECISION_MODEL_ID: &str = "laya-MLX-8bit";
const MULTILINGUAL_DECISION_MODEL_ID: &str = "laya-MLX-8bit-multilingual";
const HAMLET_LINE: &str = "To be, or not to be, that is the question.";

#[tokio::test]
async fn should_answer_a_choice_question_through_the_public_http_journey() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.decisions_outcome = Ok(DecisionsOutput {
        question_outcomes: vec![play_outcome()],
        input_token_count: 31,
        elapsed_millis: 4,
    });
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should return a decisions response");

    assert_eq!(response.status(), StatusCode::OK);
    let response_text = response_text(response).await;
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text).expect("the response should contain JSON");
    assert_eq!(response_document["model"], DECISION_MODEL_ID);
    assert_eq!(response_document["answers"]["play"]["type"], "choice");
    assert_eq!(response_document["answers"]["play"]["choice"], "hamlet");
    assert_eq!(
        response_document["answers"]["play"]["probabilities"]["hamlet"],
        0.8123
    );
    assert_eq!(response_document["usage"]["input_tokens"], 31);
    assert_eq!(response_document["usage"]["output_tokens"], 0);
    assert!(
        response_text.contains(r#""probabilities":{"macbeth":0.1877,"hamlet":0.8123}"#),
        "probabilities must come back in the order the options were listed: {response_text}"
    );

    let commands = received_commands
        .lock()
        .expect("the decisions command log should remain available");
    assert_eq!(commands.len(), 1);
    assert_eq!(
        commands[0].model, DECISION_MODEL_ID,
        "with no model named, the only installed decision model answers"
    );
    assert_eq!(
        commands[0].state,
        OrderedJsonValue::Text(HAMLET_LINE.to_owned())
    );
}

#[tokio::test]
async fn should_route_a_named_model_when_several_decision_models_are_installed() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![
            decision_model(DECISION_MODEL_ID),
            decision_model(MULTILINGUAL_DECISION_MODEL_ID),
        ],
    );
    let mut request_document = valid_request_document();
    request_document["model"] = serde_json::json!(MULTILINGUAL_DECISION_MODEL_ID);

    let response = application
        .oneshot(system_one_request(request_document))
        .await
        .expect("the application should return a decisions response");

    assert_eq!(response.status(), StatusCode::OK);
    let commands = received_commands
        .lock()
        .expect("the decisions command log should remain available");
    assert_eq!(commands[0].model, MULTILINGUAL_DECISION_MODEL_ID);
}

#[tokio::test]
async fn should_answer_with_the_default_decision_model_when_several_are_installed() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![
            decision_model(MULTILINGUAL_DECISION_MODEL_ID),
            decision_model(DECISION_MODEL_ID),
        ],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should answer with the default decision model");

    assert_eq!(response.status(), StatusCode::OK);
    let commands = received_commands
        .lock()
        .expect("the decisions command log should remain available");
    assert_eq!(
        commands[0].model, DECISION_MODEL_ID,
        "one download installs both models, so the default must answer unnamed requests"
    );
}

#[tokio::test]
async fn should_ask_for_a_model_name_when_several_other_decision_models_are_installed() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model("decider-b"), decision_model("decider-a")],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should reject an ambiguous request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "decision_model_ambiguous"
    );
    assert_eq!(
        response_document["error"]["message"],
        "more than one decision model is installed (decider-a, decider-b); name one with `model`"
    );
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_reject_a_chat_model_before_decisions_dispatch() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![chat_model(), decision_model(DECISION_MODEL_ID)],
    );
    let mut request_document = valid_request_document();
    request_document["model"] = serde_json::json!("chat-only-model");

    let response = application
        .oneshot(system_one_request(request_document))
        .await
        .expect("the application should reject a chat model");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "model_capability_mismatch"
    );
    assert_eq!(response_document["error"]["param"], "model");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty()
    );
}

#[tokio::test]
async fn should_reject_a_request_when_no_decision_model_is_installed() {
    let application = astronomical_daemon::build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        vec![chat_model()],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should reject the request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(response_document["error"]["code"], "no_decision_model");
}

#[tokio::test]
async fn should_reject_an_unknown_question_type_before_decisions_dispatch() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_decisions_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(serde_json::json!({
            "state": HAMLET_LINE,
            "questions": {
                "play": {"type": "ranking", "instructions": "Which play?"}
            }
        })))
        .await
        .expect("the application should reject malformed questions");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(response_document["error"]["code"], "invalid_questions");
    assert_eq!(response_document["error"]["param"], "questions");
    assert!(
        received_commands
            .lock()
            .expect("command log should remain available")
            .is_empty(),
        "malformed questions must never reach the worker or trigger a model swap"
    );
}

#[tokio::test]
async fn should_reject_a_request_without_a_state() {
    let application = astronomical_daemon::build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(serde_json::json!({
            "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}
        })))
        .await
        .expect("the application should reject a request without a state");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(response_document["error"]["param"], "state");
}

#[tokio::test]
async fn should_return_the_models_reason_for_a_request_it_cannot_answer() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.decisions_outcome = Err(DecisionsExecutionError::WorkerFailure(
        DecisionFailure::invalid_request("question `play` leaves no room for the state"),
    ));
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should relay the model's refusal");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text(response).await).expect("the response should be JSON");
    assert_eq!(
        response_document["error"]["code"],
        "invalid_decision_request"
    );
    assert_eq!(
        response_document["error"]["message"],
        "question `play` leaves no room for the state"
    );
}

#[tokio::test]
async fn should_not_expose_worker_decisions_failure_reasons_or_local_paths() {
    let fictional_private_path = "/Users/fictional-person/private-model/model.safetensors";
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.decisions_outcome = Err(DecisionsExecutionError::WorkerFailure(
        DecisionFailure::fatal_execution(&format!(
            "native execution failed while mapping {fictional_private_path}"
        )),
    ));
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should sanitize the worker failure");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let response_text = response_text(response).await;
    let response_document: serde_json::Value =
        serde_json::from_str(&response_text).expect("the response should be JSON");
    assert_eq!(response_document["error"]["code"], "decisions_failed");
    assert!(
        !response_text.contains(fictional_private_path),
        "worker failure must not leak a local path: {response_text}"
    );
}

#[tokio::test]
async fn should_report_a_busy_decision_engine_as_retryable() {
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor.decisions_outcome = Err(DecisionsExecutionError::WorkerFailure(
        DecisionFailure::EngineBusy,
    ));
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        vec![decision_model(DECISION_MODEL_ID)],
    );

    let response = application
        .oneshot(system_one_request(valid_request_document()))
        .await
        .expect("the application should answer a busy engine");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

fn valid_request_document() -> serde_json::Value {
    serde_json::json!({
        "state": HAMLET_LINE,
        "questions": {
            "play": {
                "type": "choice",
                "instructions": "Which play is this line from?",
                "criteria": ["macbeth", "hamlet"]
            }
        }
    })
}

/// The scripted worker's answer to `valid_request_document`.
fn play_outcome() -> QuestionOutcome {
    QuestionOutcome {
        question_name: "play".to_owned(),
        answer: DecisionAnswer::Choice(ChoiceAnswer {
            chosen_key: "hamlet".to_owned(),
            option_probabilities: vec![
                OptionProbability {
                    key: "macbeth".to_owned(),
                    probability: 0.18766,
                },
                OptionProbability {
                    key: "hamlet".to_owned(),
                    probability: 0.81234,
                },
            ],
            distribution_confidence: 0.3,
            answer_confidence: 0.81234,
        }),
        option_logits: vec![-0.7, 0.7],
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 12,
            state_tokens_read: 12,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
    }
}

fn system_one_request(request_document: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the decisions request should be valid")
}

async fn response_text(response: axum::response::Response) -> String {
    let response_body = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("the bounded response body should be readable");
    String::from_utf8(response_body.to_vec()).expect("the response should be UTF-8")
}

fn decision_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: Some("cavi-ai/laya-MLX-8bit".to_owned()),
        model_family: ModelFamily::Laya,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/laya"),
        capabilities: ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: 512,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: 1,
    }
}

fn chat_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "chat-only-model".to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat-only"),
        capabilities: ModelCapabilities::Chat(astronomical_config::ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1,
    }
}
