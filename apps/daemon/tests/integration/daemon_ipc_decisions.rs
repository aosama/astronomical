//! Daemon IPC decisions journeys: routing DecideGenerate requests to the
//! worker executor and refusing them before dispatch when they cannot run.

use std::sync::{Arc, RwLock};

use astronomical_daemon::{DecisionsExecutionError, DecisionsOutput};
use astronomical_decision_contract::{DecisionFailure, OrderedJsonValue};
use astronomical_ipc_protocol::{DaemonIpcClient, DaemonRequest, DaemonResponse};
use tokio::time::timeout;

use crate::support::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ipc_chat_discovered_model, ipc_decision_discovered_model, ipc_runtime_config,
    ready_stub_executor, ready_stub_executor_with_decisions,
    start_stub_daemon_ipc_service_with_runtime_config,
};

const DECISION_MODEL_ID: &str = "test-local-decider";
const CHAT_MODEL_ID: &str = "test-local-chatter";

fn questions_from(questions_json: &str) -> OrderedJsonValue {
    serde_json::from_str(questions_json).expect("the test questions should be JSON")
}

fn urgent_question() -> OrderedJsonValue {
    questions_from(r#"{"urgent": {"type": "noul", "instructions": "Is it urgent?"}}"#)
}

/// Sends one decide request to a daemon that knows a chat model and a
/// decision model, and returns the single terminal frame.
async fn decide_once(
    test_name: &str,
    stub_executor: Arc<StubGenerationExecutor>,
    decide_request: DaemonRequest,
) -> DaemonResponse {
    let state_directory = fresh_instance_state_directory(test_name);
    let runtime_config = ipc_runtime_config(
        vec![
            ipc_chat_discovered_model(CHAT_MODEL_ID),
            ipc_decision_discovered_model(DECISION_MODEL_ID),
        ],
        &[CHAT_MODEL_ID],
    );
    let daemon_ipc_service = start_stub_daemon_ipc_service_with_runtime_config(
        &state_directory,
        stub_executor,
        Some(Arc::new(RwLock::new(runtime_config))),
    )
    .await;
    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&decide_request),
    )
    .await
    .expect("the decide request send should finish inside the test timeout")
    .expect("the decide request should transmit");
    let decide_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the decide response should arrive inside the test timeout")
        .expect("the decide response read should not fail at the transport layer")
        .expect("the daemon should answer the decide request");
    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
    decide_response
}

#[tokio::test]
async fn should_route_a_decide_request_to_the_only_installed_decision_model() {
    // A chat model is resident: the daemon must still pick the decision
    // model, and leave making it resident to the worker loop.
    let stub_executor = ready_stub_executor_with_decisions(
        CHAT_MODEL_ID,
        Ok(DecisionsOutput {
            question_outcomes: Vec::new(),
            input_token_count: 9,
            elapsed_millis: 5,
        }),
    );

    let decide_response = decide_once(
        "decide-routed",
        stub_executor.clone(),
        DaemonRequest::DecideGenerate {
            model: None,
            state: OrderedJsonValue::Text("The server room is on fire.".to_owned()),
            questions: urgent_question(),
        },
    )
    .await;

    assert_eq!(
        decide_response,
        DaemonResponse::DecisionsCompleted {
            model: DECISION_MODEL_ID.to_owned(),
            question_outcomes: Vec::new(),
            input_token_count: 9,
            elapsed_millis: 5,
        }
    );
    let received_decisions_commands = stub_executor.logged_decisions_commands();
    assert_eq!(received_decisions_commands.len(), 1);
    assert_eq!(received_decisions_commands[0].model, DECISION_MODEL_ID);
    assert_eq!(received_decisions_commands[0].questions, urgent_question());
}

#[tokio::test]
async fn should_reject_a_decide_request_that_names_a_chat_model() {
    let stub_executor = ready_stub_executor(CHAT_MODEL_ID);

    let decide_response = decide_once(
        "decide-chat",
        stub_executor.clone(),
        DaemonRequest::DecideGenerate {
            model: Some(CHAT_MODEL_ID.to_owned()),
            state: OrderedJsonValue::Text("The server room is on fire.".to_owned()),
            questions: urgent_question(),
        },
    )
    .await;

    assert_eq!(
        decide_response,
        DaemonResponse::GenerationRejected {
            reason: "the requested model does not answer decision requests".to_owned(),
        }
    );
    assert!(stub_executor.logged_decisions_commands().is_empty());
}

#[tokio::test]
async fn should_reject_empty_questions_before_dispatch() {
    let stub_executor = ready_stub_executor(CHAT_MODEL_ID);

    let decide_response = decide_once(
        "decide-empty",
        stub_executor.clone(),
        DaemonRequest::DecideGenerate {
            model: None,
            state: OrderedJsonValue::Text("The server room is on fire.".to_owned()),
            questions: questions_from("{}"),
        },
    )
    .await;

    assert!(
        matches!(decide_response, DaemonResponse::GenerationRejected { .. }),
        "a request with no questions must be refused: {decide_response:?}"
    );
    assert!(stub_executor.logged_decisions_commands().is_empty());
}

#[tokio::test]
async fn should_relay_the_workers_failure_as_a_failed_frame() {
    let stub_executor = ready_stub_executor_with_decisions(
        CHAT_MODEL_ID,
        Err(DecisionsExecutionError::WorkerFailure(
            DecisionFailure::invalid_request("question `urgent` leaves no room for the state"),
        )),
    );

    let decide_response = decide_once(
        "decide-failure",
        stub_executor,
        DaemonRequest::DecideGenerate {
            model: None,
            state: OrderedJsonValue::Text("The server room is on fire.".to_owned()),
            questions: urgent_question(),
        },
    )
    .await;

    assert_eq!(
        decide_response,
        DaemonResponse::DecisionsFailed {
            reason: DecisionFailure::invalid_request(
                "question `urgent` leaves no room for the state"
            ),
        }
    );
}
