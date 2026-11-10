//! The `astronomical decide` one-shot journey: a state and typed questions
//! in, one JSON answers document on stdout, process exits. The daemon side
//! is a stub speaking the real framed protocol over a real unix socket.

use std::{
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
};

use astronomical_cli::{DecideArguments, DecideDependencies, DecideError, DecideQuestionsSource};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, NoulAnswer, OrderedJsonValue, QuestionInputCoverage,
    QuestionOutcome,
};
use astronomical_ipc_protocol::DaemonRequest;
use tokio::time::timeout;

use crate::integration::support::environment::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
};
use crate::integration::support::stub_daemon::{
    StubCatalogEntry, StubDaemonConfig, StubDecisionsOutcome, StubInstalledModel, spawn_stub_daemon,
};

/// The model id `decide` uses when the invocation names none.
const DEFAULT_DECISION_MODEL_ID: &str = astronomical_config::BUILTIN_DEFAULT_DECISION_MODEL_ID;

fn asked_decide_arguments(state_text: Option<&str>, model_id: Option<&str>) -> DecideArguments {
    DecideArguments {
        state_text: state_text.map(str::to_owned),
        state_file_path: None,
        state_is_json: false,
        questions_source: DecideQuestionsSource::Ask("Is a refund requested?".to_owned()),
        model_id: model_id.map(str::to_owned),
    }
}

fn decide_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdin: &'a mut dyn Read,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> DecideDependencies<'a> {
    DecideDependencies {
        candidate_socket_paths,
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdin,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
        daemon_work_poll_interval: DOWNLOAD_POLL_INTERVAL,
        model_load_stall_bound: TEST_TIMEOUT,
    }
}

/// One yes-or-no outcome whose state was read whole.
fn refund_outcome(probability_true: f64) -> QuestionOutcome {
    QuestionOutcome {
        question_name: "answer".to_owned(),
        answer: DecisionAnswer::Noul(NoulAnswer {
            probability_true,
            answer_confidence: probability_true.max(1.0 - probability_true),
        }),
        option_logits: vec![-1.0, 1.0],
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 6,
            state_tokens_read: 6,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
    }
}

/// Stub with the default decision model installed beside a resident chat
/// model: the no-flag resolution path, with the daemon doing the swap.
fn installed_decider_stub_config(decisions_outcome: StubDecisionsOutcome) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![
            StubInstalledModel::chat("test/local-chatter", true),
            StubInstalledModel::decisions(DEFAULT_DECISION_MODEL_ID, false),
        ],
        default_model_id: Some("test/local-chatter".to_owned()),
        decisions_outcome,
        ..Default::default()
    }
}

#[tokio::test]
async fn should_answer_an_asked_question_as_one_json_document() {
    let test_directory = fresh_test_directory("decide", "asked-question");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_config =
        installed_decider_stub_config(StubDecisionsOutcome::Completed(vec![refund_outcome(
            0.91234,
        )]));
    let decide_request_capture = Arc::clone(&stub_config.decide_request_capture);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");
    decide_outcome.expect("the decide journey should complete against the stub daemon");

    let answers_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the decide payload should be exactly one JSON document");
    assert_eq!(answers_document["model"], DEFAULT_DECISION_MODEL_ID);
    assert_eq!(answers_document["answers"]["answer"]["type"], "noul");
    assert_eq!(
        answers_document["answers"]["answer"]["noul"], 0.9123,
        "probabilities are reported to four decimal places"
    );
    assert_eq!(
        answers_document["answers"]["answer"]["input"]["state_was_cut"],
        false
    );
    let captured_request = decide_request_capture
        .lock()
        .expect("the decide capture should stay live")
        .clone();
    let Some(DaemonRequest::DecideGenerate {
        model,
        state,
        questions,
    }) = captured_request
    else {
        panic!("the CLI must send one decide request: {captured_request:?}");
    };
    assert_eq!(model.as_deref(), Some(DEFAULT_DECISION_MODEL_ID));
    assert_eq!(
        state,
        OrderedJsonValue::Text("We were billed twice.".to_owned())
    );
    assert_eq!(
        questions
            .member("answer")
            .and_then(|question_definition| question_definition.member("instructions"))
            .and_then(OrderedJsonValue::as_text),
        Some("Is a refund requested?")
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_read_the_state_from_stdin_and_use_the_named_model() {
    let test_directory = fresh_test_directory("decide", "stdin-state");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::decisions("test/other-decider", true)],
            decisions_outcome: StubDecisionsOutcome::Completed(vec![refund_outcome(0.5)]),
            ..Default::default()
        },
    );
    let mut stdin_content = std::io::Cursor::new(b"We were billed twice.".to_vec());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(None, Some("test/other-decider")),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");
    decide_outcome.expect("the decide journey should complete against the stub daemon");

    let answers_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the decide payload should be exactly one JSON document");
    assert_eq!(answers_document["model"], "test/other-decider");
    assert_eq!(
        answers_document["answers"]["answer"]["noul"], 0.5,
        "an evenly split answer sits exactly at the boundary"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_before_contacting_the_daemon_when_the_state_is_empty() {
    let test_directory = fresh_test_directory("decide", "empty-state");
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    // No daemon listens on this socket: an empty state must be reported as
    // the usage mistake it is, not as a daemon that is not running.
    let mut decide_dependencies = decide_dependencies(
        vec![test_directory.join(SOCKET_FILE_NAME)],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(None, None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");

    assert!(
        matches!(decide_outcome, Err(DecideError::StateRequired)),
        "an empty state must fail as a usage error: {decide_outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running_for_decide() {
    let test_directory = fresh_test_directory("decide", "not-running");
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![test_directory.join(SOCKET_FILE_NAME)],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");

    assert!(
        matches!(decide_outcome, Err(DecideError::DaemonNotRunning { .. })),
        "a missing daemon must fail with the not-running error: {decide_outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_refuse_a_model_that_cannot_decide() {
    let test_directory = fresh_test_directory("decide", "not-a-decider");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            ..Default::default()
        },
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), Some("test/local-chatter")),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");

    assert!(
        matches!(&decide_outcome, Err(DecideError::ModelUnavailable { reason })
            if reason.contains("is not a decision model")),
        "a chat model must be refused as not a decision model: {decide_outcome:?}"
    );
    assert!(stdout.is_empty(), "a refusal must print nothing on stdout");
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_surface_the_models_reason_when_the_request_fails() {
    let test_directory = fresh_test_directory("decide", "model-failure");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        installed_decider_stub_config(StubDecisionsOutcome::Failed(
            DecisionFailure::invalid_request("a question needs instructions"),
        )),
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");

    assert!(
        format!("{decide_outcome:?}").contains("a question needs instructions"),
        "the failure must carry the model's reason: {decide_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_use_a_ready_catalog_decision_model_that_discovery_has_not_listed() {
    let test_directory = fresh_test_directory("decide", "catalog-ready");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            catalog_entries: vec![StubCatalogEntry::decisions(
                "test/laya-MLX-8bit",
                DEFAULT_DECISION_MODEL_ID,
                true,
            )],
            decisions_outcome: StubDecisionsOutcome::Completed(vec![refund_outcome(0.2)]),
            ..Default::default()
        },
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");
    decide_outcome.expect("a ready catalog decision model should serve the request");

    let answers_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the decide payload should be exactly one JSON document");
    assert_eq!(answers_document["answers"]["answer"]["noul"], 0.2);
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_surface_the_daemons_reason_when_it_declines_the_request() {
    let test_directory = fresh_test_directory("decide", "daemon-declines");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        installed_decider_stub_config(StubDecisionsOutcome::Rejected(
            "questions must be a JSON object".to_owned(),
        )),
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut decide_dependencies = decide_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let decide_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_decide(
            &asked_decide_arguments(Some("We were billed twice."), None),
            &mut decide_dependencies,
        ),
    )
    .await
    .expect("the decide journey should finish inside the test timeout");

    assert!(
        matches!(&decide_outcome, Err(DecideError::DecisionsRejected { reason })
            if reason == "questions must be a JSON object"),
        "a declined request must carry the daemon's reason: {decide_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
