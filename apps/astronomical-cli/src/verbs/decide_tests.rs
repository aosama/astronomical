//! Unit contracts for turning the daemon's one terminal frame into the
//! printed document, and for which failures count as usage mistakes.

use astronomical_decision_contract::DecisionFailure;
use astronomical_ipc_protocol::DaemonResponse;

use super::{DecideError, write_decisions_outcome};

#[test]
fn should_print_a_completed_frame_as_one_json_line() {
    let mut stdout = Vec::new();

    write_decisions_outcome(
        DaemonResponse::DecisionsCompleted {
            model: "test/local-decider".to_owned(),
            question_outcomes: Vec::new(),
            input_token_count: 12,
            elapsed_millis: 3,
        },
        &mut stdout,
    )
    .expect("a completed frame should print");

    let printed_text = String::from_utf8(stdout).expect("the document should be UTF-8");
    assert!(printed_text.ends_with('\n'));
    assert_eq!(printed_text.lines().count(), 1);
    let answers_document: serde_json::Value =
        serde_json::from_str(&printed_text).expect("the line should be one JSON document");
    assert_eq!(answers_document["model"], "test/local-decider");
    assert_eq!(answers_document["usage"]["input_tokens"], 12);
}

#[test]
fn should_surface_a_failed_frame_with_the_models_reason() {
    let mut stdout = Vec::new();

    let write_outcome = write_decisions_outcome(
        DaemonResponse::DecisionsFailed {
            reason: DecisionFailure::InvalidRequest {
                reason: "a choice question needs at least two options".to_owned(),
            },
        },
        &mut stdout,
    );

    let decide_error = write_outcome.expect_err("a failed frame must be an error");
    assert!(
        decide_error
            .to_string()
            .contains("a choice question needs at least two options")
    );
    assert!(stdout.is_empty(), "a failure must print nothing on stdout");
}

#[test]
fn should_treat_an_unrelated_frame_as_a_daemon_that_stopped_responding() {
    let mut stdout = Vec::new();

    let write_outcome =
        write_decisions_outcome(DaemonResponse::DownloadStatus { job: None }, &mut stdout);

    assert!(matches!(
        write_outcome,
        Err(DecideError::DaemonStoppedResponding)
    ));
}

#[test]
fn should_class_input_mistakes_as_usage_errors_and_daemon_trouble_as_failures() {
    assert!(DecideError::StateRequired.is_usage_error());
    assert!(
        DecideError::QuestionsNotJson {
            cause: String::new()
        }
        .is_usage_error()
    );
    assert!(!DecideError::DaemonStoppedResponding.is_usage_error());
    assert!(
        !DecideError::DecisionsFailed {
            reason: DecisionFailure::EngineBusy
        }
        .is_usage_error()
    );
}
