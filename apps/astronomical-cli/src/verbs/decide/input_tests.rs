//! Unit contracts for reading a `decide` invocation's state and questions:
//! every source they can come from, and every way the input can be wrong.

use std::path::PathBuf;

use astronomical_decision_contract::OrderedJsonValue;

use super::{ASK_QUESTION_NAME, resolve_questions, resolve_state};
use crate::verbs::decide::{DecideArguments, DecideError, DecideQuestionsSource};

fn decide_arguments(state_text: Option<&str>, state_is_json: bool) -> DecideArguments {
    DecideArguments {
        state_text: state_text.map(str::to_owned),
        state_file_path: None,
        state_is_json,
        questions_source: DecideQuestionsSource::Ask("Is a refund requested?".to_owned()),
        model_id: None,
    }
}

#[test]
fn should_turn_an_asked_question_into_one_noul_question() {
    let questions = resolve_questions(&DecideQuestionsSource::Ask(
        "Is a refund requested?".to_owned(),
    ))
    .expect("an asked question should resolve");

    let question_definition = questions
        .member(ASK_QUESTION_NAME)
        .expect("the asked question should be named");
    assert_eq!(
        question_definition
            .member("type")
            .and_then(OrderedJsonValue::as_text),
        Some("noul")
    );
    assert_eq!(
        question_definition
            .member("instructions")
            .and_then(OrderedJsonValue::as_text),
        Some("Is a refund requested?")
    );
}

#[test]
fn should_reject_a_blank_asked_question() {
    let questions_outcome = resolve_questions(&DecideQuestionsSource::Ask("  \n".to_owned()));

    assert!(matches!(questions_outcome, Err(DecideError::EmptyQuestion)));
}

#[test]
fn should_keep_inline_questions_in_the_order_they_were_written() {
    let questions = resolve_questions(&DecideQuestionsSource::InlineJson(
        r#"{"zeta": {"type": "noul"}, "alpha": {"type": "noul"}}"#.to_owned(),
    ))
    .expect("inline questions should parse");

    let question_names: Vec<&str> = questions
        .as_object()
        .expect("the questions should be an object")
        .iter()
        .map(|question_member| question_member.key.as_str())
        .collect();
    assert_eq!(question_names, ["zeta", "alpha"]);
}

#[test]
fn should_reject_inline_questions_that_are_not_json() {
    let questions_outcome =
        resolve_questions(&DecideQuestionsSource::InlineJson("{not json".to_owned()));

    assert!(matches!(
        questions_outcome,
        Err(DecideError::QuestionsNotJson { .. })
    ));
}

#[test]
fn should_report_an_unreadable_questions_file_by_its_path() {
    let missing_file_path = PathBuf::from("this-questions-file-does-not-exist.json");

    let questions_outcome =
        resolve_questions(&DecideQuestionsSource::File(missing_file_path.clone()));

    assert!(matches!(
        questions_outcome,
        Err(DecideError::InputFileUnreadable { file_path, .. }) if file_path == missing_file_path
    ));
}

#[test]
fn should_read_a_plain_state_as_text_even_when_it_looks_like_json() {
    let mut stdin = std::io::empty();

    let state = resolve_state(&decide_arguments(Some(r#"{"a": 1}"#), false), &mut stdin)
        .expect("a text state should resolve");

    assert_eq!(state, OrderedJsonValue::Text(r#"{"a": 1}"#.to_owned()));
}

#[test]
fn should_read_a_json_state_as_structured_data_when_asked() {
    let mut stdin = std::io::empty();

    let state = resolve_state(&decide_arguments(Some(r#"{"a": 1}"#), true), &mut stdin)
        .expect("a JSON state should resolve");

    assert_eq!(state.member("a"), Some(&OrderedJsonValue::SignedInteger(1)));
}

#[test]
fn should_read_the_state_from_stdin_when_none_is_given() {
    let mut stdin = std::io::Cursor::new(b"We were billed twice.".to_vec());

    let state = resolve_state(&decide_arguments(None, false), &mut stdin)
        .expect("a stdin state should resolve");

    assert_eq!(
        state,
        OrderedJsonValue::Text("We were billed twice.".to_owned())
    );
}

#[test]
fn should_reject_a_state_that_is_only_whitespace() {
    let mut stdin = std::io::Cursor::new(b" \n".to_vec());

    let state_outcome = resolve_state(&decide_arguments(None, false), &mut stdin);

    assert!(matches!(state_outcome, Err(DecideError::StateRequired)));
}

#[test]
fn should_reject_a_json_state_that_does_not_parse() {
    let mut stdin = std::io::empty();

    let state_outcome = resolve_state(&decide_arguments(Some("{broken"), true), &mut stdin);

    assert!(matches!(
        state_outcome,
        Err(DecideError::StateNotJson { .. })
    ));
}

#[test]
fn should_read_the_questions_from_a_file() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let questions_file_path = temporary_directory.path().join("questions.json");
    std::fs::write(
        &questions_file_path,
        r#"{"urgent": {"type": "noul", "instructions": "Is it urgent?"}}"#,
    )
    .expect("the questions file should be writable");

    let questions = resolve_questions(&DecideQuestionsSource::File(questions_file_path))
        .expect("a questions file should resolve");

    assert!(questions.member("urgent").is_some());
}

#[test]
fn should_read_the_state_from_a_file() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let state_file_path = temporary_directory.path().join("state.txt");
    std::fs::write(&state_file_path, "Once more unto the breach, dear friends.")
        .expect("the state file should be writable");
    let mut stdin = std::io::Cursor::new(b"standard input must not be read".to_vec());
    let mut file_arguments = decide_arguments(None, false);
    file_arguments.state_file_path = Some(state_file_path);

    let state = resolve_state(&file_arguments, &mut stdin).expect("a state file should resolve");

    assert_eq!(
        state,
        OrderedJsonValue::Text("Once more unto the breach, dear friends.".to_owned())
    );
}

#[test]
fn should_report_an_unreadable_state_file_by_its_path() {
    let missing_file_path = PathBuf::from("this-state-file-does-not-exist.txt");
    let mut stdin = std::io::empty();
    let mut file_arguments = decide_arguments(None, false);
    file_arguments.state_file_path = Some(missing_file_path.clone());

    let state_outcome = resolve_state(&file_arguments, &mut stdin);

    assert!(matches!(
        state_outcome,
        Err(DecideError::InputFileUnreadable { file_path, .. }) if file_path == missing_file_path
    ));
}
