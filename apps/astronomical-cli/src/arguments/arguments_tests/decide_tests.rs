//! Parsing of the `decide` verb: one state source at most (text, file, or
//! neither for stdin) and exactly one questions source.

use std::path::PathBuf;

use crate::arguments::CliCommand;
use crate::arguments::UsageError;
use crate::verbs::decide::{DecideArguments, DecideQuestionsSource};

use super::parse;

fn asked_decide_arguments(state_text: Option<&str>, question_text: &str) -> DecideArguments {
    DecideArguments {
        state_text: state_text.map(str::to_owned),
        state_file_path: None,
        state_is_json: false,
        questions_source: DecideQuestionsSource::Ask(question_text.to_owned()),
        model_id: None,
    }
}

#[test]
fn should_parse_decide_with_a_state_and_one_asked_question() {
    let parsed_command = parse(&[
        "decide",
        "We were billed twice.",
        "--ask",
        "Is it a refund?",
    ])
    .expect("a state with --ask should parse");

    assert_eq!(
        parsed_command,
        CliCommand::Decide(asked_decide_arguments(
            Some("We were billed twice."),
            "Is it a refund?"
        ))
    );
}

#[test]
fn should_parse_decide_with_no_state_for_stdin_mode() {
    let parsed_command = parse(&["decide", "--ask", "Is it a refund?"])
        .expect("decide without a state should parse");

    assert_eq!(
        parsed_command,
        CliCommand::Decide(asked_decide_arguments(None, "Is it a refund?"))
    );
}

#[test]
fn should_parse_decide_with_files_a_json_state_and_a_model() {
    let parsed_command = parse(&[
        "decide",
        "--state-file",
        "ticket.json",
        "--json-state",
        "--questions-file",
        "questions.json",
        "--model",
        "test/local-decider",
    ])
    .expect("decide with file sources should parse");

    assert_eq!(
        parsed_command,
        CliCommand::Decide(DecideArguments {
            state_text: None,
            state_file_path: Some(PathBuf::from("ticket.json")),
            state_is_json: true,
            questions_source: DecideQuestionsSource::File(PathBuf::from("questions.json")),
            model_id: Some("test/local-decider".to_owned()),
        })
    );
}

#[test]
fn should_parse_decide_with_inline_questions_json() {
    let parsed_command = parse(&["decide", "state", "--questions", "{}"])
        .expect("decide with inline questions should parse");

    let CliCommand::Decide(decide_arguments) = parsed_command else {
        panic!("decide must parse to the decide command");
    };
    assert_eq!(
        decide_arguments.questions_source,
        DecideQuestionsSource::InlineJson("{}".to_owned())
    );
}

#[test]
fn should_reject_decide_without_any_questions_as_a_usage_error() {
    assert_eq!(
        parse(&["decide", "state"]),
        Err(UsageError::DecideQuestionsRequired)
    );
}

#[test]
fn should_reject_decide_with_two_question_sources_as_a_usage_error() {
    assert_eq!(
        parse(&["decide", "state", "--ask", "Is it?", "--questions", "{}"]),
        Err(UsageError::DecideQuestionsConflict)
    );
}

#[test]
fn should_reject_decide_with_both_a_state_and_a_state_file_as_a_usage_error() {
    assert_eq!(
        parse(&[
            "decide",
            "state",
            "--state-file",
            "s.txt",
            "--ask",
            "Is it?"
        ]),
        Err(UsageError::DecideStateConflict)
    );
}

#[test]
fn should_reject_decide_with_a_second_positional_state_as_a_usage_error() {
    assert_eq!(
        parse(&["decide", "first", "second", "--ask", "Is it?"]),
        Err(UsageError::UnknownArgument("second".to_owned()))
    );
}

#[test]
fn should_reject_decide_with_a_repeated_flag_as_a_usage_error() {
    assert_eq!(
        parse(&["decide", "state", "--ask", "Is it?", "--ask", "Is it not?"]),
        Err(UsageError::RepeatedArgument("--ask"))
    );
}

#[test]
fn should_reject_decide_with_an_unknown_flag_as_a_usage_error() {
    assert_eq!(
        parse(&["decide", "state", "--ask", "Is it?", "--unknown"]),
        Err(UsageError::UnknownArgument("--unknown".to_owned()))
    );
}
