//! Parsing of the `embed` verb: bare text, `--file`, or neither for stdin,
//! with the text-and-file combination rejected.

use std::path::PathBuf;

use crate::arguments::CliCommand;
use crate::embed_arguments::EmbedArguments;
use crate::errors::UsageError;

use super::parse;

fn embed_arguments(
    text: Option<&str>,
    file_path: Option<PathBuf>,
    model_id: Option<&str>,
) -> EmbedArguments {
    EmbedArguments {
        text: text.map(str::to_owned),
        file_path,
        model_id: model_id.map(str::to_owned),
    }
}

#[test]
fn should_parse_embed_with_a_bare_text_argument() {
    let parsed_command = parse(&["embed", "hello world"]).expect("a bare embed text should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(Some("hello world"), None, None))
    );
}

#[test]
fn should_parse_embed_with_model_and_file() {
    let parsed_command = parse(&[
        "embed",
        "--model",
        "test/local-embedder",
        "--file",
        "input.txt",
    ])
    .expect("embed with --model and --file should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(
            None,
            Some(PathBuf::from("input.txt")),
            Some("test/local-embedder")
        ))
    );
}

#[test]
fn should_parse_embed_with_no_input_for_stdin_mode() {
    let parsed_command = parse(&["embed"]).expect("embed without input should parse for stdin");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(None, None, None))
    );
}

#[test]
fn should_reject_embed_with_an_unknown_argument_as_a_usage_error() {
    assert!(matches!(
        parse(&["embed", "hello", "--unknown"]),
        Err(UsageError::UnknownArgument(unknown_argument)) if unknown_argument == "--unknown"
    ));
}

#[test]
fn should_reject_embed_with_both_text_and_file_as_a_usage_error() {
    assert!(matches!(
        parse(&["embed", "hello", "--file", "input.txt"]),
        Err(UsageError::EmbedInputConflict)
    ));
}
