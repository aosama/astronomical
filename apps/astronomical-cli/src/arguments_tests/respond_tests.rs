//! Parsing of the `respond` verb: positional or `--text` prompt, repeatable
//! `--image` flags, the generation controls, and the `--schema` file flag.

use std::path::PathBuf;

use crate::arguments::CliCommand;
use crate::errors::UsageError;
use crate::respond_arguments::RespondArguments;

use super::parse;

/// Builds `respond` arguments with only the prompt, model, and stream mode
/// set, for tests that exercise one dimension at a time.
fn respond_arguments(prompt: &str, model_id: Option<&str>, no_stream: bool) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: model_id.map(str::to_owned),
        instructions: None,
        thinking_budget: None,
        schema_path: None,
        no_stream,
    }
}

/// Builds `respond` arguments with the generation controls set: the
/// instructions and the thinking budget.
fn respond_arguments_with(
    prompt: &str,
    instructions: Option<&str>,
    thinking_budget: Option<u16>,
) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: None,
        instructions: instructions.map(str::to_owned),
        thinking_budget,
        schema_path: None,
        no_stream: false,
    }
}

/// Builds `respond` arguments with the structured-output schema file set.
fn respond_arguments_with_schema(prompt: &str, schema_path: Option<PathBuf>) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: None,
        instructions: None,
        thinking_budget: None,
        schema_path,
        no_stream: false,
    }
}

#[test]
fn should_parse_respond_with_a_bare_prompt() {
    let parsed_command =
        parse(&["respond", "Hello there"]).expect("a bare respond prompt should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments("Hello there", None, false))
    );
}

#[test]
fn should_parse_respond_prompt_with_model_and_no_stream() {
    let parsed_command = parse(&[
        "respond",
        "Hello there",
        "--model",
        "test/model",
        "--no-stream",
    ])
    .expect("respond with --model and --no-stream should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments("Hello there", Some("test/model"), true))
    );
}

#[test]
fn should_reject_respond_without_a_prompt_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond"]),
        Err(UsageError::RespondPromptRequired)
    ));
}

#[test]
fn should_reject_respond_with_an_unknown_argument_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond", "Hi", "--bogus"]),
        Err(UsageError::UnknownArgument(argument)) if argument == "--bogus"
    ));
}

#[test]
fn should_parse_repeatable_image_arguments_after_the_prompt() {
    let parsed_command = parse(&[
        "respond",
        "Describe this",
        "--image",
        "snapshot.png",
        "--image",
        "photo.jpg",
    ])
    .expect("respond with repeatable --image should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(RespondArguments {
            prompt: "Describe this".to_owned(),
            images: vec![PathBuf::from("snapshot.png"), PathBuf::from("photo.jpg")],
            model_id: None,
            instructions: None,
            thinking_budget: None,
            schema_path: None,
            no_stream: false,
        })
    );
}

#[test]
fn should_parse_image_arguments_before_the_prompt() {
    let parsed_command = parse(&["respond", "--image", "a.png", "Hello there", "--no-stream"])
        .expect("respond with --image before the prompt should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(RespondArguments {
            prompt: "Hello there".to_owned(),
            images: vec![PathBuf::from("a.png")],
            model_id: None,
            instructions: None,
            thinking_budget: None,
            schema_path: None,
            no_stream: true,
        })
    );
}

#[test]
fn should_reject_an_image_flag_missing_a_value_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond", "Hello", "--image"]),
        Err(UsageError::MissingValue("--image"))
    ));
}

#[test]
fn should_reject_an_image_flag_with_a_flag_shaped_value_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond", "Hello", "--image", "--no-stream"]),
        Err(UsageError::MissingValue("--image"))
    ));
}

#[test]
fn should_parse_respond_prompt_from_the_text_flag() {
    let parsed_command =
        parse(&["respond", "--text", "Hello there"]).expect("--text should parse as the prompt");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments("Hello there", None, false))
    );
}

#[test]
fn should_parse_respond_instructions_flag() {
    let parsed_command = parse(&[
        "respond",
        "Hello there",
        "--instructions",
        "Answer in one word",
    ])
    .expect("respond with --instructions should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments_with(
            "Hello there",
            Some("Answer in one word"),
            None
        ))
    );
}

#[test]
fn should_parse_respond_thinking_budget_flag() {
    let parsed_command = parse(&["respond", "Hello there", "--thinking-budget", "512"])
        .expect("respond with --thinking-budget should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments_with("Hello there", None, Some(512)))
    );
}

#[test]
fn should_parse_the_schema_flag_value_as_a_path() {
    let parsed_command = parse(&["respond", "Hello there", "--schema", "answer-schema.json"])
        .expect("respond with --schema should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(RespondArguments {
            schema_path: Some(PathBuf::from("answer-schema.json")),
            ..respond_arguments_with_schema("Hello there", None)
        })
    );
}

#[test]
fn should_reject_a_repeated_schema_flag_as_a_repeated_argument() {
    assert!(matches!(
        parse(&["respond", "--schema", "a.json", "--schema", "b.json"]),
        Err(UsageError::RepeatedArgument("--schema"))
    ));
}

#[test]
fn should_parse_a_zero_thinking_budget_as_a_valid_disable() {
    let parsed_command = parse(&["respond", "Hello there", "--thinking-budget", "0"])
        .expect("a zero thinking budget should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments_with("Hello there", None, Some(0)))
    );
}

#[test]
fn should_reject_respond_with_both_a_positional_prompt_and_text_flag() {
    assert!(matches!(
        parse(&["respond", "Hello there", "--text", "Other"]),
        Err(UsageError::RespondPromptConflict)
    ));
}

#[test]
fn should_reject_a_repeated_text_flag_as_a_repeated_argument() {
    assert!(matches!(
        parse(&["respond", "--text", "One", "--text", "Two"]),
        Err(UsageError::RepeatedArgument("--text"))
    ));
}

#[test]
fn should_reject_respond_with_two_positional_prompts() {
    assert!(matches!(
        parse(&["respond", "One", "Two"]),
        Err(UsageError::RespondPromptConflict)
    ));
}

#[test]
fn should_reject_a_non_numeric_thinking_budget() {
    assert!(matches!(
        parse(&["respond", "Hello there", "--thinking-budget", "lots"]),
        Err(UsageError::InvalidThinkingBudget(value)) if value == "lots"
    ));
}

#[test]
fn should_reject_a_thinking_budget_above_the_u16_maximum() {
    assert!(matches!(
        parse(&["respond", "Hello there", "--thinking-budget", "70000"]),
        Err(UsageError::InvalidThinkingBudget(value)) if value == "70000"
    ));
}

#[test]
fn should_reject_missing_values_for_the_new_respond_flags() {
    assert!(matches!(
        parse(&["respond", "--text"]),
        Err(UsageError::MissingValue("--text"))
    ));
    assert!(matches!(
        parse(&["respond", "Hello there", "--instructions"]),
        Err(UsageError::MissingValue("--instructions"))
    ));
    assert!(matches!(
        parse(&["respond", "Hello there", "--thinking-budget"]),
        Err(UsageError::MissingValue("--thinking-budget"))
    ));
}

#[test]
fn should_parse_all_new_respond_flags_together() {
    let parsed_command = parse(&[
        "respond",
        "--text",
        "Describe this",
        "--instructions",
        "Answer in one word",
        "--thinking-budget",
        "512",
        "--no-stream",
    ])
    .expect("respond with every new flag together should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(RespondArguments {
            prompt: "Describe this".to_owned(),
            images: Vec::new(),
            model_id: None,
            instructions: Some("Answer in one word".to_owned()),
            thinking_budget: Some(512),
            schema_path: None,
            no_stream: true,
        })
    );
}

#[test]
fn should_parse_the_maximum_thinking_budget() {
    let parsed_command = parse(&["respond", "Hello there", "--thinking-budget", "65535"])
        .expect("the inclusive upper bound should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments_with("Hello there", None, Some(65_535)))
    );
}
