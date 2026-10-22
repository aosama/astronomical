//! Parsing of the `launch` verb: optional tool slug and `--model` flag in
//! either order, with unknown tool names accepted as launch arguments.

use crate::arguments::CliCommand;
use crate::arguments::LaunchArguments;
use crate::errors::UsageError;

use super::parse;

#[test]
fn should_parse_launch_without_a_tool_name() {
    assert_eq!(
        parse(&["launch"]).expect("launch"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: None,
            model_id: None,
        })
    );
}

#[test]
fn should_parse_launch_opencode_and_model_flag_in_either_order() {
    let expected = CliCommand::Launch(LaunchArguments {
        tool_slug: Some("opencode".to_owned()),
        model_id: Some("library-chat-model".to_owned()),
    });
    assert_eq!(
        parse(&["launch", "opencode", "--model", "library-chat-model"]).expect("tool then model"),
        expected
    );
    assert_eq!(
        parse(&["launch", "--model", "library-chat-model", "opencode"]).expect("model then tool"),
        expected
    );
}

#[test]
fn should_accept_unknown_tools_as_launch_arguments_not_usage() {
    assert_eq!(
        parse(&["launch", "copilot"]).expect("copilot is a launch argument"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("copilot".to_owned()),
            model_id: None,
        })
    );
}

#[test]
fn should_parse_launch_codex_with_a_model_flag() {
    assert_eq!(
        parse(&["launch", "codex", "--model", "library-chat-model"]).expect("codex with model"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("codex".to_owned()),
            model_id: Some("library-chat-model".to_owned()),
        })
    );
}

#[test]
fn should_reject_two_tool_names() {
    match parse(&["launch", "opencode", "pi"]) {
        Err(UsageError::MultipleTools) => {}
        other => panic!("expected multiple tools, got {other:?}"),
    }
}

#[test]
fn should_reject_a_missing_model_value() {
    match parse(&["launch", "--model"]) {
        Err(UsageError::MissingValue("--model")) => {}
        other => panic!("expected missing model value, got {other:?}"),
    }
}
