//! Parsing of the `launch` verb: optional tool slug and `--model` flag in
//! either order, with unknown tool names accepted as launch arguments.

use crate::arguments::CliCommand;
use crate::arguments::LaunchArguments;
use crate::arguments::UsageError;

use super::parse;

#[test]
fn should_parse_launch_without_a_tool_name() {
    assert_eq!(
        parse(&["launch"]).expect("launch"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: None,
            model_id: None,
            harness_arguments: Vec::new(),
        })
    );
}

#[test]
fn should_parse_launch_opencode_and_model_flag_in_either_order() {
    let expected = CliCommand::Launch(LaunchArguments {
        tool_slug: Some("opencode".to_owned()),
        model_id: Some("library-chat-model".to_owned()),
        harness_arguments: Vec::new(),
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
            harness_arguments: Vec::new(),
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
            harness_arguments: Vec::new(),
        })
    );
}

#[test]
fn should_pass_arguments_after_a_double_dash_to_the_harness_unchanged() {
    assert_eq!(
        parse(&[
            "launch",
            "claude",
            "--model",
            "library-chat-model",
            "--",
            "-p",
            "--model",
            "x"
        ])
        .expect("claude with harness arguments"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: Some("library-chat-model".to_owned()),
            harness_arguments: vec!["-p".into(), "--model".into(), "x".into()],
        })
    );
}

#[test]
fn should_accept_a_trailing_double_dash_with_no_harness_arguments() {
    assert_eq!(
        parse(&["launch", "claude", "--"]).expect("claude with an empty pass-through"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: None,
            harness_arguments: Vec::new(),
        })
    );
}

#[test]
fn should_still_reject_an_unknown_launcher_flag_before_the_double_dash() {
    match parse(&["launch", "claude", "-p", "--", "hello"]) {
        Err(UsageError::UnknownArgument(unknown_argument)) => {
            assert_eq!(unknown_argument, "-p");
        }
        other => panic!("expected an unknown launcher argument, got {other:?}"),
    }
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
