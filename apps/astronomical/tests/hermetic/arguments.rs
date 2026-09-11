use std::ffi::OsString;

use astronomical_cli::errors::UsageError;
use astronomical_cli::{CliCommand, LaunchArguments, parse_command};

fn parse(arguments: &[&str]) -> Result<CliCommand, UsageError> {
    let process_arguments =
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from));
    parse_command(process_arguments)
}

#[test]
fn should_print_help_for_help_flag() {
    assert_eq!(parse(&["--help"]).expect("help"), CliCommand::Help);
    assert_eq!(
        parse(&["launch", "-h"]).expect("launch help"),
        CliCommand::Help
    );
    assert!(
        astronomical_cli::help_text()
            .contains("       astronomical launch opencode [--model MODEL_ID]")
    );
}

#[test]
fn should_ignore_verbose_as_a_global_flag() {
    assert_eq!(
        parse(&["--verbose", "launch", "opencode"]).expect("verbose then launch"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
        })
    );
    assert_eq!(
        parse(&["launch", "-v", "opencode"]).expect("launch verbose"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
        })
    );
}

#[test]
fn should_print_version_for_version_flag() {
    assert_eq!(parse(&["--version"]).expect("version"), CliCommand::Version);
}

#[test]
fn should_require_a_command_when_invoked_bare() {
    match parse(&[]) {
        Err(UsageError::MissingCommand) => {}
        other => panic!("expected missing command, got {other:?}"),
    }
}

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
fn should_reject_unknown_tools_as_launch_arguments_not_usage() {
    assert_eq!(
        parse(&["launch", "copilot"]).expect("copilot is a launch argument"),
        CliCommand::Launch(LaunchArguments {
            tool_slug: Some("copilot".to_owned()),
            model_id: None,
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
