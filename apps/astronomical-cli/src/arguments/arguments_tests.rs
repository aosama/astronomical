//! Unit tests for `crate::arguments`: command-line parsing. The per-verb
//! grammar lives in the child modules; this root holds the shared parse
//! entry point and the global-flag behavior.

mod decide_tests;
mod embed_tests;
mod launch_tests;
mod models_tests;
mod quantize_tests;
mod respond_tests;
mod validate_config_tests;

use std::ffi::OsString;

use crate::arguments::UsageError;
use crate::arguments::{CliCommand, help_text, is_verbose_requested, parse_command};

fn process_arguments(arguments: &[&str]) -> Vec<OsString> {
    let mut process_arguments = vec![OsString::from("astronomical")];
    for argument in arguments {
        process_arguments.push(OsString::from(argument));
    }
    process_arguments
}

#[test]
fn should_request_verbose_tracing_from_a_launcher_flag() {
    assert!(is_verbose_requested(&process_arguments(&["-v", "launch"])));
    assert!(is_verbose_requested(&process_arguments(&[
        "launch",
        "--verbose"
    ])));
}

#[test]
fn should_not_request_verbose_tracing_from_a_harness_flag() {
    assert!(!is_verbose_requested(&process_arguments(&[
        "launch",
        "claude",
        "--",
        "-p",
        "hi",
        "--verbose"
    ])));
}

#[test]
fn should_not_request_verbose_tracing_without_arguments() {
    assert!(!is_verbose_requested(&process_arguments(&[])));
    assert!(!is_verbose_requested(&[]));
}

#[test]
fn should_strip_the_launcher_verbose_flag_but_keep_the_harness_one() {
    assert!(is_verbose_requested(&process_arguments(&[
        "launch", "-v", "claude", "--", "-v"
    ])));
    assert_eq!(
        parse(&["launch", "-v", "claude", "--", "-v"]).expect("both verbose flags"),
        CliCommand::Launch(crate::arguments::LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: None,
            harness_arguments: vec!["-v".into()],
        })
    );
}

/// Parses arguments as the CLI would see them after the binary name.
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
    assert!(help_text().contains("       astronomical launch opencode [--model MODEL_ID]"));
    assert!(help_text().contains("       astronomical launch codex [--model MODEL_ID]"));
}

#[test]
fn should_ignore_verbose_as_a_global_flag() {
    assert_eq!(
        parse(&["--verbose", "launch", "opencode"]).expect("verbose then launch"),
        CliCommand::Launch(crate::arguments::LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
            harness_arguments: Vec::new(),
        })
    );
    assert_eq!(
        parse(&["launch", "-v", "opencode"]).expect("launch verbose"),
        CliCommand::Launch(crate::arguments::LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
            harness_arguments: Vec::new(),
        })
    );
}

#[test]
fn should_leave_help_version_and_verbose_after_a_double_dash_to_the_harness() {
    assert_eq!(
        parse(&["launch", "claude", "--", "--help", "--version", "-v"])
            .expect("harness flags pass through"),
        CliCommand::Launch(crate::arguments::LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: None,
            harness_arguments: vec!["--help".into(), "--version".into(), "-v".into()],
        })
    );
}

#[test]
fn should_still_honor_help_before_a_double_dash() {
    assert_eq!(
        parse(&["launch", "--help", "--", "-p"]).expect("launcher help"),
        CliCommand::Help
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
fn should_parse_status_as_a_plain_verb() {
    assert_eq!(parse(&["status"]), Ok(CliCommand::Status));
}
