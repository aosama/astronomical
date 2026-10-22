//! Unit tests for `crate::arguments`: command-line parsing. The per-verb
//! grammar lives in the child modules; this root holds the shared parse
//! entry point and the global-flag behavior.

mod embed_tests;
mod launch_tests;
mod models_tests;
mod respond_tests;
mod validate_config_tests;

use std::ffi::OsString;

use crate::arguments::{CliCommand, help_text, parse_command};
use crate::errors::UsageError;

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
        })
    );
    assert_eq!(
        parse(&["launch", "-v", "opencode"]).expect("launch verbose"),
        CliCommand::Launch(crate::arguments::LaunchArguments {
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
fn should_parse_status_as_a_plain_verb() {
    assert_eq!(parse(&["status"]), Ok(CliCommand::Status));
}
