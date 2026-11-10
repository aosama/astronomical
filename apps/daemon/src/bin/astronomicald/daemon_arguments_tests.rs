//! Beside-source tests for where one daemon invocation keeps its state: the
//! channel default, a caller-chosen root, or an app group container.

use std::ffi::OsString;
use std::path::PathBuf;

use super::{DaemonArgumentError, DaemonArguments, DaemonCommand, StateLocation};

fn parse_words(argument_words: &[&str]) -> Result<DaemonCommand, DaemonArgumentError> {
    let mut process_arguments = vec![OsString::from("astronomicald")];
    for argument_word in argument_words {
        process_arguments.push(OsString::from(argument_word));
    }
    DaemonArguments::parse(process_arguments)
}

fn parsed_state_location(argument_words: &[&str]) -> StateLocation {
    match parse_words(argument_words) {
        Ok(DaemonCommand::Run(daemon_arguments)) => daemon_arguments.state_location,
        Ok(_) => panic!("{argument_words:?} should run the daemon"),
        Err(argument_error) => panic!("{argument_words:?} should parse: {argument_error}"),
    }
}

fn parse_error_text(argument_words: &[&str]) -> String {
    match parse_words(argument_words) {
        Err(argument_error) => argument_error.to_string(),
        Ok(_) => panic!("{argument_words:?} should be refused"),
    }
}

#[test]
fn should_keep_state_in_the_channel_default_when_no_location_is_named() {
    assert_eq!(
        parsed_state_location(&["--instance", "stable"]),
        StateLocation::Default
    );
    assert_eq!(parsed_state_location(&[]), StateLocation::Default);
}

#[test]
fn should_accept_an_application_group_container_in_either_argument_order() {
    let group_container = "/Users/example/Library/Group Containers/TEAM123456.astronomical";

    for argument_words in [
        [
            "--instance",
            "stable",
            "--application-group-container",
            group_container,
        ],
        [
            "--application-group-container",
            group_container,
            "--instance",
            "stable",
        ],
    ] {
        assert_eq!(
            parsed_state_location(&argument_words),
            StateLocation::ApplicationGroupContainer(PathBuf::from(group_container))
        );
    }
}

#[test]
fn should_resolve_group_container_state_as_the_standard_stable_instance() {
    let Ok(DaemonCommand::Run(daemon_arguments)) = parse_words(&[
        "--instance",
        "stable",
        "--application-group-container",
        "/group",
    ]) else {
        panic!("a group container invocation should run the daemon");
    };

    let instance_paths = daemon_arguments
        .resolve_instance_paths()
        .expect("a group container needs no home directory to resolve");

    assert_eq!(
        instance_paths.state_directory(),
        PathBuf::from("/group/Astronomical")
    );
    assert!(instance_paths.is_standard_state_directory());
}

#[test]
fn should_still_accept_a_caller_chosen_state_directory() {
    assert_eq!(
        parsed_state_location(&["--state-directory", "/scratch/state"]),
        StateLocation::StateDirectory(PathBuf::from("/scratch/state"))
    );
}

/// One invocation has one state root: two would leave it unclear which the
/// instance lock and the models belong to.
#[test]
fn should_refuse_a_state_directory_together_with_a_group_container() {
    for argument_words in [
        [
            "--state-directory",
            "/scratch/state",
            "--application-group-container",
            "/group",
        ],
        [
            "--application-group-container",
            "/group",
            "--state-directory",
            "/scratch/state",
        ],
    ] {
        assert_eq!(
            parse_error_text(&argument_words),
            "--state-directory and --application-group-container name two state locations; supply one"
        );
    }
}

#[test]
fn should_refuse_a_location_flag_supplied_twice() {
    assert_eq!(
        parse_error_text(&[
            "--application-group-container",
            "/group",
            "--application-group-container",
            "/other"
        ]),
        "argument may be supplied only once: --application-group-container"
    );
    assert_eq!(
        parse_error_text(&["--state-directory", "/one", "--state-directory", "/two"]),
        "argument may be supplied only once: --state-directory"
    );
}

#[test]
fn should_refuse_a_group_container_that_is_missing_relative_or_the_root() {
    assert_eq!(
        parse_error_text(&["--application-group-container"]),
        "missing value for --application-group-container"
    );
    for unusable_container in ["relative/container", "/"] {
        assert_eq!(
            parse_error_text(&["--application-group-container", unusable_container]),
            format!(
                "--application-group-container must be an absolute non-root path, got {unusable_container:?}"
            )
        );
    }
}

/// Each flag's refusal names that flag, so the message points at the argument
/// the caller actually got wrong.
#[test]
fn should_name_the_state_directory_flag_when_its_value_is_relative() {
    assert_eq!(
        parse_error_text(&["--state-directory", "relative/state"]),
        "--state-directory must be an absolute non-root path, got \"relative/state\""
    );
}

#[test]
fn should_refuse_an_unknown_argument_and_a_repeated_instance() {
    assert_eq!(
        parse_error_text(&["--group-container", "/group"]),
        "unrecognized argument: --group-container"
    );
    assert_eq!(
        parse_error_text(&["--instance", "stable", "--instance", "development"]),
        "argument may be supplied only once: --instance"
    );
}

#[test]
fn should_name_the_group_container_flag_in_the_help_text() {
    assert!(super::help_text().contains("--application-group-container PATH"));
}
