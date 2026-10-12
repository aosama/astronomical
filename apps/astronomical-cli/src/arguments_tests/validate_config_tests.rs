//! Parsing of the `validate config` verb: the default instance, the
//! `--instance` flag, and the required config noun.

use crate::arguments::CliCommand;
use crate::validate_config_arguments::ValidateConfigArguments;

use super::parse;

fn parsed_validate_config(arguments: &[&str]) -> ValidateConfigArguments {
    let mut full_arguments = vec!["validate", "config"];
    full_arguments.extend_from_slice(arguments);
    match parse(&full_arguments) {
        Ok(CliCommand::ValidateConfig(validate_arguments)) => validate_arguments,
        other => panic!("expected validate config command, got {other:?}"),
    }
}

#[test]
fn should_default_validate_config_to_development_instance() {
    let validate_arguments = parsed_validate_config(&[]);
    assert_eq!(
        validate_arguments.runtime_instance,
        astronomical_config::AstronomicalRuntimeInstance::Development
    );
    assert!(!validate_arguments.render_json);
}

#[test]
fn should_parse_stable_instance_flag() {
    let validate_arguments = parsed_validate_config(&["--instance", "stable"]);
    assert_eq!(
        validate_arguments.runtime_instance,
        astronomical_config::AstronomicalRuntimeInstance::Stable
    );
}

#[test]
fn should_reject_unknown_instance_name() {
    let usage_error =
        parse(&["validate", "config", "--instance", "beta"]).expect_err("unknown instance");
    assert!(usage_error.to_string().contains("stable"));
}

#[test]
fn should_reject_validate_without_config_noun() {
    let usage_error = parse(&["validate", "models"]).expect_err("only config is supported");
    assert!(usage_error.to_string().contains("config"));
}
