//! Parsing of the `models` verb: the list, supported, default, and download
//! subcommands and their required values.

use crate::arguments::CliCommand;
use crate::arguments::UsageError;
use crate::verbs::models::ModelsCommand;

use super::parse;

#[test]
fn should_parse_models_subcommands() {
    assert_eq!(
        parse(&["models", "list"]),
        Ok(CliCommand::Models(ModelsCommand::List))
    );
    assert_eq!(
        parse(&["models", "supported"]),
        Ok(CliCommand::Models(ModelsCommand::Supported))
    );
    assert_eq!(
        parse(&["models", "default"]),
        Ok(CliCommand::Models(ModelsCommand::Default {
            model_id: None
        }))
    );
    assert_eq!(
        parse(&["models", "default", "test/model"]),
        Ok(CliCommand::Models(ModelsCommand::Default {
            model_id: Some("test/model".to_owned())
        }))
    );
    assert_eq!(
        parse(&["models", "download", "test/model"]),
        Ok(CliCommand::Models(ModelsCommand::Download {
            model_id: "test/model".to_owned()
        }))
    );
}

#[test]
fn should_reject_models_without_a_subcommand_as_a_usage_error() {
    assert!(matches!(
        parse(&["models"]),
        Err(UsageError::ModelsSubcommandRequired)
    ));
}

#[test]
fn should_reject_models_with_an_unknown_subcommand_as_a_usage_error() {
    assert!(matches!(
        parse(&["models", "reboot"]),
        Err(UsageError::UnknownModelsSubcommand(subcommand)) if subcommand == "reboot"
    ));
}

#[test]
fn should_reject_models_download_without_a_model_id_as_a_usage_error() {
    assert!(matches!(
        parse(&["models", "download"]),
        Err(UsageError::ModelsDownloadModelRequired)
    ));
}
