//! The `vision` grammar. The verb's behavior is covered by the filesystem journeys
//! under `tests/integration/vision.rs`.

use std::ffi::OsString;

use crate::arguments::UsageError;

use super::*;

fn parse(argument_texts: &[&str]) -> Result<VisionCommand, UsageError> {
    let mut arguments = Vec::new();
    for argument_text in argument_texts {
        arguments.push(OsString::from(argument_text));
    }
    parse_vision_command(&arguments)
}

#[test]
fn should_parse_inspect_with_and_without_json() {
    assert_eq!(
        parse(&["inspect", "org/model"]).unwrap(),
        VisionCommand::Inspect(VisionInspectArguments {
            model: "org/model".to_owned(),
            render_json: false,
        })
    );
    assert_eq!(
        parse(&["inspect", "--json", "org/model"]).unwrap(),
        VisionCommand::Inspect(VisionInspectArguments {
            model: "org/model".to_owned(),
            render_json: true,
        })
    );
}

#[test]
fn should_keep_the_text_and_tower_models_in_order() {
    assert_eq!(
        parse(&["match", "org/text", "org/tower", "--json"]).unwrap(),
        VisionCommand::Match(VisionMatchArguments {
            text_model: "org/text".to_owned(),
            tower_model: "org/tower".to_owned(),
            render_json: true,
        })
    );
}

#[test]
fn should_refuse_missing_models() {
    assert!(matches!(parse(&[]), Err(UsageError::MissingCommand)));
    assert!(matches!(
        parse(&["inspect"]),
        Err(UsageError::MissingValue("MODEL"))
    ));
    assert!(matches!(
        parse(&["match"]),
        Err(UsageError::MissingValue("TEXT_MODEL"))
    ));
    assert!(matches!(
        parse(&["match", "org/text"]),
        Err(UsageError::MissingValue("TOWER_MODEL"))
    ));
}

#[test]
fn should_refuse_extra_arguments_unknown_flags_and_unknown_targets() {
    assert!(matches!(
        parse(&["inspect", "org/one", "org/two"]),
        Err(UsageError::UnknownArgument(_))
    ));
    assert!(matches!(
        parse(&["match", "org/a", "org/b", "org/c"]),
        Err(UsageError::UnknownArgument(_))
    ));
    assert!(matches!(
        parse(&["inspect", "--verbose-output"]),
        Err(UsageError::UnknownArgument(_))
    ));
    assert!(matches!(
        parse(&["inspect", "org/model", "--json", "--json"]),
        Err(UsageError::RepeatedArgument("--json"))
    ));
    assert!(matches!(
        parse(&["attach"]),
        Err(UsageError::UnknownVisionTarget(_))
    ));
}

#[test]
fn should_parse_add_with_flags_in_any_order() {
    let expected = VisionCommand::Add(VisionAddArguments {
        text_model: "folder/text".to_owned(),
        tower_model: "org/tower".to_owned(),
        output_directory: "folder/output".into(),
        render_json: false,
    });
    assert_eq!(
        parse(&[
            "add",
            "folder/text",
            "--tower",
            "org/tower",
            "--output",
            "folder/output"
        ])
        .unwrap(),
        expected
    );
    assert_eq!(
        parse(&[
            "add",
            "--output",
            "folder/output",
            "--tower",
            "org/tower",
            "folder/text"
        ])
        .unwrap(),
        expected
    );
}

#[test]
fn should_refuse_add_without_every_required_part() {
    assert!(matches!(
        parse(&["add", "--tower", "t", "--output", "o"]),
        Err(UsageError::MissingValue("TEXT_MODEL"))
    ));
    assert!(matches!(
        parse(&["add", "m", "--output", "o"]),
        Err(UsageError::MissingValue("--tower"))
    ));
    assert!(matches!(
        parse(&["add", "m", "--tower", "t"]),
        Err(UsageError::MissingValue("--output"))
    ));
    assert!(matches!(
        parse(&["add", "m", "--tower"]),
        Err(UsageError::MissingValue("--tower"))
    ));
    assert!(matches!(
        parse(&["add", "m", "--tower", "--output", "o"]),
        Err(UsageError::MissingValue("--tower"))
    ));
}

#[test]
fn should_refuse_repeated_flags_and_a_second_text_model_for_add() {
    assert!(matches!(
        parse(&["add", "m", "--tower", "a", "--tower", "b", "--output", "o"]),
        Err(UsageError::RepeatedArgument("--tower"))
    ));
    assert!(matches!(
        parse(&["add", "m", "n", "--tower", "t", "--output", "o"]),
        Err(UsageError::UnknownArgument(_))
    ));
}
