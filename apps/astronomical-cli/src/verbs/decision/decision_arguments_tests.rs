//! Unit tests for `astronomical decision` arguments.

use super::*;

fn arguments(words: &[&str]) -> Vec<OsString> {
    let mut argument_list = Vec::with_capacity(words.len());
    for word in words {
        argument_list.push(OsString::from(word));
    }
    argument_list
}

#[test]
fn should_parse_a_calibration_with_a_model_and_an_output() {
    let parsed = parse_decision_command(&arguments(&[
        "calibrate",
        "verona.jsonl",
        "--model",
        "laya-MLX-8bit",
        "--output",
        "laya-verona",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        DecisionCommand::Calibrate(DecisionToolArguments {
            examples_file_path: PathBuf::from("verona.jsonl"),
            model_id: Some("laya-MLX-8bit".to_owned()),
            output_directory: PathBuf::from("laya-verona"),
            training_scope: TrainingScope::FinalLayers,
        })
    );
}

#[test]
fn should_leave_the_model_to_the_default_when_none_is_named() {
    let parsed = parse_decision_command(&arguments(&[
        "calibrate",
        "--output",
        "out",
        "verona.jsonl",
    ]))
    .unwrap();
    let DecisionCommand::Calibrate(calibrate_arguments) = parsed else {
        panic!("calibrate parsed as another sub-verb");
    };
    assert_eq!(calibrate_arguments.model_id, None);
    assert_eq!(
        calibrate_arguments.examples_file_path,
        PathBuf::from("verona.jsonl")
    );
}

#[test]
fn should_require_the_examples_file_and_the_output_folder() {
    assert_eq!(
        parse_decision_command(&arguments(&["calibrate", "--output", "out"])),
        Err(UsageError::DecisionCalibrateExamplesRequired)
    );
    assert_eq!(
        parse_decision_command(&arguments(&["calibrate", "verona.jsonl"])),
        Err(UsageError::MissingValue("--output"))
    );
    assert_eq!(
        parse_decision_command(&arguments(&["calibrate", "verona.jsonl", "--output"])),
        Err(UsageError::MissingValue("--output"))
    );
}

#[test]
fn should_refuse_a_repeated_flag_a_second_examples_file_and_an_unknown_flag() {
    assert_eq!(
        parse_decision_command(&arguments(&[
            "calibrate",
            "a.jsonl",
            "--output",
            "x",
            "--output",
            "y",
        ])),
        Err(UsageError::RepeatedArgument("--output"))
    );
    assert_eq!(
        parse_decision_command(&arguments(&[
            "calibrate",
            "a.jsonl",
            "b.jsonl",
            "--output",
            "x"
        ])),
        Err(UsageError::UnknownArgument("b.jsonl".to_owned()))
    );
    assert_eq!(
        parse_decision_command(&arguments(&["calibrate", "a.jsonl", "--epochs", "3"])),
        Err(UsageError::UnknownArgument("--epochs".to_owned()))
    );
}

#[test]
fn should_refuse_a_missing_or_unknown_sub_verb() {
    assert_eq!(
        parse_decision_command(&[]),
        Err(UsageError::UnknownDecisionTarget(String::new()))
    );
    assert_eq!(
        parse_decision_command(&arguments(&["fine-tune"])),
        Err(UsageError::UnknownDecisionTarget("fine-tune".to_owned()))
    );
}

#[test]
fn should_parse_a_training_with_the_same_arguments_as_a_calibration() {
    let parsed = parse_decision_command(&arguments(&[
        "train",
        "verona.jsonl",
        "--output",
        "laya-trained",
        "--model",
        "laya-MLX-8bit",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        DecisionCommand::Train(DecisionToolArguments {
            examples_file_path: PathBuf::from("verona.jsonl"),
            model_id: Some("laya-MLX-8bit".to_owned()),
            output_directory: PathBuf::from("laya-trained"),
            training_scope: TrainingScope::FinalLayers,
        })
    );
    assert_eq!(
        parse_decision_command(&arguments(&["train", "verona.jsonl"])),
        Err(UsageError::MissingValue("--output"))
    );
}

#[test]
fn should_train_the_whole_model_when_asked() {
    let parsed = parse_decision_command(&arguments(&[
        "train",
        "verona.jsonl",
        "--whole-model",
        "--output",
        "laya-whole",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        DecisionCommand::Train(DecisionToolArguments {
            examples_file_path: PathBuf::from("verona.jsonl"),
            model_id: None,
            output_directory: PathBuf::from("laya-whole"),
            training_scope: TrainingScope::WholeModel,
        })
    );
}

#[test]
fn should_refuse_whole_model_for_a_calibration_and_when_repeated() {
    assert_eq!(
        parse_decision_command(&arguments(&[
            "calibrate",
            "verona.jsonl",
            "--whole-model",
            "--output",
            "laya-whole",
        ])),
        Err(UsageError::UnknownArgument("--whole-model".to_owned()))
    );
    assert_eq!(
        parse_decision_command(&arguments(&[
            "train",
            "verona.jsonl",
            "--whole-model",
            "--whole-model",
            "--output",
            "laya-whole",
        ])),
        Err(UsageError::RepeatedArgument("--whole-model"))
    );
}
