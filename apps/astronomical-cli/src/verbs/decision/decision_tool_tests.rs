//! Unit tests for the steps both decision verbs share.

use std::path::Path;

use astronomical_config::ModelFamily;

use super::*;

fn lookup_with_root(models_directory: &Path) -> ModelFolderLookup {
    ModelFolderLookup {
        model_directory: models_directory.join("laya-MLX-8bit"),
        model_family: ModelFamily::Laya,
        scanned_model_roots: vec![models_directory.to_path_buf()],
        installed_model_ids: vec!["laya-MLX-8bit".to_owned()],
        nested_model_directories: Vec::new(),
    }
}

#[test]
fn should_refuse_an_output_named_like_an_installed_model() {
    let models_directory = tempfile::tempdir().unwrap();
    let lookup = lookup_with_root(models_directory.path());
    let taken_name = models_directory
        .path()
        .join("elsewhere")
        .join("laya-MLX-8bit");
    std::fs::create_dir(models_directory.path().join("elsewhere")).unwrap();
    std::fs::create_dir(models_directory.path().join("laya-MLX-8bit")).unwrap();
    assert!(matches!(
        check_output_is_new(&lookup, &taken_name),
        Err(DecisionToolError::OutputNameTaken { .. })
    ));
    assert!(check_output_is_new(&lookup, &models_directory.path().join("laya-verona")).is_ok());
}

#[test]
fn should_report_a_missing_examples_file_as_the_invocation_s_fault() {
    let read_outcome = read_labelled_examples(Path::new("no-such-verona.jsonl"));
    let Err(read_error) = read_outcome else {
        panic!("a missing file was read");
    };
    assert!(matches!(
        read_error,
        DecisionToolError::ExamplesUnreadable { .. }
    ));
    assert!(read_error.is_usage_error());
}

#[test]
fn should_not_count_a_change_that_did_not_help_as_a_usage_error() {
    let nothing_improved = DecisionToolError::NothingImproved {
        model_id: "laya-MLX-8bit".to_owned(),
    };
    assert!(!nothing_improved.is_usage_error());
    assert!(nothing_improved.to_string().contains("nothing was written"));
    let training_refused = DecisionToolError::TrainingRefused {
        model_id: "laya-MLX-8bit".to_owned(),
    };
    assert!(!training_refused.is_usage_error());
}

#[test]
fn should_say_how_much_more_data_training_needs_and_blame_the_invocation() {
    let too_few = DecisionToolError::TooLittleData(TooLittleData::TooFewQuestions {
        have: 70,
        need: 200,
    });
    assert!(too_few.is_usage_error());
    let message = too_few.to_string();
    assert!(
        message.contains("at least 200 labelled questions"),
        "{message}"
    );
    assert!(message.contains("has 70"), "{message}");
    assert!(message.contains("decision calibrate"), "{message}");
}

#[test]
fn should_take_the_output_folder_name_as_the_model_id() {
    assert_eq!(
        output_model_id(Path::new("models/calibrated/laya-verona")),
        "laya-verona"
    );
    assert_eq!(output_model_id(Path::new("/")), "");
}
