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
        scope: TrainingScope::FinalLayers,
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
fn should_point_a_too_small_whole_model_training_at_the_final_layers() {
    let too_few = DecisionToolError::TooLittleData(TooLittleData::TooFewQuestions {
        have: 400,
        need: 1_000,
        scope: TrainingScope::WholeModel,
    });
    let message = too_few.to_string();
    assert!(
        message.contains("whole model needs at least 1000"),
        "{message}"
    );
    assert!(message.contains("leave out --whole-model"), "{message}");
    assert!(message.contains("works with 200"), "{message}");
}

#[test]
fn should_take_the_output_folder_name_as_the_model_id() {
    assert_eq!(
        output_model_id(Path::new("models/calibrated/laya-verona")),
        "laya-verona"
    );
    assert_eq!(output_model_id(Path::new("/")), "");
}

/// A canonical, existing models folder with one installed model in it.
fn models_root_with_one_model() -> (tempfile::TempDir, PathBuf) {
    let root_directory = tempfile::tempdir().unwrap();
    let canonical_root = std::fs::canonicalize(root_directory.path()).unwrap();
    let installed_model = canonical_root.join("laya-MLX-8bit");
    std::fs::create_dir_all(&installed_model).unwrap();
    std::fs::write(installed_model.join("config.json"), "{}").unwrap();
    (root_directory, canonical_root)
}

#[test]
fn should_accept_an_output_folder_astronomical_looks_in_even_before_its_parents_exist() {
    let (_root_directory, models_root) = models_root_with_one_model();

    assert!(
        check_output_is_discoverable(
            std::slice::from_ref(&models_root),
            &models_root.join("calibrated/laya-moods")
        )
        .is_ok()
    );
}

#[test]
fn should_refuse_an_output_folder_outside_every_models_folder_and_name_them() {
    let (_root_directory, models_root) = models_root_with_one_model();
    let elsewhere = tempfile::tempdir().unwrap();

    let refusal = check_output_is_discoverable(
        std::slice::from_ref(&models_root),
        &elsewhere.path().join("laya-moods"),
    )
    .unwrap_err();

    assert!(refusal.is_usage_error());
    let message = refusal.to_string();
    assert!(message.contains("would not find a model"), "{message}");
    assert!(
        message.contains(&models_root.display().to_string()),
        "{message}"
    );
    assert!(message.contains("calibrated/laya-moods"), "{message}");
}

#[test]
fn should_refuse_an_output_folder_inside_an_installed_model_or_too_deep() {
    let (_root_directory, models_root) = models_root_with_one_model();
    let roots = std::slice::from_ref(&models_root);

    assert!(check_output_is_discoverable(roots, &models_root.join("laya-MLX-8bit/moods")).is_err());
    assert!(check_output_is_discoverable(roots, &models_root.join("a/b/c/d/moods")).is_err());
}

#[test]
fn should_skip_a_configured_models_folder_that_does_not_exist() {
    let (_root_directory, models_root) = models_root_with_one_model();
    let missing_root = models_root.join("not-yet-created-root");

    assert!(
        check_output_is_discoverable(
            &[missing_root, models_root.clone()],
            &models_root.join("laya-moods")
        )
        .is_ok()
    );
    assert!(check_output_is_discoverable(&[], &models_root.join("laya-moods")).is_err());
}

#[test]
fn should_accept_an_output_named_through_a_link_to_the_models_folder() {
    let (_root_directory, models_root) = models_root_with_one_model();
    let link_directory = tempfile::tempdir().unwrap();
    let linked_root = link_directory.path().join("models-link");
    std::os::unix::fs::symlink(&models_root, &linked_root).unwrap();

    // The configured folder is the link; the output is named through it.
    assert!(
        check_output_is_discoverable(
            std::slice::from_ref(&linked_root),
            &linked_root.join("calibrated/laya-moods")
        )
        .is_ok()
    );
    // The configured folder is the link; the output is named by the real path.
    assert!(
        check_output_is_discoverable(
            std::slice::from_ref(&linked_root),
            &models_root.join("laya-moods")
        )
        .is_ok()
    );
}
