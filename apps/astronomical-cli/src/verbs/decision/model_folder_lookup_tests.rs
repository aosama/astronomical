//! Unit tests for finding a decision model's folder.

use std::path::Path;

use config::AstronomicalRuntimeInstance;

use super::*;

fn instance_in(state_directory: &Path) -> AstronomicalInstancePaths {
    AstronomicalInstancePaths::for_state_directory(
        state_directory.to_path_buf(),
        AstronomicalRuntimeInstance::Development,
    )
}

#[test]
fn should_report_a_model_that_is_not_installed() {
    let state_directory = tempfile::tempdir().unwrap();
    let lookup_outcome = find_model_folder(&instance_in(state_directory.path()), "laya-MLX-8bit");
    assert!(matches!(
        lookup_outcome,
        Err(ModelFolderLookupError::ModelNotFound { model_id }) if model_id == "laya-MLX-8bit"
    ));
}

#[test]
fn should_report_a_configuration_it_cannot_read() {
    let state_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        state_directory.path().join("config.json"),
        b"A plague o' both your houses!",
    )
    .unwrap();
    let lookup_outcome = find_model_folder(&instance_in(state_directory.path()), "laya-MLX-8bit");
    assert!(matches!(
        lookup_outcome,
        Err(ModelFolderLookupError::ConfigUnreadable { .. })
    ));
}

#[test]
fn should_name_only_the_models_strictly_inside_the_model_folder() {
    let model_directory = Path::new("models/cavi-ai/laya-MLX-8bit");
    let discovered_directories = [
        model_directory.to_path_buf(),
        model_directory.join("multilingual"),
        PathBuf::from("models/cavi-ai/laya-MLX-8bit-other"),
        PathBuf::from("models/mlx-community/clef-flash-8bit"),
    ];
    assert_eq!(
        nested_directories(model_directory, &discovered_directories),
        vec![PathBuf::from("multilingual")]
    );
    assert!(nested_directories(model_directory, &[]).is_empty());
}
