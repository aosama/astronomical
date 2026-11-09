use std::path::PathBuf;

use super::{MacTrashCommand, ModelFolderTrash, ModelFolderTrashError};

fn model_folder_fixture() -> (tempfile::TempDir, PathBuf) {
    let fixture_root = tempfile::tempdir().expect("fixture root should be created");
    let model_folder = fixture_root.path().join("fixture-model");
    std::fs::create_dir(&model_folder).expect("fixture model folder should be created");
    (fixture_root, model_folder)
}

#[test]
fn should_move_the_folder_when_the_command_succeeds_and_the_folder_is_gone() {
    let (_fixture_root, model_folder) = model_folder_fixture();
    // `rmdir` stands in for `trash`: it succeeds and the empty folder is gone.
    let trash_command = MacTrashCommand::at_path(PathBuf::from("/bin/rmdir"));

    assert!(trash_command.is_available());
    assert_eq!(trash_command.move_to_trash(&model_folder), Ok(()));
    assert!(!model_folder.exists());
}

#[test]
fn should_fail_when_the_command_succeeds_but_the_folder_is_still_there() {
    let (_fixture_root, model_folder) = model_folder_fixture();
    let trash_command = MacTrashCommand::at_path(PathBuf::from("/usr/bin/true"));

    assert_eq!(
        trash_command.move_to_trash(&model_folder),
        Err(ModelFolderTrashError::Failed)
    );
    assert!(model_folder.is_dir());
}

#[test]
fn should_fail_when_the_command_refuses() {
    let (_fixture_root, model_folder) = model_folder_fixture();
    let trash_command = MacTrashCommand::at_path(PathBuf::from("/usr/bin/false"));

    assert_eq!(
        trash_command.move_to_trash(&model_folder),
        Err(ModelFolderTrashError::Failed)
    );
}

#[test]
fn should_report_unavailable_where_this_macos_has_no_trash_command() {
    let (_fixture_root, model_folder) = model_folder_fixture();
    let trash_command = MacTrashCommand::at_path(PathBuf::from("/nonexistent/trash-command"));

    assert!(!trash_command.is_available());
    assert_eq!(
        trash_command.move_to_trash(&model_folder),
        Err(ModelFolderTrashError::Unavailable)
    );
    assert!(model_folder.is_dir());
}
