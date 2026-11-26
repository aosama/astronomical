//! Contracts for the steps `validate_required_file` is built from: the
//! canonical model directory, the symlink-aware inspection, the no-follow
//! open, and the pinned-size check.

use std::fs;
use std::os::unix::fs::symlink;

use super::{
    canonicalize_model_directory, ensure_required_file_size_matches, inspect_required_file_path,
    open_regular_required_file,
};
use crate::{ArtifactValidationError, RequiredFileProfile};

const REQUIRED_FILE_NAME: &str = "config.json";

fn profile_pinning_size(size_bytes: u64) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: REQUIRED_FILE_NAME.to_owned(),
        size_bytes,
    }
}

#[test]
fn should_resolve_a_symlinked_model_directory_to_its_real_location() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let real_directory = scratch_directory.path().join("snapshot");
    fs::create_dir(&real_directory).expect("the real folder should be created");
    let linked_directory = scratch_directory.path().join("install");
    symlink(&real_directory, &linked_directory).expect("the link should be created");

    assert_eq!(
        canonicalize_model_directory(&linked_directory),
        fs::canonicalize(&real_directory).expect("the real folder should resolve")
    );
}

#[test]
fn should_keep_the_original_path_when_the_model_directory_does_not_exist() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let missing_directory = scratch_directory.path().join("never-created");

    assert_eq!(
        canonicalize_model_directory(&missing_directory),
        missing_directory
    );
}

#[test]
fn should_report_a_symlink_as_a_symlink_without_following_it() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let target_path = scratch_directory.path().join("blob");
    fs::write(&target_path, b"{}").expect("the target should be written");
    let link_path = scratch_directory.path().join(REQUIRED_FILE_NAME);
    symlink(&target_path, &link_path).expect("the link should be created");

    let path_metadata = inspect_required_file_path(&link_path, REQUIRED_FILE_NAME)
        .expect("the link itself should be inspectable");

    assert!(path_metadata.file_type().is_symlink());
}

#[test]
fn should_fail_inspection_of_a_missing_required_file_with_its_name() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let missing_path = scratch_directory.path().join(REQUIRED_FILE_NAME);

    let inspection_failure = inspect_required_file_path(&missing_path, REQUIRED_FILE_NAME)
        .expect_err("a missing file should not be inspectable");

    assert!(matches!(
        inspection_failure,
        ArtifactValidationError::InspectRequiredFile { file_name, source }
            if file_name == REQUIRED_FILE_NAME && source.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn should_open_a_regular_file_and_return_its_metadata() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let file_path = scratch_directory.path().join(REQUIRED_FILE_NAME);
    fs::write(&file_path, b"{\"a\":1}").expect("the file should be written");
    let path_metadata = fs::symlink_metadata(&file_path).expect("the file should be inspectable");

    let (_file, file_metadata) =
        open_regular_required_file(&file_path, &path_metadata, REQUIRED_FILE_NAME)
            .expect("a regular file should open");

    assert_eq!(file_metadata.len(), 7);
}

#[test]
fn should_reject_a_directory_in_place_of_a_required_file() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let directory_path = scratch_directory.path().join(REQUIRED_FILE_NAME);
    fs::create_dir(&directory_path).expect("the folder should be created");
    let path_metadata =
        fs::symlink_metadata(&directory_path).expect("the folder should be inspectable");

    let open_failure =
        open_regular_required_file(&directory_path, &path_metadata, REQUIRED_FILE_NAME)
            .expect_err("a folder is not a regular file");

    assert!(matches!(
        open_failure,
        ArtifactValidationError::RequiredFileIsNotRegular { file_name }
            if file_name == REQUIRED_FILE_NAME
    ));
}

#[test]
fn should_refuse_to_follow_a_symlink_at_open_time() {
    let scratch_directory = tempfile::tempdir().expect("a scratch folder should exist");
    let target_path = scratch_directory.path().join("blob");
    fs::write(&target_path, b"{}").expect("the target should be written");
    let link_path = scratch_directory.path().join(REQUIRED_FILE_NAME);
    symlink(&target_path, &link_path).expect("the link should be created");
    let path_metadata = fs::symlink_metadata(&link_path).expect("the link should be inspectable");

    let open_failure = open_regular_required_file(&link_path, &path_metadata, REQUIRED_FILE_NAME)
        .expect_err("the no-follow open should reject a symlink");

    assert!(matches!(
        open_failure,
        ArtifactValidationError::InspectRequiredFile { file_name, .. }
            if file_name == REQUIRED_FILE_NAME
    ));
}

#[test]
fn should_accept_a_size_equal_to_the_pinned_size() {
    assert!(ensure_required_file_size_matches(&profile_pinning_size(2048), 2048).is_ok());
}

#[test]
fn should_accept_any_size_when_the_profile_pins_none() {
    assert!(ensure_required_file_size_matches(&profile_pinning_size(0), 0).is_ok());
    assert!(ensure_required_file_size_matches(&profile_pinning_size(0), 123_456).is_ok());
}

#[test]
fn should_reject_a_size_that_differs_from_the_pinned_size() {
    let size_failure = ensure_required_file_size_matches(&profile_pinning_size(2048), 2047)
        .expect_err("one byte short should not match the pinned size");

    assert!(matches!(
        size_failure,
        ArtifactValidationError::RequiredFileSizeMismatch {
            expected_size_bytes: 2048,
            actual_size_bytes: 2047,
            ..
        }
    ));
}
