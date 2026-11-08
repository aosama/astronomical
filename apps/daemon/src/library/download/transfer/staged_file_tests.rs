//! Beside-source coverage for creating the staged copies of zero-byte files (issue #1896).

use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    sync::atomic::AtomicBool,
};

use tempfile::TempDir;

use super::{EmptyFileCreation, create_empty_staged_files};
use crate::DownloadJob;
use crate::library::DownloadJobStoreError;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EMPTY_FILE_PATH: &str = "tools/__init__.py";
const WEIGHTS_PATH: &str = "weights/model.safetensors";

/// A paused job holding one nested empty file and one file that still needs bytes.
fn job_with_an_empty_file_and_weights() -> DownloadJob {
    let job_document = format!(
        "{{\
            \"huggingface_id\":\"astronomical-test/example-qwen\",\
            \"revision\":\"{REVISION}\",\
            \"state\":\"paused\",\
            \"bytes_completed\":0,\
            \"bytes_total\":12,\
            \"current_file_relative_path\":null,\
            \"files\":[\
                {{\"relative_path\":\"{EMPTY_FILE_PATH}\",\"expected_bytes\":0,\
                  \"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\
                  \"bytes_on_disk\":0}},\
                {{\"relative_path\":\"{WEIGHTS_PATH}\",\"expected_bytes\":12,\
                  \"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\
                  \"bytes_on_disk\":0}}\
            ],\
            \"error_code\":null,\
            \"updated_at\":100\
        }}"
    );
    DownloadJob::parse_json(&job_document).expect("fixture job should be valid")
}

fn staged_empty_file_path(models_directory: &std::path::Path, job: &DownloadJob) -> PathBuf {
    job.staging_directory(models_directory)
        .join(EMPTY_FILE_PATH)
}

#[test]
fn should_create_a_nested_empty_file_privately_and_leave_non_empty_files_alone() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();

    let creation = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(false))
        .expect("the empty file should be created");

    assert_eq!(creation, EmptyFileCreation::Created);
    let staged_path = staged_empty_file_path(test_directory.path(), &job);
    let staged_metadata = fs::metadata(&staged_path).expect("the empty file should exist");
    assert_eq!(staged_metadata.len(), 0);
    assert_eq!(staged_metadata.permissions().mode() & 0o777, 0o600);
    assert!(
        !job.staging_directory(test_directory.path())
            .join(WEIGHTS_PATH)
            .exists(),
        "a file with bytes left is the transfer window's job, not this function's"
    );
}

#[test]
fn should_be_idempotent_when_the_empty_file_already_exists() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();
    let not_paused = AtomicBool::new(false);
    create_empty_staged_files(test_directory.path(), &job, &not_paused)
        .expect("the first creation should succeed");

    let second_creation = create_empty_staged_files(test_directory.path(), &job, &not_paused)
        .expect("a second creation after an interrupted attempt should succeed");

    assert_eq!(second_creation, EmptyFileCreation::Created);
    assert_eq!(
        fs::metadata(staged_empty_file_path(test_directory.path(), &job))
            .expect("the empty file should still exist")
            .len(),
        0
    );
}

#[test]
fn should_create_nothing_once_a_pause_or_cancel_was_requested() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();

    let creation = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(true))
        .expect("an interrupted creation is not an error");

    assert_eq!(creation, EmptyFileCreation::Interrupted);
    assert!(
        !job.staging_directory(test_directory.path()).exists(),
        "a cancel that removed the staging folder must not see it re-created"
    );
}

#[test]
fn should_refuse_a_symlink_in_place_of_the_empty_file() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();
    let staged_path = staged_empty_file_path(test_directory.path(), &job);
    fs::create_dir_all(staged_path.parent().expect("the staged file has a parent"))
        .expect("the staged folder should be created");
    let outside_file = test_directory.path().join("outside.txt");
    fs::write(&outside_file, b"").expect("the outside file should be written");
    symlink(&outside_file, &staged_path).expect("the symlink should be created");

    let failure = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(false))
        .expect_err("a symlink must be refused");

    assert!(matches!(
        failure,
        DownloadJobStoreError::UnsafeFilesystemObject { .. }
    ));
}

#[test]
fn should_refuse_a_directory_in_place_of_the_empty_file() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();
    fs::create_dir_all(staged_empty_file_path(test_directory.path(), &job))
        .expect("the directory should be created");

    let failure = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(false))
        .expect_err("a directory must be refused");

    assert!(matches!(
        failure,
        DownloadJobStoreError::UnsafeFilesystemObject { .. }
    ));
}

#[test]
fn should_refuse_a_symlinked_parent_folder() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();
    let staging_directory = job.staging_directory(test_directory.path());
    fs::create_dir_all(&staging_directory).expect("the staging folder should be created");
    let outside_directory = test_directory.path().join("outside");
    fs::create_dir_all(&outside_directory).expect("the outside folder should be created");
    symlink(&outside_directory, staging_directory.join("tools"))
        .expect("the symlink should be created");

    let failure = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(false))
        .expect_err("a symlinked parent must be refused");

    assert!(matches!(
        failure,
        DownloadJobStoreError::UnsafeFilesystemObject { .. }
    ));
    assert!(!outside_directory.join("__init__.py").exists());
}

#[test]
fn should_refuse_a_staged_empty_file_that_already_holds_bytes() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job = job_with_an_empty_file_and_weights();
    let staged_path = staged_empty_file_path(test_directory.path(), &job);
    fs::create_dir_all(staged_path.parent().expect("the staged file has a parent"))
        .expect("the staged folder should be created");
    fs::write(&staged_path, b"stray").expect("the stray bytes should be written");

    let failure = create_empty_staged_files(test_directory.path(), &job, &AtomicBool::new(false))
        .expect_err("a file that should be empty but is not must be refused");

    assert!(matches!(
        failure,
        DownloadJobStoreError::StagedFileProgressMismatch { .. }
    ));
}
