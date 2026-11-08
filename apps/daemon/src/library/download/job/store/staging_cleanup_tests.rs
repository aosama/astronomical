//! Startup removal of empty staging folders left by earlier versions (#1929).

use std::{fs, os::unix::fs::symlink, path::PathBuf};

use tempfile::TempDir;

use crate::{DownloadJobStore, DownloadJobStoreError};

struct StagingFixture {
    _test_directory: TempDir,
    models_directory: PathBuf,
    incomplete_directory: PathBuf,
}

fn staging_fixture() -> StagingFixture {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let incomplete_directory = models_directory.join(".incomplete");
    fs::create_dir_all(&incomplete_directory).expect("the staging folder should be created");
    StagingFixture {
        _test_directory: test_directory,
        models_directory,
        incomplete_directory,
    }
}

fn remove_empty_staging_folders(staging: &StagingFixture) {
    DownloadJobStore::new(staging.models_directory.clone())
        .remove_empty_staging_folders()
        .expect("the cleanup should succeed");
}

#[test]
fn should_remove_empty_organization_folders_and_the_empty_staging_folder() {
    let staging = staging_fixture();
    fs::create_dir(staging.incomplete_directory.join("araysuter")).expect("folder");
    fs::create_dir(staging.incomplete_directory.join("mlx-community")).expect("folder");

    remove_empty_staging_folders(&staging);

    assert!(!staging.incomplete_directory.exists());
    assert!(
        staging.models_directory.is_dir(),
        "the models folder itself stays"
    );
}

#[test]
fn should_treat_folders_holding_only_finder_metadata_as_empty() {
    let staging = staging_fixture();
    let organization_directory = staging.incomplete_directory.join("mlx-community");
    fs::create_dir(&organization_directory).expect("folder");
    fs::write(organization_directory.join(".DS_Store"), b"finder").expect("finder file");
    fs::write(staging.incomplete_directory.join(".DS_Store"), b"finder").expect("finder file");

    remove_empty_staging_folders(&staging);

    assert!(!staging.incomplete_directory.exists());
}

#[test]
fn should_keep_the_staged_bytes_of_a_paused_or_failed_download() {
    let staging = staging_fixture();
    let staged_model_directory = staging
        .incomplete_directory
        .join("mlx-community/Qwen3.5-2B-4bit");
    fs::create_dir_all(&staged_model_directory).expect("folder");
    fs::write(staged_model_directory.join("config.json"), b"{}").expect("staged file");
    fs::create_dir(staging.incomplete_directory.join("araysuter")).expect("folder");

    remove_empty_staging_folders(&staging);

    assert!(staged_model_directory.join("config.json").is_file());
    assert!(!staging.incomplete_directory.join("araysuter").exists());
    assert!(
        staging.incomplete_directory.is_dir(),
        "the staging folder stays while in use"
    );
}

#[test]
fn should_leave_symlinks_and_stray_files_untouched() {
    let staging = staging_fixture();
    let outside_directory = staging.models_directory.join("outside");
    fs::create_dir(&outside_directory).expect("folder");
    symlink(
        &outside_directory,
        staging.incomplete_directory.join("linked"),
    )
    .expect("symlink");
    fs::write(staging.incomplete_directory.join("notes.txt"), b"keep me").expect("stray file");

    remove_empty_staging_folders(&staging);

    assert!(
        outside_directory.is_dir(),
        "a symlink's target is never removed"
    );
    assert!(staging.incomplete_directory.join("linked").is_symlink());
    assert!(staging.incomplete_directory.join("notes.txt").is_file());
}

#[test]
fn should_do_nothing_when_there_is_no_staging_folder() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");

    DownloadJobStore::new(models_directory.clone())
        .remove_empty_staging_folders()
        .expect("a missing staging folder is not an error");

    assert!(!models_directory.exists(), "the cleanup creates nothing");
}

#[test]
fn should_keep_finder_metadata_beside_a_staged_download() {
    let staging = staging_fixture();
    let organization_directory = staging.incomplete_directory.join("mlx-community");
    let staged_model_directory = organization_directory.join("Qwen3.5-2B-4bit");
    fs::create_dir_all(&staged_model_directory).expect("folder");
    fs::write(staged_model_directory.join("config.json"), b"{}").expect("staged file");
    fs::write(organization_directory.join(".DS_Store"), b"finder").expect("finder file");

    remove_empty_staging_folders(&staging);

    assert!(organization_directory.join(".DS_Store").is_file());
    assert!(staged_model_directory.join("config.json").is_file());
}

#[test]
fn should_keep_an_organization_folder_holding_an_empty_model_folder() {
    let staging = staging_fixture();
    let empty_model_directory = staging
        .incomplete_directory
        .join("mlx-community/Qwen3.5-2B-4bit");
    fs::create_dir_all(&empty_model_directory).expect("folder");

    remove_empty_staging_folders(&staging);

    assert!(
        empty_model_directory.is_dir(),
        "only organization folders are tidied; a model folder belongs to its download job"
    );
}

#[test]
fn should_change_nothing_when_run_again_on_a_tidy_tree() {
    let staging = staging_fixture();
    fs::create_dir(staging.incomplete_directory.join("araysuter")).expect("folder");
    remove_empty_staging_folders(&staging);

    remove_empty_staging_folders(&staging);

    assert!(!staging.incomplete_directory.exists());
    assert!(staging.models_directory.is_dir());
}

#[test]
fn should_leave_a_staging_path_that_is_a_file_alone() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    fs::create_dir_all(&models_directory).expect("folder");
    fs::write(models_directory.join(".incomplete"), b"not a folder").expect("file");

    DownloadJobStore::new(models_directory.clone())
        .remove_empty_staging_folders()
        .expect("a file in the way is not an error");

    assert!(models_directory.join(".incomplete").is_file());
}

#[test]
fn should_refuse_a_symlinked_staging_folder() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let outside_directory = test_directory.path().join("outside");
    fs::create_dir_all(&models_directory).expect("folder");
    fs::create_dir_all(outside_directory.join("empty-organization")).expect("folder");
    symlink(&outside_directory, models_directory.join(".incomplete")).expect("symlink");

    let cleanup_failure = DownloadJobStore::new(models_directory)
        .remove_empty_staging_folders()
        .expect_err("a symlinked staging folder is never followed");

    assert!(matches!(
        cleanup_failure,
        DownloadJobStoreError::UnsafeFilesystemObject { .. }
    ));
    assert!(outside_directory.join("empty-organization").is_dir());
}
