use std::io::Read;
use std::os::unix::fs::symlink;

use astronomical_model_serving::{
    ArtifactValidationError, RequiredFileProfile, validate_required_file_for_tests,
};

#[test]
fn should_accept_a_hugging_face_snapshot_symlink_to_its_own_blob_directory() {
    let temporary_directory = tempfile::tempdir().expect("the test should create a temporary root");
    let model_cache_directory = temporary_directory.path().join("models--example--model");
    let blob_directory = model_cache_directory.join("blobs");
    let snapshot_directory = model_cache_directory.join("snapshots/commit-hash");
    std::fs::create_dir_all(&blob_directory).expect("the test should create the blob directory");
    std::fs::create_dir_all(&snapshot_directory)
        .expect("the test should create the snapshot directory");
    let expected_config_bytes = br#"{"model_type":"qwen3_5_moe"}"#;
    std::fs::write(blob_directory.join("config-blob"), expected_config_bytes)
        .expect("the test should write the immutable blob");
    symlink(
        "../../blobs/config-blob",
        snapshot_directory.join("config.json"),
    )
    .expect("the test should create the Hugging Face snapshot symlink");

    let validated_weights_file = validate_required_file_for_tests(
        &snapshot_directory,
        &RequiredFileProfile {
            file_name: "config.json".to_owned(),
            size_bytes: expected_config_bytes.len() as u64,
        },
    )
    .expect("a snapshot symlink confined to its own blob directory should validate");
    let mut validated_file = validated_weights_file.into_file();
    let mut actual_config_bytes = Vec::new();
    validated_file
        .read_to_end(&mut actual_config_bytes)
        .expect("the retained descriptor should read the validated blob");
    assert_eq!(actual_config_bytes, expected_config_bytes);
}

#[test]
fn should_reject_a_hugging_face_snapshot_symlink_that_escapes_its_blob_directory() {
    let temporary_directory = tempfile::tempdir().expect("the test should create a temporary root");
    let model_cache_directory = temporary_directory.path().join("models--example--model");
    let blob_directory = model_cache_directory.join("blobs");
    let snapshot_directory = model_cache_directory.join("snapshots/commit-hash");
    std::fs::create_dir_all(&blob_directory).expect("the test should create the blob directory");
    std::fs::create_dir_all(&snapshot_directory)
        .expect("the test should create the snapshot directory");
    std::fs::write(
        temporary_directory.path().join("outside-config.json"),
        b"outside",
    )
    .expect("the test should write the out-of-bound target");
    symlink(
        "../../../outside-config.json",
        snapshot_directory.join("config.json"),
    )
    .expect("the test should create the escaping snapshot symlink");

    let validation_error = validate_required_file_for_tests(
        &snapshot_directory,
        &RequiredFileProfile {
            file_name: "config.json".to_owned(),
            size_bytes: 0,
        },
    )
    .expect_err("a snapshot symlink outside its own blob directory must fail closed");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::HuggingFaceSnapshotSymlinkEscapesBlobDirectory { file_name, .. }
            if file_name == "config.json"
    ));
}

#[test]
fn should_continue_rejecting_symlinks_in_regular_model_directories() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    std::fs::write(
        model_directory.path().join("config-contents.json"),
        b"contents",
    )
    .expect("the test should write the target file");
    symlink(
        "config-contents.json",
        model_directory.path().join("config.json"),
    )
    .expect("the test should create a regular artifact symlink");

    let validation_error = validate_required_file_for_tests(
        model_directory.path(),
        &RequiredFileProfile {
            file_name: "config.json".to_owned(),
            size_bytes: 0,
        },
    )
    .expect_err("regular model directories must continue rejecting symlinks");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::RequiredFileIsSymlink { file_name }
            if file_name == "config.json"
    ));
}

#[test]
fn should_reject_a_required_file_name_with_parent_directory_components() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    std::fs::write(model_directory.path().join("outside.json"), b"contents")
        .expect("the test should write a file");

    let validation_error = validate_required_file_for_tests(
        model_directory.path(),
        &RequiredFileProfile {
            file_name: "../outside.json".to_owned(),
            size_bytes: 0,
        },
    )
    .expect_err("required file names must not escape the model directory");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::InvalidProfileFileName { file_name }
            if file_name == "../outside.json"
    ));
}
