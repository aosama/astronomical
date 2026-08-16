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

#[test]
fn should_read_an_ordinary_json_sidecar_through_its_retained_descriptor() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let sidecar_file_name = "model.safetensors.index.json";
    let sidecar_path = model_directory.path().join(sidecar_file_name);
    let retained_sidecar_bytes = br#"{"weight_map":{"model.weight":"model.safetensors"}}"#;
    std::fs::write(&sidecar_path, retained_sidecar_bytes)
        .expect("the test should write the original JSON sidecar");
    let required_file_profile = RequiredFileProfile {
        file_name: sidecar_file_name.to_owned(),
        size_bytes: retained_sidecar_bytes.len() as u64,
    };
    let validated_required_file =
        validate_required_file_for_tests(model_directory.path(), &required_file_profile)
            .expect("the ordinary JSON sidecar should validate");

    // Replacing the pathname proves that the read stays on the retained inode.
    std::fs::rename(
        &sidecar_path,
        model_directory.path().join("validated-index.json"),
    )
    .expect("the test should preserve the validated inode under another name");
    std::fs::write(&sidecar_path, br#"{"replacement":true}"#)
        .expect("the test should replace the original pathname");

    let actual_sidecar_bytes = validated_required_file
        .read_bounded_bytes_for_tests(retained_sidecar_bytes.len() as u64)
        .expect("the exact bounded read should use the validated descriptor");

    assert_eq!(actual_sidecar_bytes, retained_sidecar_bytes);
}

#[test]
fn should_reject_a_bounded_required_file_read_above_its_explicit_limit() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let sidecar_file_name = "model.safetensors.index.json";
    let sidecar_bytes = br#"{"weight_map":{}}"#;
    std::fs::write(
        model_directory.path().join(sidecar_file_name),
        sidecar_bytes,
    )
    .expect("the test should write the JSON sidecar");
    let validated_required_file = validate_required_file_for_tests(
        model_directory.path(),
        &RequiredFileProfile {
            file_name: sidecar_file_name.to_owned(),
            size_bytes: sidecar_bytes.len() as u64,
        },
    )
    .expect("the ordinary JSON sidecar should validate");

    let validation_error = validated_required_file
        .read_bounded_bytes_for_tests((sidecar_bytes.len() - 1) as u64)
        .expect_err("a sidecar above the caller's explicit limit must fail closed");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::BoundedRequiredFileTooLarge {
            file_name,
            actual_size_bytes,
            maximum_size_bytes,
        } if file_name == sidecar_file_name
            && actual_size_bytes == sidecar_bytes.len() as u64
            && maximum_size_bytes == (sidecar_bytes.len() - 1) as u64
    ));
}

#[test]
fn should_preserve_the_source_when_a_retained_descriptor_becomes_short() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let sidecar_file_name = "model.safetensors.index.json";
    let sidecar_path = model_directory.path().join(sidecar_file_name);
    let sidecar_bytes = br#"{"weight_map":{"model.weight":"model.safetensors"}}"#;
    std::fs::write(&sidecar_path, sidecar_bytes).expect("the test should write the JSON sidecar");
    let validated_required_file = validate_required_file_for_tests(
        model_directory.path(),
        &RequiredFileProfile {
            file_name: sidecar_file_name.to_owned(),
            size_bytes: sidecar_bytes.len() as u64,
        },
    )
    .expect("the ordinary JSON sidecar should validate");
    std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&sidecar_path)
        .expect("the test should truncate the validated inode");

    let validation_error = validated_required_file
        .read_bounded_bytes_for_tests(sidecar_bytes.len() as u64)
        .expect_err("a short retained descriptor must fail with its read source");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::ReadBoundedRequiredFile { file_name, source }
            if file_name == sidecar_file_name
                && source.kind() == std::io::ErrorKind::UnexpectedEof
    ));
}

#[test]
fn should_reject_a_duplicate_required_profile_before_replacing_the_first_file() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let config_bytes = br#"{"model_type":"example"}"#;
    std::fs::write(model_directory.path().join("config.json"), config_bytes)
        .expect("the test should write the required file");
    let duplicate_profiles = [
        RequiredFileProfile {
            file_name: "config.json".to_owned(),
            size_bytes: config_bytes.len() as u64,
        },
        RequiredFileProfile {
            file_name: "config.json".to_owned(),
            size_bytes: config_bytes.len() as u64 + 1,
        },
    ];

    let validation_error =
        RequiredFileProfile::validate_all_for_tests(model_directory.path(), &duplicate_profiles)
            .expect_err(
                "a repeated profile name must fail instead of replacing its first descriptor",
            );

    assert!(matches!(
        validation_error,
        ArtifactValidationError::DuplicateProfileFileName { file_name }
            if file_name == "config.json"
    ));
}
