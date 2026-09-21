use std::io::Read;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

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

/// Digest the shared store names its object after in the shared-blob fixtures.
const SHARED_BLOB_CONTENT_DIGEST: &str =
    "ab00000000000000000000000000000000000000000000000000000000000000";

/// Digest the shared store does not name its object after; a snapshot tree
/// record carrying it must not authenticate the fixture's shared blob.
const UNRELATED_CONTENT_DIGEST: &str =
    "cd00000000000000000000000000000000000000000000000000000000000000";

const SHARED_BLOB_CONFIG_BYTES: &[u8] = br#"{"model_type":"qwen3_5_moe"}"#;

/// One Hugging Face hub whose model entry reaches an object in the hub-level
/// shared blob store, so each test can vary exactly one verification input.
struct SharedBlobLayout {
    hub_directory: tempfile::TempDir,
    snapshot_directory: PathBuf,
}

impl SharedBlobLayout {
    /// Creates the shared store object, the entry's local blob link, the
    /// snapshot symlink, and the entry's tree metadata directory.
    fn create() -> Self {
        let hub_directory =
            tempfile::tempdir().expect("the test should create a temporary hub root");
        let model_cache_directory = hub_directory.path().join("models--example--model");
        let local_blob_directory = model_cache_directory.join("blobs");
        let snapshot_directory = model_cache_directory.join("snapshots/commit-hash");
        let shared_blob_directory = hub_directory.path().join("blobs/ab");
        std::fs::create_dir_all(&local_blob_directory)
            .expect("the test should create the entry blob directory");
        std::fs::create_dir_all(&shared_blob_directory)
            .expect("the test should create the shared blob directory");
        std::fs::create_dir_all(&snapshot_directory)
            .expect("the test should create the snapshot directory");
        std::fs::create_dir_all(model_cache_directory.join("trees"))
            .expect("the test should create the tree metadata directory");
        std::fs::write(
            shared_blob_directory.join(SHARED_BLOB_CONTENT_DIGEST),
            SHARED_BLOB_CONFIG_BYTES,
        )
        .expect("the test should write the shared immutable blob");
        symlink(
            "../../blobs/ab/ab00000000000000000000000000000000000000000000000000000000000000",
            local_blob_directory.join("local-config-blob"),
        )
        .expect("the test should link the entry blob name to the shared blob");
        symlink(
            "../../blobs/local-config-blob",
            snapshot_directory.join("config.json"),
        )
        .expect("the test should create the Hugging Face snapshot symlink");

        Self {
            hub_directory,
            snapshot_directory,
        }
    }

    /// Writes the snapshot tree record that anchors trust for the shared blob.
    fn write_tree_record(&self, recorded_digest_text: &str, recorded_size_bytes: u64) {
        std::fs::write(
            self.hub_directory
                .path()
                .join("models--example--model/trees/commit-hash.json"),
            serde_json::json!({
                "format_version": 1,
                "files": {
                    "config.json": {
                        "size": recorded_size_bytes,
                        "blob_id": "git-blob-id",
                        "lfs_sha256": recorded_digest_text,
                        "lfs_size": recorded_size_bytes
                    }
                }
            })
            .to_string(),
        )
        .expect("the test should write the snapshot tree metadata");
    }

    fn validate(&self) -> Result<(), ArtifactValidationError> {
        validate_required_file_for_tests(
            &self.snapshot_directory,
            &RequiredFileProfile {
                file_name: "config.json".to_owned(),
                size_bytes: SHARED_BLOB_CONFIG_BYTES.len() as u64,
            },
        )
        .map(drop)
    }
}

#[test]
fn should_accept_a_hugging_face_snapshot_symlink_to_a_verified_shared_blob() {
    let shared_blob_layout = SharedBlobLayout::create();
    shared_blob_layout.write_tree_record(
        SHARED_BLOB_CONTENT_DIGEST,
        SHARED_BLOB_CONFIG_BYTES.len() as u64,
    );

    shared_blob_layout
        .validate()
        .expect("a snapshot symlink to a snapshot-recorded shared blob should validate");
}

#[test]
fn should_reject_a_shared_blob_that_is_not_the_snapshot_recorded_content_address() {
    let shared_blob_layout = SharedBlobLayout::create();
    shared_blob_layout.write_tree_record(
        UNRELATED_CONTENT_DIGEST,
        SHARED_BLOB_CONFIG_BYTES.len() as u64,
    );

    let validation_error = shared_blob_layout
        .validate()
        .expect_err("a shared blob outside its recorded content address must fail closed");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::HuggingFaceSharedBlobIdentityMismatch {
            file_name,
            recorded_digest_text,
        } if file_name == "config.json" && recorded_digest_text == UNRELATED_CONTENT_DIGEST
    ));
}

#[test]
fn should_reject_a_shared_blob_whose_size_disagrees_with_its_snapshot_tree_record() {
    let shared_blob_layout = SharedBlobLayout::create();
    shared_blob_layout.write_tree_record(
        SHARED_BLOB_CONTENT_DIGEST,
        SHARED_BLOB_CONFIG_BYTES.len() as u64 + 1,
    );

    let validation_error = shared_blob_layout
        .validate()
        .expect_err("a shared blob whose size disagrees with its record must fail closed");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::HuggingFaceSharedBlobSizeMismatch {
            file_name,
            recorded_size_bytes,
            actual_size_bytes,
        } if file_name == "config.json"
            && recorded_size_bytes == SHARED_BLOB_CONFIG_BYTES.len() as u64 + 1
            && actual_size_bytes == SHARED_BLOB_CONFIG_BYTES.len() as u64
    ));
}

#[test]
fn should_reject_a_shared_blob_without_a_snapshot_tree_record() {
    let shared_blob_layout = SharedBlobLayout::create();

    let validation_error = shared_blob_layout
        .validate()
        .expect_err("a shared blob no tree record describes must fail closed");

    assert!(matches!(
        validation_error,
        ArtifactValidationError::HuggingFaceSharedBlobMetadataUnavailable { file_name, .. }
            if file_name == "config.json"
    ));
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
