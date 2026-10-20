//! Unit tests for the artifact file layout: where OptiQ files are written
//! and the legacy root-level metadata name that published quants still carry.

use super::*;

#[test]
fn should_place_optiq_files_under_the_optiq_subdirectory() {
    let model_directory = Path::new("artifact");

    assert_eq!(
        artifact_path(model_directory, METADATA_FILE_NAME),
        PathBuf::from("artifact/optiq/metadata.json")
    );
    assert_eq!(
        artifact_path(model_directory, SENSITIVITY_FILE_NAME),
        PathBuf::from("artifact/optiq/sensitivity.json")
    );
}

#[test]
fn should_prefer_the_canonical_metadata_location_over_the_legacy_root_name() {
    let model_directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::create_dir_all(model_directory.path().join(OPTIQ_SUBDIRECTORY))
        .expect("the optiq folder should be creatable");
    std::fs::write(model_directory.path().join(LEGACY_METADATA_FILE_NAME), "{}")
        .expect("the legacy file should be writable");
    std::fs::write(
        artifact_path(model_directory.path(), METADATA_FILE_NAME),
        "{}",
    )
    .expect("the canonical file should be writable");

    assert_eq!(
        resolve_metadata_path(model_directory.path()),
        Some(artifact_path(model_directory.path(), METADATA_FILE_NAME))
    );
}

#[test]
fn should_fall_back_to_the_legacy_root_metadata_name() {
    let model_directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(model_directory.path().join(LEGACY_METADATA_FILE_NAME), "{}")
        .expect("the legacy file should be writable");

    assert_eq!(
        resolve_metadata_path(model_directory.path()),
        Some(model_directory.path().join(LEGACY_METADATA_FILE_NAME))
    );
}

#[test]
fn should_resolve_nothing_for_a_directory_without_metadata() {
    let model_directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    assert_eq!(resolve_metadata_path(model_directory.path()), None);
}
