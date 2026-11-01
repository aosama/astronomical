use std::fs;

use super::*;

#[test]
fn should_prefer_the_canonical_metadata_when_both_locations_exist() {
    assert_eq!(
        choose_optiq_metadata_relative_path(true, true),
        Some(OPTIQ_METADATA_RELATIVE_PATH)
    );
}

#[test]
fn should_fall_back_to_the_legacy_metadata_when_only_it_exists() {
    assert_eq!(
        choose_optiq_metadata_relative_path(false, true),
        Some(LEGACY_OPTIQ_METADATA_RELATIVE_PATH)
    );
}

#[test]
fn should_report_no_metadata_when_neither_location_exists() {
    assert_eq!(choose_optiq_metadata_relative_path(false, false), None);
}

#[test]
fn should_use_the_canonical_metadata_when_only_it_exists() {
    assert_eq!(
        choose_optiq_metadata_relative_path(true, false),
        Some(OPTIQ_METADATA_RELATIVE_PATH)
    );
}

#[test]
fn should_resolve_the_metadata_location_from_the_files_on_disk() {
    let model_directory = tempfile::tempdir().expect("temporary model directory");
    assert_eq!(
        resolve_optiq_metadata_relative_path(model_directory.path()),
        None
    );

    fs::write(
        model_directory
            .path()
            .join(LEGACY_OPTIQ_METADATA_RELATIVE_PATH),
        b"{}",
    )
    .expect("legacy metadata written");
    assert_eq!(
        resolve_optiq_metadata_relative_path(model_directory.path()),
        Some(LEGACY_OPTIQ_METADATA_RELATIVE_PATH)
    );

    fs::create_dir(model_directory.path().join("optiq")).expect("optiq folder created");
    fs::write(
        model_directory.path().join(OPTIQ_METADATA_RELATIVE_PATH),
        b"{}",
    )
    .expect("canonical metadata written");
    assert_eq!(
        resolve_optiq_metadata_relative_path(model_directory.path()),
        Some(OPTIQ_METADATA_RELATIVE_PATH)
    );
}
