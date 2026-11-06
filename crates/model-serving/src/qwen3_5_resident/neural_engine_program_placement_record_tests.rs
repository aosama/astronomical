use std::fs;

use super::{
    PlacementCheck, forget_placement_verified, placement_check_for, record_placement_verified,
};

const PLACEMENT_ENVIRONMENT: &str = "macOS 27.0.1 (26A434) on Mac99,1";

#[test]
fn should_require_the_check_without_a_record() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_skip_the_check_after_recording_it_in_this_environment() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::AlreadyProven
    );
}

#[test]
fn should_accept_a_record_that_ends_with_a_line_break() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    fs::write(
        set_directory.path().join("layer-7.placement-verified"),
        format!("{PLACEMENT_ENVIRONMENT}\n"),
    )
    .expect("a record exists");

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::AlreadyProven
    );
}

#[test]
fn should_keep_the_record_beside_the_program_and_not_inside_it() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    fs::create_dir_all(&kept_program_path).expect("a program folder");

    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    assert!(
        set_directory
            .path()
            .join("layer-7.placement-verified")
            .is_file()
    );
    assert_eq!(
        fs::read_dir(&kept_program_path)
            .expect("the program folder lists")
            .count(),
        0
    );
}

#[test]
fn should_require_the_check_for_a_record_from_another_macos_build() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    record_placement_verified(&kept_program_path, "macOS 27.0.1 (26A400) on Mac99,1");

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_require_the_check_for_a_record_from_another_mac() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    record_placement_verified(&kept_program_path, "macOS 27.0.1 (26A434) on Mac98,2");

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_require_the_check_when_the_environment_is_unknown() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    assert_eq!(
        placement_check_for(&kept_program_path, None),
        PlacementCheck::Required
    );
}

#[test]
fn should_require_the_check_for_an_empty_record_against_an_empty_environment() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    record_placement_verified(&kept_program_path, "");

    assert_eq!(
        placement_check_for(&kept_program_path, Some("")),
        PlacementCheck::Required
    );
}

#[test]
fn should_keep_the_record_of_one_layer_apart_from_another() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");

    record_placement_verified(
        &set_directory.path().join("layer-1.mlmodelc"),
        PLACEMENT_ENVIRONMENT,
    );

    assert_eq!(
        placement_check_for(
            &set_directory.path().join("layer-10.mlmodelc"),
            Some(PLACEMENT_ENVIRONMENT)
        ),
        PlacementCheck::Required
    );
}

#[test]
fn should_not_panic_when_the_record_cannot_be_written() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("absent-set/layer-7.mlmodelc");

    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_forget_a_record_and_tolerate_a_missing_one() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    forget_placement_verified(&kept_program_path);
    forget_placement_verified(&kept_program_path);

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_name_the_macos_build_and_the_mac_in_an_environment() {
    assert_eq!(
        crate::qwen3_5_resident::host_description::host_environment_of(
            "27.0.1", "26A434", "Mac99,1"
        ),
        PLACEMENT_ENVIRONMENT
    );
}

#[test]
fn should_describe_this_process_environment_once() {
    let placement_environment =
        crate::qwen3_5_resident::host_description::current_host_environment()
            .expect("macOS describes itself");

    assert!(placement_environment.starts_with("macOS "));
    assert!(placement_environment.contains(" on "));
    assert_eq!(
        crate::qwen3_5_resident::host_description::current_host_environment(),
        Some(placement_environment)
    );
}
