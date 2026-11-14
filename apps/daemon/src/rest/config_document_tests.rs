//! Which settings the configuration response marks as unavailable, by where
//! the instance keeps its state.

use std::path::PathBuf;

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};

use super::unavailable_settings;

#[test]
fn should_mark_the_folder_settings_unavailable_for_a_sandboxed_instance() {
    let group_container_paths = AstronomicalInstancePaths::for_application_group_container(
        PathBuf::from("/Users/example/Library/Group Containers/TEAM123456.astronomical"),
        AstronomicalRuntimeInstance::Stable,
    );

    assert_eq!(
        unavailable_settings(&group_container_paths),
        [
            "runtime.model_directories",
            "runtime.model_download_directory",
            "runtime.neural_engine_program_directory",
        ]
    );
}

#[test]
fn should_mark_nothing_unavailable_for_a_direct_instance() {
    let direct_paths = AstronomicalInstancePaths::for_home_directory(
        PathBuf::from("/Users/example"),
        AstronomicalRuntimeInstance::Stable,
    );

    assert!(unavailable_settings(&direct_paths).is_empty());
}

#[test]
fn should_mark_nothing_unavailable_for_a_caller_chosen_state_directory() {
    let custom_paths = AstronomicalInstancePaths::for_state_directory(
        PathBuf::from("/somewhere/else"),
        AstronomicalRuntimeInstance::Development,
    );

    assert!(unavailable_settings(&custom_paths).is_empty());
}
