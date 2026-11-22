use std::path::Path;

use crate::model_serving_default_features::{
    DEFAULT_WORLD_PACKAGES, QUANTIZATION_PROBE_PACKAGE, package_clippy_command,
    packages_this_host_builds,
};

#[test]
fn should_lint_every_default_world_crate_when_nothing_is_left_out() {
    let buildable_packages = packages_this_host_builds(&DEFAULT_WORLD_PACKAGES, &[]);

    assert_eq!(buildable_packages, DEFAULT_WORLD_PACKAGES);
}

#[test]
fn should_drop_only_the_left_out_crates_and_keep_the_order() {
    let packages_left_out = [
        "engine-core".to_owned(),
        "qwen35".to_owned(),
        "model-serving".to_owned(),
    ];

    let buildable_packages = packages_this_host_builds(&DEFAULT_WORLD_PACKAGES, &packages_left_out);

    assert_eq!(
        buildable_packages,
        [
            "model-primitives",
            "speech",
            "artifact-validation",
            "image-generation",
            "decision-models"
        ]
    );
}

#[test]
fn should_leave_nothing_to_lint_when_every_crate_is_left_out() {
    let packages_left_out = [QUANTIZATION_PROBE_PACKAGE.to_owned()];

    assert!(
        packages_this_host_builds(&[QUANTIZATION_PROBE_PACKAGE], &packages_left_out).is_empty()
    );
}

#[test]
fn should_keep_the_macos_default_world_command_exactly_as_the_hook_ran_it() {
    let clippy_command = package_clippy_command(Path::new("/workspace"), &DEFAULT_WORLD_PACKAGES);

    let clippy_arguments: Vec<_> = clippy_command.get_args().collect();
    assert_eq!(
        clippy_arguments,
        [
            "clippy",
            "-p",
            "model-primitives",
            "-p",
            "speech",
            "-p",
            "artifact-validation",
            "-p",
            "image-generation",
            "-p",
            "engine-core",
            "-p",
            "qwen35",
            "-p",
            "decision-models",
            "-p",
            "model-serving",
            "--all-targets",
        ]
    );
}

#[test]
fn should_keep_the_quantization_probe_in_its_own_call() {
    let clippy_command =
        package_clippy_command(Path::new("/workspace"), &[QUANTIZATION_PROBE_PACKAGE]);

    let clippy_arguments: Vec<_> = clippy_command.get_args().collect();
    assert_eq!(
        clippy_arguments,
        ["clippy", "-p", "qwen35-quantization-probe", "--all-targets"]
    );
}
