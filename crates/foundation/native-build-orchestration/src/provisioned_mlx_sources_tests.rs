//! Where native code finds the patched MLX sources, and what it is told
//! when they are not there.

use std::fs;

use super::{pinned_mlx_macos_deployment_target, provisioned_mlx_sources_under};

const SOURCE_IDENTITY: &str = "0123456789abcdef";

#[test]
fn should_point_at_both_source_trees_of_a_completed_extraction() {
    let dependencies_directory = tempfile::tempdir().expect("a temporary directory");
    let extraction_directory = dependencies_directory
        .path()
        .join("bindgen-headers")
        .join(SOURCE_IDENTITY);
    fs::create_dir_all(&extraction_directory).expect("the extraction directory");
    fs::write(extraction_directory.join("complete"), b"").expect("the completion marker");

    let provisioned_sources =
        provisioned_mlx_sources_under(dependencies_directory.path(), SOURCE_IDENTITY)
            .expect("a completed extraction resolves");

    assert_eq!(
        provisioned_sources.mlx_source_directory,
        extraction_directory.join("mlx-src")
    );
    assert_eq!(
        provisioned_sources.mlx_c_source_directory,
        extraction_directory.join("mlx_c-src")
    );
}

#[test]
fn should_name_the_provisioning_command_when_nothing_is_extracted() {
    let dependencies_directory = tempfile::tempdir().expect("a temporary directory");

    let failure = provisioned_mlx_sources_under(dependencies_directory.path(), SOURCE_IDENTITY)
        .expect_err("a missing extraction is refused");

    assert!(
        failure
            .to_string()
            .contains("cargo xtask provision-bindgen-headers"),
        "{failure}"
    );
}

#[test]
fn should_refuse_an_extraction_that_never_finished() {
    let dependencies_directory = tempfile::tempdir().expect("a temporary directory");
    let extraction_directory = dependencies_directory
        .path()
        .join("bindgen-headers")
        .join(SOURCE_IDENTITY);
    // The trees exist but the marker that says every file was verified does
    // not, which is the state an interrupted extraction leaves behind.
    fs::create_dir_all(extraction_directory.join("mlx-src")).expect("the MLX tree");
    fs::create_dir_all(extraction_directory.join("mlx_c-src")).expect("the MLX-C tree");

    assert!(provisioned_mlx_sources_under(dependencies_directory.path(), SOURCE_IDENTITY).is_err());
}

fn repository_with_pins(pins: &[(&str, &str)]) -> tempfile::TempDir {
    let repository_root = tempfile::tempdir().expect("a temporary directory");
    let pins_directory = repository_root.path().join("third-party/pins");
    fs::create_dir_all(&pins_directory).expect("the pins directory");
    for (pin_file_name, pin_text) in pins {
        fs::write(pins_directory.join(pin_file_name), pin_text).expect("a pin file");
    }
    repository_root
}

#[test]
fn should_read_the_deployment_target_from_the_pin_that_sets_it() {
    let repository_root = repository_with_pins(&[
        (
            "mlx-c-v0.8.0.cmake",
            "set(ASTRONOMICAL_MLX_C_VERSION \"0.8.0\")\n",
        ),
        (
            "mlx-v0.32.4.cmake",
            "set(ASTRONOMICAL_MLX_VERSION \"0.32.4\")\n  set(ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET \"26.2\")\n",
        ),
    ]);

    assert_eq!(
        pinned_mlx_macos_deployment_target(repository_root.path()).expect("the pin sets it"),
        "26.2"
    );
}

#[test]
fn should_refuse_pins_that_set_no_deployment_target() {
    let repository_root = repository_with_pins(&[
        (
            "mlx-v0.32.4.cmake",
            "set(ASTRONOMICAL_MLX_VERSION \"0.32.4\")\n",
        ),
        // Not a pin: a note beside them that happens to quote the setting.
        (
            "notes.txt",
            "set(ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET \"26.2\")\n",
        ),
    ]);

    assert!(pinned_mlx_macos_deployment_target(repository_root.path()).is_err());
}

#[test]
fn should_refuse_a_checkout_without_pins() {
    let repository_root = tempfile::tempdir().expect("a temporary directory");

    assert!(pinned_mlx_macos_deployment_target(repository_root.path()).is_err());
}
