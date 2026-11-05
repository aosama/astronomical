use std::path::Path;

use super::compile_model_to;
use crate::core_ml::CoreMlProjectionError;

#[test]
fn should_refuse_a_package_path_with_an_interior_nul() {
    let scratch_directory = tempfile::tempdir().expect("a scratch directory");

    let compile_result = compile_model_to(
        Path::new("not\0a-path.mlpackage"),
        &scratch_directory.path().join("compiled.mlmodelc"),
    );

    assert!(matches!(
        compile_result,
        Err(CoreMlProjectionError::ModelPath)
    ));
}

#[test]
fn should_report_core_ml_failure_and_leave_no_compiled_model_for_an_absent_package() {
    let scratch_directory = tempfile::tempdir().expect("a scratch directory");
    let destination = scratch_directory.path().join("compiled.mlmodelc");

    let compile_result = compile_model_to(
        &scratch_directory.path().join("absent.mlpackage"),
        &destination,
    );

    assert!(matches!(
        compile_result,
        Err(CoreMlProjectionError::Load(_))
    ));
    assert!(!destination.exists());
}

#[test]
fn should_create_missing_destination_folders_before_compiling() {
    let scratch_directory = tempfile::tempdir().expect("a scratch directory");
    let destination_parent = scratch_directory.path().join("cache").join("program-set");

    let compile_result = compile_model_to(
        &scratch_directory.path().join("absent.mlpackage"),
        &destination_parent.join("compiled.mlmodelc"),
    );

    assert!(compile_result.is_err());
    assert!(destination_parent.is_dir());
}
