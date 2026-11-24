use std::path::Path;

use crate::core_ml::{CoreMlComputeUnits, CoreMlProgram, CoreMlProgramError};
use crate::program_shape::ProgramShape;

#[test]
fn should_refuse_a_model_path_with_an_interior_nul_before_calling_core_ml() {
    let shape = ProgramShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProgram::load_from_path(
        Path::new("not\0a-path.mlpackage"),
        shape,
        CoreMlComputeUnits::CpuOnly,
    );

    assert!(matches!(load_result, Err(CoreMlProgramError::ModelPath)));
}

#[test]
fn should_report_core_ml_load_failure_for_a_model_that_does_not_exist() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProgramShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProgram::load_from_path(
        &model_directory.path().join("absent.mlpackage"),
        shape,
        CoreMlComputeUnits::CpuOnly,
    );

    assert!(matches!(load_result, Err(CoreMlProgramError::Load(_))));
}

#[test]
fn should_report_core_ml_load_failure_without_a_placement_check_for_a_model_that_does_not_exist() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProgramShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProgram::load_from_path_without_placement_check(
        &model_directory.path().join("absent.mlmodelc"),
        shape,
        CoreMlComputeUnits::CpuOnly,
    );

    assert!(matches!(load_result, Err(CoreMlProgramError::Load(_))));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn should_explain_that_core_ml_is_unavailable_when_loading_off_macos() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProgramShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProgram::load_from_path(
        &model_directory.path().join("any.mlpackage"),
        shape,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    );

    let Err(CoreMlProgramError::Load(load_failure)) = load_result else {
        panic!("loading must fail where Core ML does not exist");
    };
    assert!(
        load_failure.contains("only on macOS"),
        "the failure should say why: {load_failure}"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn should_explain_that_core_ml_is_unavailable_when_compiling_off_macos() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");

    let compile_result = crate::core_ml::compile_model_to(
        &model_directory.path().join("any.mlpackage"),
        &model_directory.path().join("compiled/any.mlmodelc"),
    );

    let Err(CoreMlProgramError::Load(compile_failure)) = compile_result else {
        panic!("compiling must fail where Core ML does not exist");
    };
    assert!(compile_failure.contains("only on macOS"));
}
