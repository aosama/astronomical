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
