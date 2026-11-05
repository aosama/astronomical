use std::path::Path;

use crate::core_ml::{CoreMlComputeUnits, CoreMlProjection, CoreMlProjectionError};
use crate::projection_shape::ProjectionShape;

#[test]
fn should_refuse_a_model_path_with_an_interior_nul_before_calling_core_ml() {
    let shape = ProjectionShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProjection::load_from_path(
        Path::new("not\0a-path.mlpackage"),
        shape,
        CoreMlComputeUnits::CpuOnly,
    );

    assert!(matches!(load_result, Err(CoreMlProjectionError::ModelPath)));
}

#[test]
fn should_report_core_ml_load_failure_for_a_model_that_does_not_exist() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProjectionShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProjection::load_from_path(
        &model_directory.path().join("absent.mlpackage"),
        shape,
        CoreMlComputeUnits::CpuOnly,
    );

    assert!(matches!(load_result, Err(CoreMlProjectionError::Load(_))));
}
