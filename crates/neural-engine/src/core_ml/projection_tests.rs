use crate::core_ml::{CoreMlComputeUnits, CoreMlProjection, CoreMlProjectionError};
use crate::ml_program_mlp_document::MlpFp16Weights;
use crate::ml_program_projection_document::MlProgramWeights;
use crate::projection_model_interface::ProjectionDocumentError;
use crate::projection_shape::ProjectionShape;

#[test]
fn should_refuse_an_mlp_program_whose_weights_do_not_fit_before_writing_anything() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProjectionShape::new(4, 4, 1).expect("valid shape");

    let load_result = CoreMlProjection::load_ml_program_mlp(
        shape,
        8,
        MlpFp16Weights {
            gate_and_up: &[0; 7],
            down: &[0; 32],
        },
        4,
        CoreMlComputeUnits::CpuOnly,
        model_directory.path(),
    );

    assert!(matches!(
        load_result,
        Err(CoreMlProjectionError::Document(
            ProjectionDocumentError::WeightCountMismatch { .. }
        ))
    ));
    assert_eq!(
        std::fs::read_dir(model_directory.path())
            .expect("the directory is readable")
            .count(),
        0
    );
}

#[test]
fn should_refuse_an_ml_program_whose_weights_do_not_fit_before_writing_anything() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProjectionShape::new(4, 2, 1).expect("valid shape");

    let load_result = CoreMlProjection::load_ml_program(
        shape,
        MlProgramWeights::Fp16(&[0; 7]),
        CoreMlComputeUnits::CpuOnly,
        1,
        model_directory.path(),
    );

    assert!(matches!(
        load_result,
        Err(CoreMlProjectionError::Document(
            ProjectionDocumentError::WeightCountMismatch {
                expected: 8,
                actual: 7
            }
        ))
    ));
    assert!(!model_directory.path().join("projection.mlpackage").exists());
}
