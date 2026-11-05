use super::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProjection, CoreMlProjectionError,
    check_element_count, layer_placement_from_native, native_error_text,
    placement_from_native_code,
};
use crate::ml_program_mlp_document::MlpFp16Weights;
use crate::ml_program_projection_document::MlProgramWeights;
use crate::native_bridge::NativeLayerPlacement;
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

#[test]
fn should_map_every_compute_unit_choice_to_the_bridge_code() {
    assert_eq!(CoreMlComputeUnits::CpuOnly.native_code(), 0);
    assert_eq!(CoreMlComputeUnits::CpuAndGpu.native_code(), 1);
    assert_eq!(CoreMlComputeUnits::CpuAndNeuralEngine.native_code(), 2);
}

#[test]
fn should_decode_each_native_placement_code() {
    assert_eq!(placement_from_native_code(1), ComputePlacement::Cpu);
    assert_eq!(placement_from_native_code(2), ComputePlacement::Gpu);
    assert_eq!(
        placement_from_native_code(3),
        ComputePlacement::NeuralEngine
    );
}

#[test]
fn should_treat_an_unrecognized_placement_code_as_unknown() {
    assert_eq!(placement_from_native_code(0), ComputePlacement::Unknown);
    assert_eq!(placement_from_native_code(-7), ComputePlacement::Unknown);
}

#[test]
fn should_decode_the_supported_device_flags() {
    let placement = layer_placement_from_native(NativeLayerPlacement {
        preferred_placement: 3,
        neural_engine_supported: 1,
        gpu_supported: 0,
        cpu_supported: 1,
        operations_off_neural_engine: 2,
    });

    assert_eq!(placement.preferred, ComputePlacement::NeuralEngine);
    assert!(placement.neural_engine_supported);
    assert!(!placement.gpu_supported);
    assert!(placement.cpu_supported);
    assert_eq!(placement.operations_off_neural_engine, 2);
}

#[test]
fn should_reject_a_buffer_of_the_wrong_length() {
    assert!(check_element_count(4, 4).is_ok());
    assert!(check_element_count(4, 3).is_err());
}

#[test]
fn should_read_the_native_error_message_up_to_its_terminator() {
    let mut native_error = [0 as std::ffi::c_char; 8];
    for (byte_index, byte) in b"bad".iter().enumerate() {
        native_error[byte_index] = *byte as std::ffi::c_char;
    }

    assert_eq!(native_error_text(&native_error), "bad");
}
