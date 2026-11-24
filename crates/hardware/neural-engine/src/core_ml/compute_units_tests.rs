use super::{
    ComputePlacement, CoreMlComputeUnits, layer_placement_from_native, placement_from_native_code,
};
use crate::core_ml::ffi::NativeLayerPlacement;

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
