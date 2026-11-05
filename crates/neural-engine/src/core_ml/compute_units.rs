//! Which devices Core ML may use, and where it reports having placed a model.

use crate::core_ml::ffi::NativeLayerPlacement;

/// The devices Core ML may schedule on (`MLComputeUnits`). There is no
/// "Neural Engine only" setting; excluding the GPU is how a GPU fallback is
/// kept from passing as a Neural Engine result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
}

impl CoreMlComputeUnits {
    pub(super) fn native_code(self) -> i32 {
        match self {
            Self::CpuOnly => 0,
            Self::CpuAndGpu => 1,
            Self::CpuAndNeuralEngine => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputePlacement {
    Unknown,
    Cpu,
    Gpu,
    NeuralEngine,
}

/// `MLComputePlan`'s verdict for the projection layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPlacement {
    pub preferred: ComputePlacement,
    pub neural_engine_supported: bool,
    pub gpu_supported: bool,
    pub cpu_supported: bool,
    /// Across the whole ML Program, not only the verdict's convolution.
    pub operations_off_neural_engine: i32,
}

fn placement_from_native_code(native_code: i32) -> ComputePlacement {
    match native_code {
        1 => ComputePlacement::Cpu,
        2 => ComputePlacement::Gpu,
        3 => ComputePlacement::NeuralEngine,
        _ => ComputePlacement::Unknown,
    }
}

pub(super) fn layer_placement_from_native(
    native_placement: NativeLayerPlacement,
) -> LayerPlacement {
    LayerPlacement {
        preferred: placement_from_native_code(native_placement.preferred_placement),
        neural_engine_supported: native_placement.neural_engine_supported != 0,
        gpu_supported: native_placement.gpu_supported != 0,
        cpu_supported: native_placement.cpu_supported != 0,
        operations_off_neural_engine: native_placement.operations_off_neural_engine,
    }
}

#[cfg(test)]
#[path = "compute_units_tests.rs"]
mod tests;
