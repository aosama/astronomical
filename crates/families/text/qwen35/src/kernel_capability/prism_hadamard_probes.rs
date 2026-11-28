//! Bounded capability probe for the folded ternary family's one-launch
//! signed Hadamard rotation.
//!
//! The probe compiles the kernel at one Hadamard block, rotates a
//! small patterned float16 activation, and compares the result with the
//! composed public-MLX route (float32 cast, sign multiply, block Hadamard,
//! cast back) that serves as the production fallback. The comparison is
//! exact, not tolerance-based: the kernel exists to remove launches, never
//! to change a bit, and the upstream parity journey depends on that.

use mlx_bridge::{MlxArray, MlxCError, MlxDtype};
use runtime_integration::MlxRuntime;

use super::{CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError};
use crate::qwen3_5_core::model_math::signed_rotation_kernel::{
    apply_signed_rotation, signed_rotation_kernel,
};

/// The block the worker probe checks, the 27B packs' block. It learns
/// whether this GPU runs the kernel at all; the probe tests prove the bits at
/// every block a pack may declare (512 for the 2B, 4096 for the 9B), each a
/// different radix schedule.
const PROBE_BLOCK_SIZE: u32 = 1024;
/// Two blocks per row so the per-block sign indexing is exercised.
const PROBE_BLOCKS_PER_ROW: i32 = 2;
const PROBE_ROW_COUNT: i32 = 3;

/// Registry identifier of the folded ternary signed-rotation family. The
/// family owns this constant; the capability module stays agnostic.
pub const PRISM_HADAMARD_SIGNED_ROTATION: CustomMetalKernelFamily =
    CustomMetalKernelFamily::new("prism_hadamard_signed_rotation");

/// Supplies this family's probes for the once-per-process probe run. The
/// worker registration list is the only central touch point.
pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(PrismHadamardSignedRotationProbe::new(runtime))]
}

pub struct PrismHadamardSignedRotationProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> PrismHadamardSignedRotationProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for PrismHadamardSignedRotationProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        PRISM_HADAMARD_SIGNED_ROTATION
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        verify_signed_rotation_against_the_composed_route(self.runtime, PROBE_BLOCK_SIZE)
    }
}

/// Builds the one-launch rotation at `block_size`, rotates two blocks per row
/// of a patterned activation, and requires the composed route's exact bits.
/// The worker probe runs it at one block to learn whether this GPU runs the
/// kernel at all; the probe tests run it at every block a pack may declare,
/// since each block takes a different butterfly radix schedule.
pub(super) fn verify_signed_rotation_against_the_composed_route(
    runtime: &MlxRuntime,
    block_size: u32,
) -> Result<(), KernelCapabilityError> {
    let kernel =
        signed_rotation_kernel(block_size).map_err(|error| KernelCapabilityError::Compilation {
            description: error.to_string(),
        })?;
    let (activations, signs) = probe_arrays(runtime, block_size).map_err(execution_error)?;
    let fused = apply_signed_rotation(runtime, &kernel, &activations, &signs, block_size)
        .map_err(execution_error)?;
    let composed =
        composed_rotation(runtime, &activations, &signs, block_size).map_err(execution_error)?;
    let fused_values = float32_values(runtime, &fused).map_err(execution_error)?;
    let composed_values = float32_values(runtime, &composed).map_err(execution_error)?;
    assert_bit_exact(&fused_values, &composed_values)
}

/// Patterned float16 activations spanning both signs and several
/// magnitudes, and an alternating-run sign vector.
fn probe_arrays(runtime: &MlxRuntime, block_size: u32) -> Result<(MlxArray, MlxArray), MlxCError> {
    let probe_width = probe_width(block_size);
    let activation_values = (0..(PROBE_ROW_COUNT * probe_width) as usize)
        .map(|value_index| ((value_index % 29) as f32 - 14.0) / 16.0)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, PROBE_ROW_COUNT, probe_width])
        .and_then(|array| runtime.astype(&array, MlxDtype::Float16))?;
    let sign_values = (0..probe_width as usize)
        .map(|sign_index| if (sign_index / 3) % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<f32>>();
    let signs = runtime.array_from_f32(&sign_values, &[probe_width])?;
    Ok((activations, signs))
}

/// The composed public-MLX rotation the fused kernel must reproduce.
fn composed_rotation(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    signs: &MlxArray,
    block_size: u32,
) -> Result<MlxArray, MlxCError> {
    let probe_width = probe_width(block_size);
    let signed = runtime
        .astype(activations, MlxDtype::Float32)
        .and_then(|float32_activations| runtime.multiply(&float32_activations, signs))?;
    let block = block_size as i32;
    let blocked = runtime.reshape(&signed, &[PROBE_ROW_COUNT * PROBE_BLOCKS_PER_ROW, block])?;
    let transformed =
        runtime.hadamard_transform(&blocked, Some(1.0 / (block_size as f32).sqrt()))?;
    let restored = runtime.reshape(&transformed, &[1, PROBE_ROW_COUNT, probe_width])?;
    runtime.astype(&restored, MlxDtype::Float16)
}

const fn probe_width(block_size: u32) -> i32 {
    PROBE_BLOCKS_PER_ROW * block_size as i32
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray) -> Result<Vec<f32>, MlxCError> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
}

fn execution_error(error: MlxCError) -> KernelCapabilityError {
    KernelCapabilityError::Execution {
        description: error.to_string(),
    }
}

fn assert_bit_exact(
    fused_values: &[f32],
    composed_values: &[f32],
) -> Result<(), KernelCapabilityError> {
    if fused_values.len() != composed_values.len() {
        return Err(KernelCapabilityError::OutputMismatch {
            description: format!(
                "the fused rotation produced {} values but the composed route produced {}",
                fused_values.len(),
                composed_values.len()
            ),
        });
    }
    if fused_values.iter().all(|value| *value == 0.0) {
        return Err(KernelCapabilityError::OutputMismatch {
            description:
                "the fused rotation returned all zeros, the signature of a dropped dispatch"
                    .to_owned(),
        });
    }
    for (value_index, (fused_value, composed_value)) in
        fused_values.iter().zip(composed_values).enumerate()
    {
        if fused_value.to_bits() != composed_value.to_bits() {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "fused rotation value {value_index} read {fused_value:.9e} but the composed route produced {composed_value:.9e}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "prism_hadamard_probes_tests.rs"]
mod prism_hadamard_probes_tests;
