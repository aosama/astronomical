//! Bounded capability probe for the folded ternary family's one-launch
//! signed Hadamard rotation.
//!
//! The probe compiles the kernel at the shipped pack's block size, rotates a
//! small patterned float16 activation, and compares the result with the
//! composed public-MLX route (float32 cast, sign multiply, block Hadamard,
//! cast back) that serves as the production fallback. The comparison is
//! exact, not tolerance-based: the kernel exists to remove launches, never
//! to change a bit, and the upstream parity journey depends on that.

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxCError, MlxDtype};

use super::{CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError};
use crate::prism_hadamard_qwen35::model::signed_rotation_kernel::{
    apply_signed_rotation, signed_rotation_kernel,
};

/// The shipped pack's Hadamard block, so the probe exercises the exact
/// radix schedule production runs.
const PROBE_BLOCK_SIZE: u32 = 1024;
/// Two blocks per row so the per-block sign indexing is exercised.
const PROBE_WIDTH: i32 = 2 * PROBE_BLOCK_SIZE as i32;
const PROBE_ROW_COUNT: i32 = 3;

impl CustomMetalKernelFamily {
    /// Registry identifier of the folded ternary signed-rotation family. The
    /// family owns this constant; the capability module stays agnostic.
    pub const PRISM_HADAMARD_SIGNED_ROTATION: Self = Self::new("prism_hadamard_signed_rotation");
}

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
        CustomMetalKernelFamily::PRISM_HADAMARD_SIGNED_ROTATION
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = signed_rotation_kernel(PROBE_BLOCK_SIZE).map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let (activations, signs) = probe_arrays(self.runtime).map_err(execution_error)?;
        let fused = apply_signed_rotation(
            self.runtime,
            &kernel,
            &activations,
            &signs,
            PROBE_BLOCK_SIZE,
        )
        .map_err(execution_error)?;
        let composed =
            composed_rotation(self.runtime, &activations, &signs).map_err(execution_error)?;
        let fused_values = float32_values(self.runtime, &fused).map_err(execution_error)?;
        let composed_values = float32_values(self.runtime, &composed).map_err(execution_error)?;
        assert_bit_exact(&fused_values, &composed_values)
    }
}

/// Patterned float16 activations spanning both signs and several
/// magnitudes, and an alternating-run sign vector.
fn probe_arrays(runtime: &MlxRuntime) -> Result<(MlxArray, MlxArray), MlxCError> {
    let activation_values = (0..(PROBE_ROW_COUNT * PROBE_WIDTH) as usize)
        .map(|value_index| ((value_index % 29) as f32 - 14.0) / 16.0)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, PROBE_ROW_COUNT, PROBE_WIDTH])
        .and_then(|array| runtime.astype(&array, MlxDtype::Float16))?;
    let sign_values = (0..PROBE_WIDTH as usize)
        .map(|sign_index| if (sign_index / 3) % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<f32>>();
    let signs = runtime.array_from_f32(&sign_values, &[PROBE_WIDTH])?;
    Ok((activations, signs))
}

/// The composed public-MLX rotation the fused kernel must reproduce.
fn composed_rotation(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    signs: &MlxArray,
) -> Result<MlxArray, MlxCError> {
    let signed = runtime
        .astype(activations, MlxDtype::Float32)
        .and_then(|float32_activations| runtime.multiply(&float32_activations, signs))?;
    let block = PROBE_BLOCK_SIZE as i32;
    let blocked = runtime.reshape(&signed, &[PROBE_ROW_COUNT * (PROBE_WIDTH / block), block])?;
    let transformed =
        runtime.hadamard_transform(&blocked, Some(1.0 / (PROBE_BLOCK_SIZE as f32).sqrt()))?;
    let restored = runtime.reshape(&transformed, &[1, PROBE_ROW_COUNT, PROBE_WIDTH])?;
    runtime.astype(&restored, MlxDtype::Float16)
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
