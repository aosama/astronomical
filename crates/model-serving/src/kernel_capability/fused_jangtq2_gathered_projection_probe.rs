//! Bounded capability probe for the fused jangtq2 gathered projection kernel.
//!
//! The probe compiles the kernel and launches it on a minimal two-expert,
//! two-row, one-group (32 inputs) 2-bit page with two assignments. Expected
//! values come from the format's own host arithmetic (`unpack`, cubic level,
//! row scale, float64 dot product), so a wrong GPU result — including the
//! all-zeros signature of a silently dropped dispatch — fails the verdict.

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use astronomical_jangtq2_qwen35::codebook;
use astronomical_jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel;

const PROBE_BIT_WIDTH: u32 = 2;
const PROBE_EXPERT_COUNT: usize = 2;
const PROBE_OUTPUT_ROWS: usize = 2;
/// One 32-code group: the smallest row the format allows.
const PROBE_INPUT_WIDTH: usize = 32;
/// 32 codes × 2 bits = 64 bits = two uint32 words per row.
const PROBE_WORDS_PER_ROW: usize = 2;
/// Two assignments routed to the two experts in reverse order.
const PROBE_EXPERT_INDICES: [i32; 2] = [1, 0];

impl CustomMetalKernelFamily {
    /// Registry identifier of the fused jangtq2 gathered projection family.
    /// The family owns this constant; the capability module stays agnostic.
    pub const FUSED_JANGTQ2_GATHERED_PROJECTION: Self =
        Self::new("fused_jangtq2_gathered_projection");
}

/// Supplies this family's probes for the once-per-process probe run. The
/// worker registration list is the only central touch point.
pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(FusedJangtq2GatheredProjectionProbe::new(runtime))]
}

pub struct FusedJangtq2GatheredProjectionProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> FusedJangtq2GatheredProjectionProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for FusedJangtq2GatheredProjectionProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::FUSED_JANGTQ2_GATHERED_PROJECTION
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = FusedJangtq2GatheredProjectionKernel::new().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let projected = fused_projection_probe_journey(self.runtime, &kernel).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        validate_probe_outputs(&projected, &expected_probe_projection())
    }
}

/// Fixed packed words: every code value appears in every row so the level
/// table is exercised end to end.
fn probe_packed_words() -> Vec<u32> {
    (0..PROBE_EXPERT_COUNT * PROBE_OUTPUT_ROWS * PROBE_WORDS_PER_ROW)
        .map(|word_index| 0x1B2E_39C4_u32.rotate_left((word_index as u32) * 5))
        .collect()
}

/// Row scales exactly representable in float16.
fn probe_row_scales() -> Vec<f32> {
    vec![0.5, 1.5, 2.0, 0.25]
}

fn probe_activations() -> Vec<f32> {
    (0..PROBE_INPUT_WIDTH)
        .map(|element_index| (element_index as f32 - 15.5) / 8.0)
        .collect()
}

/// `scale[e, r] · Σ_j level(code[e, r, j]) · x[j]` in float64 for each
/// assignment and output row.
fn expected_probe_projection() -> Vec<f32> {
    let packed_words = probe_packed_words();
    let row_scales = probe_row_scales();
    let activations = probe_activations();
    let mut expected = Vec::with_capacity(PROBE_EXPERT_INDICES.len() * PROBE_OUTPUT_ROWS);
    for &expert_index in &PROBE_EXPERT_INDICES {
        for output_row in 0..PROBE_OUTPUT_ROWS {
            let row_index = expert_index as usize * PROBE_OUTPUT_ROWS + output_row;
            let row_start = row_index * PROBE_WORDS_PER_ROW;
            let codes = codebook::unpack_codes_from_packed_row(
                &packed_words[row_start..row_start + PROBE_WORDS_PER_ROW],
                PROBE_BIT_WIDTH,
            )
            .expect("the probe row is one whole 32-code group");
            let dot: f64 = codes
                .iter()
                .zip(&activations)
                .map(|(&code, &activation)| {
                    f64::from(
                        codebook::codebook_level_value(PROBE_BIT_WIDTH, u32::from(code))
                            .expect("2-bit codes are always in range"),
                    ) * f64::from(activation)
                })
                .sum();
            expected.push((dot * f64::from(row_scales[row_index])) as f32);
        }
    }
    expected
}

fn fused_projection_probe_journey(
    runtime: &MlxRuntime,
    kernel: &FusedJangtq2GatheredProjectionKernel,
) -> Result<Vec<f32>, MlxRuntimeError> {
    let packed_code_words = MlxArray::from_u32(
        &probe_packed_words(),
        &[
            PROBE_EXPERT_COUNT as i32,
            PROBE_OUTPUT_ROWS as i32,
            PROBE_WORDS_PER_ROW as i32,
        ],
    )?;
    let row_scales = runtime.astype(
        &runtime.array_from_f32(
            &probe_row_scales(),
            &[PROBE_EXPERT_COUNT as i32, PROBE_OUTPUT_ROWS as i32],
        )?,
        MlxDtype::Float16,
    )?;
    let hidden = runtime.array_from_f32(&probe_activations(), &[1, PROBE_INPUT_WIDTH as i32])?;
    let expert_indices =
        MlxArray::from_i32(&PROBE_EXPERT_INDICES, &[PROBE_EXPERT_INDICES.len() as i32])?;
    let projected = kernel.apply(
        runtime,
        &hidden,
        &expert_indices,
        &packed_code_words,
        &row_scales,
        PROBE_BIT_WIDTH,
    )?;
    runtime
        .array_to_vec_f32(&projected)
        .map_err(MlxRuntimeError::from)
}
