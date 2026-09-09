//! Bounded capability probe for the fused quantized expert decode kernels.
//!
//! The probe compiles all three kernels and executes the value-expert kernel
//! and the routed FFN chain on a minimal two-assignment, two-expert,
//! one-quantization-group shape. Expected values are derived from the
//! documented affine decode (`scale * nibble + bias`, silu activation,
//! score-weighted reduction) with plain host arithmetic, so a wrong GPU
//! result — including the all-zeros signature of a silently dropped dispatch —
//! fails the capability verdict.

use astronomical_runtime_integration::MlxRuntime;

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use crate::k2_horizon_mova::{FusedExpertDecodeKernels, K2HorizonMoVAAffineLinear};
use crate::performance_attribution::PerformanceAttribution;

/// One quantization group, one 64-wide input, two assignments, two experts.
const PROBE_ASSIGNMENT_COUNT: usize = 2;
const PROBE_EXPERT_COUNT: usize = 2;
const PROBE_INPUT_DIMENSION: usize = 64;
const PROBE_INTERMEDIATE_DIMENSION: usize = 64;
const PROBE_VALUE_OUTPUT_DIMENSION: usize = 2;
const PROBE_DOWN_OUTPUT_DIMENSION: usize = 4;
const PROBE_GROUP_SIZE: u32 = 64;
const PROBE_BITS: u32 = 4;

pub struct FusedQuantizedExpertDecodeProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> FusedQuantizedExpertDecodeProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for FusedQuantizedExpertDecodeProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::FusedQuantizedExpertDecode
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernels = FusedExpertDecodeKernels::new().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let mut probe_attribution = PerformanceAttribution::disabled();
        self.probe_value_expert_kernel(&kernels, &mut probe_attribution)?;
        self.probe_routed_ffn_kernels(&kernels, &mut probe_attribution)
    }
}

impl FusedQuantizedExpertDecodeProbe<'_> {
    fn probe_value_expert_kernel(
        &self,
        kernels: &FusedExpertDecodeKernels,
        probe_attribution: &mut PerformanceAttribution,
    ) -> Result<(), KernelCapabilityError> {
        let hidden = self
            .runtime
            .array_from_f32(&hidden_values(), &[1, PROBE_INPUT_DIMENSION as i32])
            .map_err(probe_execution)?;
        let indices = self
            .runtime
            .array_from_u32(&[0, 1], &[1, PROBE_ASSIGNMENT_COUNT as i32])
            .map_err(probe_execution)?;
        let scores = self
            .runtime
            .array_from_f32(&[1.0, 0.5], &[1, PROBE_ASSIGNMENT_COUNT as i32])
            .map_err(probe_execution)?;
        // Expert zero scales and biases every nibble by (2, 1); expert one by
        // (3, 0). Every output row holds the same weight row, so the expected
        // reduction is the same dot per output index.
        let packed = self
            .runtime
            .array_from_u32(
                &probe_packed_words(PROBE_VALUE_OUTPUT_DIMENSION),
                &[
                    PROBE_EXPERT_COUNT as i32,
                    PROBE_VALUE_OUTPUT_DIMENSION as i32,
                    words_per_group() as i32,
                ],
            )
            .map_err(probe_execution)?;
        let scales = self
            .runtime
            .array_from_f32(&[2.0, 2.0, 3.0, 3.0], &[2, 2, 1])
            .map_err(probe_execution)?;
        let biases = self
            .runtime
            .array_from_f32(&[1.0, 1.0, 0.0, 0.0], &[2, 2, 1])
            .map_err(probe_execution)?;
        let value_experts = K2HorizonMoVAAffineLinear::new(
            packed,
            scales,
            biases,
            PROBE_BITS,
            PROBE_GROUP_SIZE,
            None,
        );
        let output = kernels
            .fused_value_expert_decode(
                self.runtime,
                &hidden,
                &indices,
                &scores,
                &value_experts,
                probe_attribution,
            )
            .map_err(|error| KernelCapabilityError::Execution {
                description: error.to_string(),
            })?;
        let output_values = output.to_vec_f32().map_err(probe_execution)?;
        let assignment_scales = [2.0_f32, 3.0];
        let assignment_biases = [1.0_f32, 0.0];
        let assignment_scores = [1.0_f32, 0.5];
        let mut expected = vec![0.0_f32; PROBE_VALUE_OUTPUT_DIMENSION];
        for (assignment_index, score) in assignment_scores.iter().enumerate() {
            for (output_index, expected_value) in expected.iter_mut().enumerate() {
                let activated = silu(probe_row_dot(
                    output_index,
                    assignment_scales[assignment_index],
                    assignment_biases[assignment_index],
                ));
                *expected_value += activated * score;
            }
        }
        validate_probe_outputs(&output_values, &expected)
    }

    fn probe_routed_ffn_kernels(
        &self,
        kernels: &FusedExpertDecodeKernels,
        probe_attribution: &mut PerformanceAttribution,
    ) -> Result<(), KernelCapabilityError> {
        let hidden = self
            .runtime
            .array_from_f32(&hidden_values(), &[1, PROBE_INPUT_DIMENSION as i32])
            .map_err(probe_execution)?;
        let indices = self
            .runtime
            .array_from_u32(&[0, 1], &[1, PROBE_ASSIGNMENT_COUNT as i32])
            .map_err(probe_execution)?;
        let scores = self
            .runtime
            .array_from_f32(&[1.0, 0.5], &[1, PROBE_ASSIGNMENT_COUNT as i32])
            .map_err(probe_execution)?;
        let fused_rows = (PROBE_INTERMEDIATE_DIMENSION * 2) as i32;
        let gate_up_packed = self
            .runtime
            .array_from_u32(
                &probe_packed_words(fused_rows as usize),
                &[
                    PROBE_EXPERT_COUNT as i32,
                    fused_rows,
                    words_per_group() as i32,
                ],
            )
            .map_err(probe_execution)?;
        let gate_up_scales = self
            .runtime
            .array_from_f32(
                &probe_scales_both_experts(fused_rows as usize),
                &[2, fused_rows, 1],
            )
            .map_err(probe_execution)?;
        let gate_up_biases = self
            .runtime
            .array_from_f32(
                &vec![1.0; PROBE_EXPERT_COUNT * fused_rows as usize],
                &[2, fused_rows, 1],
            )
            .map_err(probe_execution)?;
        let switch_gate_up = K2HorizonMoVAAffineLinear::new(
            gate_up_packed,
            gate_up_scales,
            gate_up_biases,
            PROBE_BITS,
            PROBE_GROUP_SIZE,
            None,
        );
        let down_packed = self
            .runtime
            .array_from_u32(
                &probe_packed_words(PROBE_DOWN_OUTPUT_DIMENSION),
                &[
                    PROBE_EXPERT_COUNT as i32,
                    PROBE_DOWN_OUTPUT_DIMENSION as i32,
                    words_per_group() as i32,
                ],
            )
            .map_err(probe_execution)?;
        let down_scales = self
            .runtime
            .array_from_f32(
                &probe_scales_both_experts(PROBE_DOWN_OUTPUT_DIMENSION),
                &[2, PROBE_DOWN_OUTPUT_DIMENSION as i32, 1],
            )
            .map_err(probe_execution)?;
        let down_biases = self
            .runtime
            .array_from_f32(
                &[0.0; PROBE_EXPERT_COUNT * PROBE_DOWN_OUTPUT_DIMENSION],
                &[2, PROBE_DOWN_OUTPUT_DIMENSION as i32, 1],
            )
            .map_err(probe_execution)?;
        let switch_down = K2HorizonMoVAAffineLinear::new(
            down_packed,
            down_scales,
            down_biases,
            PROBE_BITS,
            PROBE_GROUP_SIZE,
            None,
        );
        let output = kernels
            .fused_routed_ffn_decode(
                self.runtime,
                &hidden,
                &indices,
                &scores,
                &switch_gate_up,
                &switch_down,
                probe_attribution,
            )
            .map_err(|error| KernelCapabilityError::Execution {
                description: error.to_string(),
            })?;
        let output_values = output.to_vec_f32().map_err(probe_execution)?;

        // Rebuild the expected values: assignment a selects expert a, and the
        // SwiGLU hidden is silu(gate row o) * up row 64+o per assignment.
        let assignment_scores = [1.0_f32, 0.5];
        let mut expected = vec![0.0_f32; PROBE_DOWN_OUTPUT_DIMENSION];
        for score in assignment_scores.iter() {
            let mut assignment_hidden = vec![0.0_f32; PROBE_INTERMEDIATE_DIMENSION];
            for (output_index, assignment_hidden_element) in
                assignment_hidden.iter_mut().enumerate()
            {
                let gate_row = output_index;
                let up_row = PROBE_INTERMEDIATE_DIMENSION + output_index;
                let gate_dot = probe_row_dot(gate_row, row_scale(gate_row), 1.0);
                let up_dot = probe_row_dot(up_row, row_scale(up_row), 1.0);
                *assignment_hidden_element = silu(gate_dot) * up_dot;
            }
            for (output_index, expected_value) in expected.iter_mut().enumerate() {
                let dot = arbitrary_row_dot(
                    &assignment_hidden,
                    output_index,
                    row_scale(output_index),
                    0.0,
                );
                *expected_value += dot * score;
            }
        }
        validate_probe_outputs(&output_values, &expected)
    }
}

fn probe_execution(
    error: astronomical_runtime_integration::MlxRuntimeError,
) -> KernelCapabilityError {
    KernelCapabilityError::Execution {
        description: error.to_string(),
    }
}

/// The probe hidden vector: element e holds e, so every dot product is a
/// plain integer-weighted sum the host can reproduce exactly.
fn hidden_values() -> Vec<f32> {
    (0..PROBE_INPUT_DIMENSION)
        .map(|element_index| element_index as f32)
        .collect()
}

fn row_scale(row: usize) -> f32 {
    if row.is_multiple_of(2) { 2.0 } else { 3.0 }
}

/// Dots the fixed probe hidden vector with one probe weight row:
/// `w(e) = scale * ((e + row) % 16) + bias`.
fn probe_row_dot(row: usize, scale: f32, bias: f32) -> f32 {
    (0..PROBE_INPUT_DIMENSION)
        .map(|element_index| {
            let nibble = ((element_index + row) % 16) as f32;
            (element_index as f32) * (scale * nibble + bias)
        })
        .sum()
}

/// Dots an arbitrary vector with one probe weight row.
fn arbitrary_row_dot(vector: &[f32], row: usize, scale: f32, bias: f32) -> f32 {
    (0..PROBE_INPUT_DIMENSION)
        .map(|element_index| {
            let nibble = ((element_index + row) % 16) as f32;
            vector[element_index] * (scale * nibble + bias)
        })
        .sum()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn words_per_group() -> usize {
    (PROBE_GROUP_SIZE as usize) * (PROBE_BITS as usize) / 32
}

/// Builds the packed uint32 words for both probe experts and `rows` rows.
///
/// Element e of row r carries nibble `(e + r) % 16`; nibbles fill each word
/// least-significant first, matching MLX's 4-bit affine packing.
fn probe_packed_words(rows: usize) -> Vec<u32> {
    let mut packed_words = Vec::with_capacity(PROBE_EXPERT_COUNT * rows * words_per_group());
    for _expert_index in 0..PROBE_EXPERT_COUNT {
        for row in 0..rows {
            for word_index in 0..words_per_group() {
                let mut packed_word = 0_u32;
                for nibble_index in 0..8 {
                    let nibble = ((word_index * 8 + nibble_index + row) % 16) as u32;
                    packed_word |= nibble << (4 * nibble_index);
                }
                packed_words.push(packed_word);
            }
        }
    }
    packed_words
}

fn probe_scales_both_experts(rows: usize) -> Vec<f32> {
    (0..PROBE_EXPERT_COUNT)
        .flat_map(|_expert_index| (0..rows).map(row_scale))
        .collect()
}
