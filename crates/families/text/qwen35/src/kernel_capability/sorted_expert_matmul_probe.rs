//! Bounded capability probe for the sorted-expert 4-bit matmul.
//!
//! The kernel needs the M5 GPU's matrix units (Metal Performance
//! Primitives); MLX compiles its source only when it first runs, so only an
//! evaluation proves a chip can use it. The probe projects a small routed
//! layer two ways, the token-row kernel and MLX's own `gather_qmm` on the
//! sorted copy of the rows, and asks for the same values to the bit: the
//! kernel is built to give MLX's bits, and a chip or MLX version on which it
//! does not keeps MLX's route.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use crate::sorted_expert_matmul::{
    ExpertStack, SortedExpertMatmulKernels, sorted_expert_quantized_matmul_from_token_rows,
};

/// Registry identifier of the sorted-expert 4-bit matmul.
pub const SORTED_EXPERT_MATMUL: CustomMetalKernelFamily =
    CustomMetalKernelFamily::new("sorted_expert_matmul");

pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(SortedExpertMatmulProbe { runtime })]
}

const PROBE_EXPERT_COUNT: usize = 4;
const PROBE_OUTPUT_WIDTH: usize = 128;
const PROBE_INPUT_WIDTH: usize = 128;
const PROBE_TOKEN_COUNT: usize = 24;
const PROBE_EXPERTS_A_TOKEN: usize = 2;
const PROBE_GROUP_SIZE: i32 = 64;

struct SortedExpertMatmulProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl CustomMetalKernelProbe for SortedExpertMatmulProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        SORTED_EXPERT_MATMUL
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernels = SortedExpertMatmulKernels::declare().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let (kernel_values, mlx_values) =
            run_probe_projections(self.runtime, &kernels).map_err(|error| {
                KernelCapabilityError::Execution {
                    description: error.to_string(),
                }
            })?;
        // The registry's check allows a relative 1e-5; this kernel promises
        // MLX's exact bits, so anything short of equality is a mismatch.
        if let Some(value_index) = kernel_values
            .iter()
            .zip(&mlx_values)
            .position(|(kernel_value, mlx_value)| kernel_value.to_bits() != mlx_value.to_bits())
        {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "probe output value {value_index} read {} but MLX gave {}",
                    kernel_values[value_index], mlx_values[value_index]
                ),
            });
        }
        validate_probe_outputs(&kernel_values, &mlx_values)
    }
}

/// A fixed, non-trivial value at `index`: small integers over 16, so every
/// weight and row value is exact in bf16 and varied enough to catch a
/// misplaced row, column or expert.
fn probe_value(index: usize, stride: usize) -> f32 {
    ((index * stride) % 31) as f32 / 16.0 - 0.9
}

/// (kernel output, MLX output) for one routed projection, as float32.
fn run_probe_projections(
    runtime: &MlxRuntime,
    kernels: &SortedExpertMatmulKernels,
) -> Result<(Vec<f32>, Vec<f32>), MlxRuntimeError> {
    let weight_values: Vec<f32> = (0..PROBE_EXPERT_COUNT * PROBE_OUTPUT_WIDTH * PROBE_INPUT_WIDTH)
        .map(|index| probe_value(index, 7))
        .collect();
    let weights = runtime.astype(
        &runtime.array_from_f32(
            &weight_values,
            &[
                PROBE_EXPERT_COUNT as i32,
                PROBE_OUTPUT_WIDTH as i32,
                PROBE_INPUT_WIDTH as i32,
            ],
        )?,
        MlxDtype::BFloat16,
    )?;
    let (packed_weight, scales, biases) = runtime.quantize_affine(&weights, PROBE_GROUP_SIZE, 4)?;
    let token_values: Vec<f32> = (0..PROBE_TOKEN_COUNT * PROBE_INPUT_WIDTH)
        .map(|index| probe_value(index, 11))
        .collect();
    let token_rows = runtime.astype(
        &runtime.array_from_f32(
            &token_values,
            &[PROBE_TOKEN_COUNT as i32, 1, PROBE_INPUT_WIDTH as i32],
        )?,
        MlxDtype::BFloat16,
    )?;
    let (sorted_order, sorted_tokens, sorted_experts) = probe_routing(runtime)?;
    let experts = ExpertStack {
        packed_weight: &packed_weight,
        scales: &scales,
        biases: &biases,
    };
    let kernel_output = sorted_expert_quantized_matmul_from_token_rows(
        runtime,
        kernels,
        (&token_rows, &sorted_order, &sorted_experts),
        PROBE_EXPERTS_A_TOKEN as i32,
        &experts,
    )
    .map_err(|projection_error| MlxRuntimeError::RuntimeOperation {
        operation: "run the sorted-expert matmul probe",
        description: projection_error.to_string(),
    })?;
    let sorted_rows = runtime.take_axis(&token_rows, &sorted_tokens, 0)?;
    let mlx_output = runtime.gather_quantized_matmul_affine(
        &sorted_rows,
        &packed_weight,
        &scales,
        &biases,
        None,
        Some(&sorted_experts),
        true,
        PROBE_GROUP_SIZE,
        4,
        true,
    )?;
    Ok((
        float_values(runtime, &kernel_output)?,
        float_values(runtime, &mlx_output)?,
    ))
}

/// Every token's two experts, uneven over the four, sorted stably as the
/// engine's assignment sort does: (sorted order, tokens, experts).
fn probe_routing(runtime: &MlxRuntime) -> Result<(MlxArray, MlxArray, MlxArray), MlxRuntimeError> {
    let mut flat_experts = Vec::with_capacity(PROBE_TOKEN_COUNT * PROBE_EXPERTS_A_TOKEN);
    for token in 0..PROBE_TOKEN_COUNT {
        let first_expert = (token * token + 1) % PROBE_EXPERT_COUNT;
        flat_experts.push(first_expert as u32);
        flat_experts.push(((first_expert + 1 + token % 2) % PROBE_EXPERT_COUNT) as u32);
    }
    let mut sorted_order: Vec<u32> = (0..flat_experts.len() as u32).collect();
    sorted_order.sort_by_key(|assignment| flat_experts[*assignment as usize]);
    let sorted_tokens: Vec<u32> = sorted_order
        .iter()
        .map(|assignment| assignment / PROBE_EXPERTS_A_TOKEN as u32)
        .collect();
    let sorted_experts: Vec<u32> = sorted_order
        .iter()
        .map(|assignment| flat_experts[*assignment as usize])
        .collect();
    let shape = [sorted_order.len() as i32];
    Ok((
        runtime.array_from_u32(&sorted_order, &shape)?,
        runtime.array_from_u32(&sorted_tokens, &shape)?,
        runtime.array_from_u32(&sorted_experts, &shape)?,
    ))
}

fn float_values(runtime: &MlxRuntime, projected: &MlxArray) -> Result<Vec<f32>, MlxRuntimeError> {
    Ok(runtime.array_to_vec_f32(&runtime.astype(projected, MlxDtype::Float32)?)?)
}

#[cfg(test)]
#[path = "sorted_expert_matmul_probe_tests.rs"]
mod sorted_expert_matmul_probe_tests;
