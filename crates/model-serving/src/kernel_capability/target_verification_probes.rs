//! Bounded capability probes for the two target-verification Metal kernels.
//!
//! Each probe compiles its kernel, executes the smallest eligible projection
//! geometry, and validates the output against the documented reference: the
//! repeated one-token MLX quantized projection. The optimized route must match
//! the reference exactly, matching the retained direct-MLX contract; the
//! four-row split-K route changes reduction topology, so its documented
//! contract is token-local argmax alignment.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use crate::qwen3_5::{
    Qwen3_5TargetVerificationProjectionDispatch, four_row_split_k_quantized_linear_kernel,
    qwen3_5_target_verification_quantized_linear, target_verification_quantized_linear_kernel,
};

struct ProbeProjectionInputs {
    activations: MlxArray,
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
}

/// Builds the smallest deterministic eligible geometry: two verification
/// tokens, one 512-wide input, eight outputs, BFloat16 activations, 4-bit
/// affine weights with 32-wide groups.
fn optimized_probe_inputs(runtime: &MlxRuntime) -> Result<ProbeProjectionInputs, MlxRuntimeError> {
    projection_inputs(runtime, 2, 512, 8, MlxDtype::BFloat16, 4, 32)
}

/// Builds the smallest four-row-eligible geometry: exactly four verification
/// tokens, a 512-wide input, twenty-four outputs, 4-bit affine weights with
/// 64-wide groups.
fn four_row_probe_inputs(runtime: &MlxRuntime) -> Result<ProbeProjectionInputs, MlxRuntimeError> {
    projection_inputs(runtime, 4, 512, 24, MlxDtype::BFloat16, 4, 64)
}

fn projection_inputs(
    runtime: &MlxRuntime,
    token_count: i32,
    input_dimension: i32,
    output_dimension: i32,
    activation_dtype: MlxDtype,
    quantization_bits: i32,
    quantization_group_size: i32,
) -> Result<ProbeProjectionInputs, MlxRuntimeError> {
    let activation_element_count = (token_count * input_dimension) as usize;
    let activation_values = (0..activation_element_count)
        .map(|activation_index| ((activation_index % 29) as f32 - 14.0) / 16.0)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, token_count, input_dimension])
        .and_then(|array| runtime.astype(&array, activation_dtype))?;

    let weight_element_count = (output_dimension * input_dimension) as usize;
    let weight_values = (0..weight_element_count)
        .map(|weight_index| ((weight_index % 31) as f32 - 15.0) / 32.0)
        .collect::<Vec<_>>();
    let weights = runtime
        .array_from_f32(&weight_values, &[output_dimension, input_dimension])
        .and_then(|array| runtime.astype(&array, activation_dtype))?;
    let (packed_weight, quantization_scales, quantization_biases) =
        runtime.quantize_affine(&weights, quantization_group_size, quantization_bits)?;
    Ok(ProbeProjectionInputs {
        activations,
        packed_weight,
        quantization_scales,
        quantization_biases,
    })
}

/// The documented reference: repeated one-token MLX quantized projection,
/// identical to ordinary decode arithmetic.
fn repeated_one_token_reference(
    runtime: &MlxRuntime,
    inputs: &ProbeProjectionInputs,
    token_count: i32,
    input_dimension: i32,
    quantization_group_size: i32,
    quantization_bits: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut token_projection_outputs = Vec::with_capacity(token_count as usize);
    for token_position_index in 0..token_count {
        let token_activations = runtime.slice(
            &inputs.activations,
            &[0, token_position_index, 0],
            &[1, token_position_index + 1, input_dimension],
            &[1, 1, 1],
        )?;
        token_projection_outputs.push(runtime.quantized_matmul_affine(
            &token_activations,
            &inputs.packed_weight,
            &inputs.quantization_scales,
            &inputs.quantization_biases,
            true,
            quantization_group_size,
            quantization_bits,
        )?);
    }
    let token_projection_output_references = token_projection_outputs.iter().collect::<Vec<_>>();
    runtime.concatenate_axis(&token_projection_output_references, 1)
}

fn float32_values(
    runtime: &MlxRuntime,
    projected_activations: &MlxArray,
) -> Result<Vec<f32>, MlxRuntimeError> {
    runtime
        .astype(projected_activations, MlxDtype::Float32)
        .and_then(|float32_activations| float32_activations.to_vec_f32())
}

pub struct TargetVerificationProjectionProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> TargetVerificationProjectionProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for TargetVerificationProjectionProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::TargetVerificationQuantizedLinear
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = target_verification_quantized_linear_kernel().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let inputs = optimized_probe_inputs(self.runtime).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        let projection = qwen3_5_target_verification_quantized_linear(
            self.runtime,
            Some(&kernel),
            None,
            &inputs.activations,
            &inputs.packed_weight,
            &inputs.quantization_scales,
            &inputs.quantization_biases,
            32,
            4,
        )
        .map_err(|error| KernelCapabilityError::Execution {
            description: error.to_string(),
        })?;
        if projection.dispatch() != Qwen3_5TargetVerificationProjectionDispatch::OptimizedMetal {
            return Err(KernelCapabilityError::Execution {
                description: "the probe geometry did not select the optimized Metal kernel route"
                    .to_owned(),
            });
        }
        let reference = repeated_one_token_reference(self.runtime, &inputs, 2, 512, 32, 4)
            .map_err(|error| KernelCapabilityError::Execution {
                description: error.to_string(),
            })?;
        let probe_output = float32_values(self.runtime, &projection.into_projected_activations())
            .map_err(|error| KernelCapabilityError::Execution {
            description: error.to_string(),
        })?;
        let reference_output = float32_values(self.runtime, &reference).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        validate_probe_outputs(&probe_output, &reference_output)
    }
}

fn argmax(row_values: &[f32]) -> usize {
    row_values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(row_index, _)| row_index)
        .unwrap_or(0)
}

pub struct TargetVerificationFourRowProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> TargetVerificationFourRowProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for TargetVerificationFourRowProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::TargetVerificationFourRowQuantizedLinear
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = four_row_split_k_quantized_linear_kernel().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let inputs = four_row_probe_inputs(self.runtime).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        let projection = qwen3_5_target_verification_quantized_linear(
            self.runtime,
            None,
            Some(&kernel),
            &inputs.activations,
            &inputs.packed_weight,
            &inputs.quantization_scales,
            &inputs.quantization_biases,
            64,
            4,
        )
        .map_err(|error| KernelCapabilityError::Execution {
            description: error.to_string(),
        })?;
        if projection.dispatch() != Qwen3_5TargetVerificationProjectionDispatch::FourRowSplitK {
            return Err(KernelCapabilityError::Execution {
                description: "the probe geometry did not select the four-row split-K kernel"
                    .to_owned(),
            });
        }
        let reference = repeated_one_token_reference(self.runtime, &inputs, 4, 512, 64, 4)
            .map_err(|error| KernelCapabilityError::Execution {
                description: error.to_string(),
            })?;
        let probe_output = float32_values(self.runtime, &projection.into_projected_activations())
            .map_err(|error| KernelCapabilityError::Execution {
            description: error.to_string(),
        })?;
        let reference_output = float32_values(self.runtime, &reference).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        if probe_output.len() != reference_output.len() {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "four-row probe produced {} values but the reference produced {}",
                    probe_output.len(),
                    reference_output.len()
                ),
            });
        }
        let output_dimension = 24_usize;
        for token_position_index in 0..4_usize {
            let row_start = token_position_index * output_dimension;
            let row_end = row_start + output_dimension;
            if argmax(&probe_output[row_start..row_end])
                != argmax(&reference_output[row_start..row_end])
            {
                return Err(KernelCapabilityError::OutputMismatch {
                    description: format!(
                        "four-row probe argmax at token {token_position_index} disagrees with the token-local reference"
                    ),
                });
            }
        }
        Ok(())
    }
}
