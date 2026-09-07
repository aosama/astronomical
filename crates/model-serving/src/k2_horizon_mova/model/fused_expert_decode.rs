//! Fused single-token expert decode kernels for K2 Horizon MoVA.
//!
//! At decode one token selects a handful of experts, so the sparse chain is
//! dispatch-bound: the gathered path costs several Metal launches per layer
//! (gathered quantized matmuls, splits, SwiGLU, weighted reduction). These
//! custom kernels collapse each chain into one or two launches that read the
//! packed 4-bit expert weights straight from global memory, dequantize each
//! nibble group inside the dot product, and apply the activation and the
//! router-weighted reduction inside the same thread.
//!
//! The kernels deliberately avoid threadgroup scratch memory because the MLX
//! custom-kernel launcher never sets threadgroup memory lengths; every partial
//! result stays thread-local, and cross-assignment accumulation happens through
//! one atomic float buffer the launcher initializes to zero.
//!
//! All inputs are indexed linearly from the base pointer, so callers may pass
//! any row-contiguous shape whose elements follow the documented layout; MLX
//! copies non-contiguous inputs at launch.
//!
//! Decode-only by contract: callers route single-token batches here and keep
//! chunked prefill on the standard gathered path.

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

use crate::PerformanceAttribution;

use super::affine::K2HorizonMoVAAffineLinear;
use super::error::K2HorizonMoVAExecutionError;

const QUANTIZED_DECODE_GROUP_SIZE: i32 = 64;
const QUANTIZED_DECODE_BITS: i32 = 4;
const THREADGROUP_SIZE: i32 = 256;

fn words_per_group() -> i32 {
    QUANTIZED_DECODE_GROUP_SIZE * QUANTIZED_DECODE_BITS / 32
}

/// Shared Metal header: the atomic include plus the dequantizing dot product.
///
/// MLX packs eight nibbles per `uint32` least-significant-byte first, so
/// element `j` within a word is the low nibble of byte `j / 2` for even `j`
/// and the high nibble for odd `j`. The affine decode is
/// `scale * nibble + bias`, matching MLX's `dequantize` quantization helper.
const FUSED_DECODE_KERNEL_HEADER: &str = r#"
#include <metal_atomic>
using namespace metal;

template <typename ScaleRowT, typename HiddenT>
METAL_FUNC float quantized_row_dot(
    const device uint32_t* packed_row,
    ScaleRowT scale_row,
    ScaleRowT bias_row,
    const device HiddenT* hidden,
    int groups_per_output,
    int group_size,
    int words_per_group) {
  float dot = 0.0f;
  for (int group_index = 0; group_index < groups_per_output; ++group_index) {
    float scale = float(scale_row[group_index]);
    float bias = float(bias_row[group_index]);
    const device uint32_t* packed_group = packed_row + group_index * words_per_group;
    int element_base = group_index * group_size;
    for (int word_index = 0; word_index < words_per_group; ++word_index) {
      uint32_t packed_word = packed_group[word_index];
      int element_base_word = element_base + word_index * 8;
      for (int nibble_index = 0; nibble_index < 8; ++nibble_index) {
        uint32_t byte_value = (packed_word >> (8u * (nibble_index / 2u))) & 0xFFu;
        uint32_t nibble_value =
            ((nibble_index % 2u) == 0u) ? (byte_value & 0xFu) : (byte_value >> 4u);
        dot += (scale * float(nibble_value) + bias) *
            float(hidden[element_base_word + nibble_index]);
      }
    }
  }
  return dot;
}
"#;

const VALUE_EXPERT_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / out_dimension;
  auto output_index = global_index % out_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto weight_row_offset = (expert_index * out_dimension + output_index) * words_per_output;
  auto scale_row_offset = (expert_index * out_dimension + output_index) * groups_per_output;
  float dot = quantized_row_dot(
      packed_weights + weight_row_offset,
      scales + scale_row_offset,
      biases + scale_row_offset,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float activated = dot / (1.0f + metal::exp(-dot));
  float contribution = activated * float(scores[assignment_index]);
  metal::atomic_fetch_add_explicit(
      &weighted_outputs[output_index], contribution, metal::memory_order_relaxed);
"#;

const ROUTED_GATE_UP_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / intermediate_dimension;
  auto output_index = global_index % intermediate_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto gate_row = expert_index * fused_dimension + output_index;
  auto up_row = gate_row + intermediate_dimension;
  float gate_dot = quantized_row_dot(
      packed_weights + gate_row * words_per_output,
      scales + gate_row * groups_per_output,
      biases + gate_row * groups_per_output,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float up_dot = quantized_row_dot(
      packed_weights + up_row * words_per_output,
      scales + up_row * groups_per_output,
      biases + up_row * groups_per_output,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float gate_activated = gate_dot / (1.0f + metal::exp(-gate_dot));
  swiglu_hidden[global_index] = (OutputT)(gate_activated * up_dot);
"#;

const ROUTED_DOWN_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / out_dimension;
  auto output_index = global_index % out_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto weight_row_offset = (expert_index * out_dimension + output_index) * words_per_output;
  auto scale_row_offset = (expert_index * out_dimension + output_index) * groups_per_output;
  float dot = quantized_row_dot(
      packed_weights + weight_row_offset,
      scales + scale_row_offset,
      biases + scale_row_offset,
      swiglu_hidden + assignment_index * intermediate_dimension,
      groups_per_output,
      group_size,
      words_per_group);
  float contribution = dot * float(scores[assignment_index]);
  metal::atomic_fetch_add_explicit(
      &weighted_outputs[output_index], contribution, metal::memory_order_relaxed);
"#;

/// Compiled owners of the three fused decode kernels.
pub struct FusedExpertDecodeKernels {
    value_experts: MlxMetalKernel,
    routed_gate_up: MlxMetalKernel,
    routed_down: MlxMetalKernel,
}

impl FusedExpertDecodeKernels {
    /// Compiles the fused decode kernels once per worker process.
    ///
    /// # Errors
    /// Returns an error when any kernel source fails to compile.
    pub fn new() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            value_experts: MlxMetalKernel::new_with_options(
                "fused_value_expert_decode",
                &[
                    "hidden",
                    "expert_indices",
                    "scores",
                    "packed_weights",
                    "scales",
                    "biases",
                ],
                &["weighted_outputs"],
                FUSED_DECODE_KERNEL_HEADER,
                VALUE_EXPERT_KERNEL_SOURCE,
                true,
            )?,
            routed_gate_up: MlxMetalKernel::new_with_options(
                "fused_routed_gate_up_decode",
                &[
                    "hidden",
                    "expert_indices",
                    "packed_weights",
                    "scales",
                    "biases",
                ],
                &["swiglu_hidden"],
                FUSED_DECODE_KERNEL_HEADER,
                ROUTED_GATE_UP_KERNEL_SOURCE,
                false,
            )?,
            routed_down: MlxMetalKernel::new_with_options(
                "fused_routed_down_decode",
                &[
                    "swiglu_hidden",
                    "expert_indices",
                    "scores",
                    "packed_weights",
                    "scales",
                    "biases",
                ],
                &["weighted_outputs"],
                FUSED_DECODE_KERNEL_HEADER,
                ROUTED_DOWN_KERNEL_SOURCE,
                true,
            )?,
        })
    }

    /// Runs the fused MoVA value-expert decode: quantized row dots, silu
    /// activation, and the router-weighted reduction in one launch.
    ///
    /// Returns `[1, output_dimension]` in the activation dtype.
    pub fn fused_value_expert_decode(
        &self,
        runtime: &MlxRuntime,
        flat_hidden: &MlxArray,
        routed_indices: &MlxArray,
        routed_scores: &MlxArray,
        value_experts: &K2HorizonMoVAAffineLinear,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeFusedValueExpertDecode,
            |_| {
                self.value_expert_inner(
                    runtime,
                    flat_hidden,
                    routed_indices,
                    routed_scores,
                    value_experts,
                )
            },
        )
    }

    /// Runs the fused routed FFN decode: one gate/up/SwiGLU launch and one
    /// score-weighted reduction launch.
    ///
    /// Returns `[1, hidden_size]` in the activation dtype.
    pub fn fused_routed_ffn_decode(
        &self,
        runtime: &MlxRuntime,
        flat_hidden: &MlxArray,
        routed_indices: &MlxArray,
        routed_scores: &MlxArray,
        switch_gate_up: &K2HorizonMoVAAffineLinear,
        switch_down: &K2HorizonMoVAAffineLinear,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeFusedRoutedExpertDecode,
            |_| {
                self.routed_ffn_inner(
                    runtime,
                    flat_hidden,
                    routed_indices,
                    routed_scores,
                    switch_gate_up,
                    switch_down,
                )
            },
        )
    }

    fn value_expert_inner(
        &self,
        runtime: &MlxRuntime,
        flat_hidden: &MlxArray,
        routed_indices: &MlxArray,
        routed_scores: &MlxArray,
        value_experts: &K2HorizonMoVAAffineLinear,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        let row_geometry =
            validate_expert_projection(value_experts, "the fused value-expert projection")?;
        let output_dimension = row_geometry.rows;
        let launch_geometry = validate_decode_launch_inputs(
            flat_hidden,
            routed_indices,
            routed_scores,
            row_geometry.input_dimension,
        )?;
        let weighted_outputs = apply_fused_reduce_kernel(
            runtime,
            &self.value_experts,
            &[
                flat_hidden,
                routed_indices,
                routed_scores,
                value_experts.packed_weight(),
                value_experts.scales(),
                value_experts.biases(),
            ],
            Vec::new(),
            output_dimension,
            &row_geometry,
            &launch_geometry,
        )?;
        Ok(runtime.reshape(
            &runtime.astype(&weighted_outputs, flat_hidden.dtype())?,
            &[1, output_dimension],
        )?)
    }

    fn routed_ffn_inner(
        &self,
        runtime: &MlxRuntime,
        flat_hidden: &MlxArray,
        routed_indices: &MlxArray,
        routed_scores: &MlxArray,
        switch_gate_up: &K2HorizonMoVAAffineLinear,
        switch_down: &K2HorizonMoVAAffineLinear,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        let gate_up_geometry =
            validate_expert_projection(switch_gate_up, "the fused gate/up projection")?;
        if gate_up_geometry.rows % 2 != 0 {
            return Err(invalid_geometry(
                "the fused gate/up projection must hold an even row count",
            ));
        }
        let intermediate_dimension = gate_up_geometry.rows / 2;
        let down_geometry = validate_expert_projection(switch_down, "the fused down projection")?;
        if down_geometry.input_dimension != intermediate_dimension {
            return Err(invalid_geometry(
                "the fused gate/up and down projections disagree on the intermediate size",
            ));
        }
        let launch_geometry = validate_decode_launch_inputs(
            flat_hidden,
            routed_indices,
            routed_scores,
            gate_up_geometry.input_dimension,
        )?;
        let mut gate_up_arguments = vec![MlxMetalKernelTemplateArgument::Dtype {
            name: "OutputT",
            dtype: flat_hidden.dtype(),
        }];
        gate_up_arguments.extend(row_template_arguments(&gate_up_geometry));
        gate_up_arguments.push(MlxMetalKernelTemplateArgument::Integer {
            name: "intermediate_dimension",
            integer_template_argument: intermediate_dimension,
        });
        gate_up_arguments.push(MlxMetalKernelTemplateArgument::Integer {
            name: "fused_dimension",
            integer_template_argument: gate_up_geometry.rows,
        });
        let swiglu_hidden = runtime
            .apply_metal_kernel(
                &self.routed_gate_up,
                &[
                    flat_hidden,
                    routed_indices,
                    switch_gate_up.packed_weight(),
                    switch_gate_up.scales(),
                    switch_gate_up.biases(),
                ],
                &[MlxMetalKernelOutput::new(
                    vec![launch_geometry.assignment_count, intermediate_dimension],
                    flat_hidden.dtype(),
                )],
                [
                    launch_geometry.assignment_count * intermediate_dimension,
                    1,
                    1,
                ],
                [THREADGROUP_SIZE, 1, 1],
                &gate_up_arguments,
            )
            .map_err(execution_error)?;
        let mut swiglu_outputs = swiglu_hidden;
        let swiglu_hidden =
            swiglu_outputs
                .pop()
                .ok_or_else(|| K2HorizonMoVAExecutionError::InvalidExecution {
                    description: "the fused gate/up kernel returned no SwiGLU hidden".to_owned(),
                })?;
        let down_arguments = vec![MlxMetalKernelTemplateArgument::Integer {
            name: "intermediate_dimension",
            integer_template_argument: intermediate_dimension,
        }];
        let weighted_outputs = apply_fused_reduce_kernel(
            runtime,
            &self.routed_down,
            &[
                &swiglu_hidden,
                routed_indices,
                routed_scores,
                switch_down.packed_weight(),
                switch_down.scales(),
                switch_down.biases(),
            ],
            down_arguments,
            down_geometry.rows,
            &down_geometry,
            &launch_geometry,
        )?;
        Ok(runtime.reshape(
            &runtime.astype(&weighted_outputs, flat_hidden.dtype())?,
            &[1, down_geometry.rows],
        )?)
    }
}

/// Shape facts one stacked 4-bit affine expert projection must satisfy.
struct FusedExpertRowGeometry {
    rows: i32,
    input_dimension: i32,
    groups_per_output: i32,
    words_per_output: i32,
}

struct FusedDecodeLaunchGeometry {
    assignment_count: i32,
}

fn row_template_arguments(
    row_geometry: &FusedExpertRowGeometry,
) -> Vec<MlxMetalKernelTemplateArgument> {
    vec![
        MlxMetalKernelTemplateArgument::Integer {
            name: "groups_per_output",
            integer_template_argument: row_geometry.groups_per_output,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "group_size",
            integer_template_argument: QUANTIZED_DECODE_GROUP_SIZE,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "words_per_group",
            integer_template_argument: words_per_group(),
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "words_per_output",
            integer_template_argument: row_geometry.words_per_output,
        },
    ]
}

fn validate_expert_projection(
    linear: &K2HorizonMoVAAffineLinear,
    label: &str,
) -> Result<FusedExpertRowGeometry, K2HorizonMoVAExecutionError> {
    if linear.bits() != QUANTIZED_DECODE_BITS || linear.group_size() != QUANTIZED_DECODE_GROUP_SIZE
    {
        return Err(invalid_geometry(
            "the fused decode kernels support exactly 4-bit group-64 affine projections",
        ));
    }
    let packed_shape = linear.packed_weight().shape();
    let scales_shape = linear.scales().shape();
    if packed_shape.len() != 3 || scales_shape != linear.biases().shape() {
        return Err(invalid_geometry(&format!(
            "{label} must hold stacked [experts, rows, packed] weights with matching scales and biases"
        )));
    }
    let words_per_output = packed_shape[2];
    let input_dimension = words_per_output * 32 / QUANTIZED_DECODE_BITS;
    let groups_per_output = input_dimension / QUANTIZED_DECODE_GROUP_SIZE;
    if input_dimension <= 0
        || input_dimension % QUANTIZED_DECODE_GROUP_SIZE != 0
        || scales_shape != vec![packed_shape[0], packed_shape[1], groups_per_output]
    {
        return Err(invalid_geometry(&format!(
            "{label} expects 4-bit group-64 affine scales matching the packed weights"
        )));
    }
    Ok(FusedExpertRowGeometry {
        rows: packed_shape[1],
        input_dimension,
        groups_per_output,
        words_per_output,
    })
}

fn validate_decode_launch_inputs(
    flat_hidden: &MlxArray,
    routed_indices: &MlxArray,
    routed_scores: &MlxArray,
    expected_hidden_dimension: i32,
) -> Result<FusedDecodeLaunchGeometry, K2HorizonMoVAExecutionError> {
    let hidden_shape = flat_hidden.shape();
    let hidden_dimension = hidden_shape
        .last()
        .copied()
        .ok_or_else(|| invalid_geometry("flat hidden must have an axis"))?;
    if hidden_shape.len() != 2
        || hidden_shape[0] != 1
        || hidden_dimension != expected_hidden_dimension
    {
        return Err(invalid_geometry(
            "the fused decode consumes exactly one token whose width matches the projection input",
        ));
    }
    if routed_indices.shape() != routed_scores.shape() || routed_indices.element_count() == 0 {
        return Err(invalid_geometry(
            "routed indices and scores must be nonempty and identically shaped",
        ));
    }
    if !matches!(routed_indices.dtype(), MlxDtype::Int32 | MlxDtype::UInt32) {
        return Err(invalid_geometry(
            "routed expert indices must have an integer dtype",
        ));
    }
    if routed_scores.dtype() != flat_hidden.dtype() {
        return Err(invalid_geometry(
            "routed scores must share the activation dtype",
        ));
    }
    if !matches!(
        flat_hidden.dtype(),
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    ) {
        return Err(invalid_geometry(
            "the fused decode kernels support half, bfloat, and float activations",
        ));
    }
    Ok(FusedDecodeLaunchGeometry {
        assignment_count: routed_indices.element_count() as i32,
    })
}

fn apply_fused_reduce_kernel(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    inputs: &[&MlxArray],
    extra_arguments: Vec<MlxMetalKernelTemplateArgument>,
    output_dimension: i32,
    row_geometry: &FusedExpertRowGeometry,
    launch_geometry: &FusedDecodeLaunchGeometry,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let mut arguments = row_template_arguments(row_geometry);
    arguments.extend(extra_arguments);
    arguments.push(MlxMetalKernelTemplateArgument::Integer {
        name: "out_dimension",
        integer_template_argument: output_dimension,
    });
    let mut outputs = runtime
        .apply_metal_kernel_with_output_initialization(
            kernel,
            inputs,
            &[MlxMetalKernelOutput::new(
                vec![output_dimension],
                MlxDtype::Float32,
            )],
            [launch_geometry.assignment_count * output_dimension, 1, 1],
            [THREADGROUP_SIZE, 1, 1],
            &arguments,
            Some(0.0),
        )
        .map_err(execution_error)?;
    outputs
        .pop()
        .ok_or_else(|| K2HorizonMoVAExecutionError::InvalidExecution {
            description: "the fused decode kernel returned no weighted output".to_owned(),
        })
}

fn invalid_geometry(description: &str) -> K2HorizonMoVAExecutionError {
    K2HorizonMoVAExecutionError::InvalidExecution {
        description: description.to_owned(),
    }
}

fn execution_error(error: MlxRuntimeError) -> K2HorizonMoVAExecutionError {
    K2HorizonMoVAExecutionError::InvalidExecution {
        description: format!("the fused expert decode failed: {error}"),
    }
}
