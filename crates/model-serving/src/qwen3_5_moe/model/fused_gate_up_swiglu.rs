use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

const FUSED_FOUR_BIT_AFFINE_GATE_UP_SWIGLU_OPERATION: &str =
    "apply fused Qwen3.5-MoE four-bit affine gate-up SwiGLU";
const FOUR_BIT_AFFINE_GROUP_SIZE: i32 = 64;
const FOUR_BIT_AFFINE_PACKED_VALUES_PER_U32: i32 = 8;
const FUSED_FOUR_BIT_AFFINE_GATE_UP_SWIGLU_SOURCE: &str = r#"
    auto thread_index = thread_position_in_threadgroup.x;
    auto output_dimension_index = thread_position_in_grid.x;

    threadgroup ActivationT staged_hidden_states[HiddenDimension];
    for (int hidden_dimension_index = thread_index;
         hidden_dimension_index < HiddenDimension;
         hidden_dimension_index += threads_per_threadgroup.x) {
        staged_hidden_states[hidden_dimension_index] = hidden_states[hidden_dimension_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (output_dimension_index >= OutputDimension) {
        return;
    }

    const device uchar* gate_weight_bytes =
        reinterpret_cast<const device uchar*>(gate_packed_weights);
    const device uchar* up_weight_bytes =
        reinterpret_cast<const device uchar*>(up_packed_weights);
    const int packed_weight_bytes_per_output = HiddenDimension / 2;
    const int gate_weight_offset =
        output_dimension_index * packed_weight_bytes_per_output;
    const int up_weight_offset =
        output_dimension_index * packed_weight_bytes_per_output;
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;

    for (int hidden_dimension_index = 0;
         hidden_dimension_index < HiddenDimension;
         ++hidden_dimension_index) {
        const int quantization_group_index =
            hidden_dimension_index / FourBitAffineGroupSize;
        const int quantization_parameter_offset =
            output_dimension_index * QuantizationGroupCount + quantization_group_index;
        const uchar gate_packed_byte =
            gate_weight_bytes[gate_weight_offset + hidden_dimension_index / 2];
        const uchar up_packed_byte =
            up_weight_bytes[up_weight_offset + hidden_dimension_index / 2];
        const uint gate_quantized_value = hidden_dimension_index % 2 == 0
            ? uint(gate_packed_byte & 0x0f)
            : uint(gate_packed_byte >> 4);
        const uint up_quantized_value = hidden_dimension_index % 2 == 0
            ? uint(up_packed_byte & 0x0f)
            : uint(up_packed_byte >> 4);
        const float hidden_value = float(staged_hidden_states[hidden_dimension_index]);
        const float gate_value =
            float(gate_scales[quantization_parameter_offset]) * float(gate_quantized_value) +
            float(gate_biases[quantization_parameter_offset]);
        const float up_value =
            float(up_scales[quantization_parameter_offset]) * float(up_quantized_value) +
            float(up_biases[quantization_parameter_offset]);
        gate_accumulator += hidden_value * gate_value;
        up_accumulator += hidden_value * up_value;
    }

    const float activated_gate = gate_accumulator / (1.0f + metal::exp(-gate_accumulator));
    activated_states[output_dimension_index] = ActivationT(activated_gate * up_accumulator);
"#;

/// Constructs the retained raw four-bit affine decode SwiGLU Metal kernel.
pub fn qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu_kernel()
-> Result<MlxMetalKernel, MlxRuntimeError> {
    MlxMetalKernel::new(
        "astronomical_qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu",
        &[
            "hidden_states",
            "gate_packed_weights",
            "gate_scales",
            "gate_biases",
            "up_packed_weights",
            "up_scales",
            "up_biases",
        ],
        &["activated_states"],
        FUSED_FOUR_BIT_AFFINE_GATE_UP_SWIGLU_SOURCE,
    )
}

/// Fuses two compatible one-token affine projections and their SwiGLU activation.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu(
    runtime: &MlxRuntime,
    fused_gate_up_swiglu_kernel: &MlxMetalKernel,
    hidden_states: &MlxArray,
    gate_packed_weights: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_packed_weights: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let (hidden_dimension, output_dimension, quantization_group_count) =
        validate_four_bit_affine_gate_up_swiglu_inputs(
            hidden_states,
            gate_packed_weights,
            gate_scales,
            gate_biases,
            up_packed_weights,
            up_scales,
            up_biases,
        )?;
    let mut kernel_outputs = runtime.apply_metal_kernel(
        fused_gate_up_swiglu_kernel,
        &[
            hidden_states,
            gate_packed_weights,
            gate_scales,
            gate_biases,
            up_packed_weights,
            up_scales,
            up_biases,
        ],
        &[MlxMetalKernelOutput::new(
            vec![1, 1, output_dimension],
            hidden_states.dtype(),
        )],
        [output_dimension, 1, 1],
        [output_dimension.min(256), 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "ActivationT",
                dtype: hidden_states.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "HiddenDimension",
                integer_template_argument: hidden_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "OutputDimension",
                integer_template_argument: output_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "FourBitAffineGroupSize",
                integer_template_argument: FOUR_BIT_AFFINE_GROUP_SIZE,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "QuantizationGroupCount",
                integer_template_argument: quantization_group_count,
            },
        ],
    )?;
    kernel_outputs
        .pop()
        .ok_or_else(|| fused_gate_up_swiglu_error("kernel returned no output"))
}

#[allow(clippy::too_many_arguments)]
fn validate_four_bit_affine_gate_up_swiglu_inputs(
    hidden_states: &MlxArray,
    gate_packed_weights: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_packed_weights: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
) -> Result<(i32, i32, i32), MlxRuntimeError> {
    let hidden_state_shape = hidden_states.shape();
    let gate_packed_weight_shape = gate_packed_weights.shape();
    let gate_scale_shape = gate_scales.shape();
    if hidden_state_shape.len() != 3
        || hidden_state_shape[0] != 1
        || hidden_state_shape[1] != 1
        || gate_packed_weight_shape.len() != 2
        || gate_scale_shape.len() != 2
        || gate_scale_shape != gate_biases.shape()
        || gate_packed_weight_shape != up_packed_weights.shape()
        || gate_scale_shape != up_scales.shape()
        || gate_scale_shape != up_biases.shape()
    {
        return Err(fused_gate_up_swiglu_error(
            "inputs must be one-token hidden states with matching affine tensor layouts",
        ));
    }
    let hidden_dimension = hidden_state_shape[2];
    let output_dimension = gate_packed_weight_shape[0];
    let packed_hidden_dimension = gate_packed_weight_shape[1];
    if hidden_dimension <= 0
        || output_dimension <= 0
        || hidden_dimension % FOUR_BIT_AFFINE_GROUP_SIZE != 0
        || packed_hidden_dimension != hidden_dimension / FOUR_BIT_AFFINE_PACKED_VALUES_PER_U32
    {
        return Err(fused_gate_up_swiglu_error(
            "four-bit affine dimensions are incompatible with the decode kernel",
        ));
    }
    let quantization_group_count = hidden_dimension / FOUR_BIT_AFFINE_GROUP_SIZE;
    if gate_scale_shape != [output_dimension, quantization_group_count]
        || hidden_states.dtype() != MlxDtype::BFloat16
        || gate_packed_weights.dtype() != MlxDtype::UInt32
        || gate_scales.dtype() != MlxDtype::BFloat16
        || gate_biases.dtype() != MlxDtype::BFloat16
        || up_scales.dtype() != MlxDtype::BFloat16
        || up_biases.dtype() != MlxDtype::BFloat16
    {
        return Err(fused_gate_up_swiglu_error(
            "the decode kernel requires bfloat16 activations and affine parameters with uint32 packed weights",
        ));
    }
    Ok((hidden_dimension, output_dimension, quantization_group_count))
}

fn fused_gate_up_swiglu_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: FUSED_FOUR_BIT_AFFINE_GATE_UP_SWIGLU_OPERATION,
        description: description.to_owned(),
    }
}
