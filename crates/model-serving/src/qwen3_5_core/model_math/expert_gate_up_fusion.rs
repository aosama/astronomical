use astronomical_runtime_integration::MlxRuntime;

use super::decoder_layer_weights::Qwen3_5AffineWeights;
use super::error::Qwen3_5ExecutionError;

pub(crate) fn fuse_compatible_expert_gate_up_projections(
    runtime: &MlxRuntime,
    gate_projection: Qwen3_5AffineWeights,
    up_projection: Qwen3_5AffineWeights,
) -> Result<Qwen3_5AffineWeights, Qwen3_5ExecutionError> {
    match (gate_projection, up_projection) {
        (
            Qwen3_5AffineWeights::NativeBfloat16 {
                weight: gate_weight,
            },
            Qwen3_5AffineWeights::NativeBfloat16 { weight: up_weight },
        ) => Ok(Qwen3_5AffineWeights::NativeBfloat16 {
            weight: runtime.concatenate_axis(&[&gate_weight, &up_weight], 1)?,
        }),
        (
            Qwen3_5AffineWeights::Quantized {
                packed_weight: gate_packed_weight,
                quantization_scales: gate_quantization_scales,
                quantization_biases: gate_quantization_biases,
                quantization_bits: gate_quantization_bits,
                quantization_group_size: gate_quantization_group_size,
            },
            Qwen3_5AffineWeights::Quantized {
                packed_weight: up_packed_weight,
                quantization_scales: up_quantization_scales,
                quantization_biases: up_quantization_biases,
                quantization_bits: up_quantization_bits,
                quantization_group_size: up_quantization_group_size,
            },
        ) if gate_quantization_bits == up_quantization_bits
            && gate_quantization_group_size == up_quantization_group_size =>
        {
            Ok(Qwen3_5AffineWeights::Quantized {
                packed_weight: runtime
                    .concatenate_axis(&[&gate_packed_weight, &up_packed_weight], 1)?,
                quantization_scales: runtime
                    .concatenate_axis(&[&gate_quantization_scales, &up_quantization_scales], 1)?,
                quantization_biases: runtime
                    .concatenate_axis(&[&gate_quantization_biases, &up_quantization_biases], 1)?,
                quantization_bits: gate_quantization_bits,
                quantization_group_size: gate_quantization_group_size,
            })
        }
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "validated expert gate/up fusion plan disagreed with loaded weights",
        }),
    }
}
