//! Applies the layer's canonical rotary policy using the family-neutral helpers.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::attention::compute_yarn_rope_frequency_denominators;
use crate::laguna::normalization::LagunaRopeDescriptor;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

pub(super) fn apply_layer_rope(
    runtime: &MlxRuntime,
    input: &MlxArray,
    rope: &LagunaRopeDescriptor,
    offset_tokens: i32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution
        .measure_operation(PerformanceOperation::RotaryEmbeddingApplication, |_| {
            apply_layer_rope_inner(runtime, input, rope, offset_tokens)
        })
}

fn apply_layer_rope_inner(
    runtime: &MlxRuntime,
    input: &MlxArray,
    rope: &LagunaRopeDescriptor,
    offset_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let rotary_dimension = rope.rotary_dimension() as i32;
    match rope {
        LagunaRopeDescriptor::Default(descriptor) => runtime.rope(
            input,
            rotary_dimension,
            descriptor.rope_theta() as f32,
            offset_tokens,
        ),
        LagunaRopeDescriptor::Yarn(descriptor) => {
            let frequency_denominators = compute_yarn_rope_frequency_denominators(
                descriptor.rope_theta(),
                descriptor.rotary_dimension(),
                descriptor.original_maximum_position_count(),
                descriptor.factor(),
                descriptor.beta_fast(),
                descriptor.beta_slow(),
            )
            .map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "apply Laguna YaRN rotary embedding",
                description: "YaRN frequency denominators are invalid".to_owned(),
            })?;
            let denominator_array = runtime.array_from_f32(
                frequency_denominators.frequency_denominators(),
                &[frequency_denominators.frequency_denominators().len() as i32],
            )?;
            let prepared_input = scale_rotary_prefix(
                runtime,
                input,
                rotary_dimension,
                descriptor.attention_factor() as f32,
            )?;
            runtime.rope_with_custom_frequencies(
                &prepared_input,
                rotary_dimension,
                &denominator_array,
                1.0,
                offset_tokens,
            )
        }
    }
}

fn scale_rotary_prefix(
    runtime: &MlxRuntime,
    input: &MlxArray,
    rotary_dimension: i32,
    attention_factor: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    if (attention_factor - 1.0).abs() <= f32::EPSILON {
        return input.retain();
    }
    let input_shape = input.shape();
    if input_shape.len() != 4 || rotary_dimension >= input_shape[3] {
        return runtime.multiply_scalar(input, attention_factor);
    }
    let scaled = runtime.multiply_scalar(input, attention_factor)?;
    let scaled_prefix = runtime.slice(
        &scaled,
        &[0, 0, 0, 0],
        &[
            input_shape[0],
            input_shape[1],
            input_shape[2],
            rotary_dimension,
        ],
        &[1, 1, 1, 1],
    )?;
    let unscaled_tail = runtime.slice(
        input,
        &[0, 0, 0, rotary_dimension],
        &[
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ],
        &[1, 1, 1, 1],
    )?;
    runtime.concatenate_axis(&[&scaled_prefix, &unscaled_tail], 3)
}
