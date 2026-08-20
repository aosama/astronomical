//! Precision boundaries and dense primitives matching the official reference.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::{Flux2KleinTransformerError, Flux2KleinTransformerGeometry};

pub(super) fn linear(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let transposed_weight = runtime.transpose_axes(weight, &[1, 0])?;
    Ok(runtime.matmul(input, &transposed_weight)?)
}

pub(super) fn fp32_layer_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let normalized_f32 = runtime
        .layer_norm_without_weight_and_bias(&runtime.astype(input, MlxDtype::Float32)?, epsilon)?;
    Ok(runtime.astype(&normalized_f32, input.dtype())?)
}

pub(super) fn fp32_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    // The reference computes variance in FP32, then returns to the value dtype
    // before applying its retained BF16 learned scale.
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let normalized_f32 = runtime.rms_norm_without_weight(&input_f32, epsilon)?;
    let normalized = runtime.astype(&normalized_f32, input.dtype())?;
    Ok(runtime.multiply(&normalized, weight)?)
}

pub(super) fn swiglu(
    runtime: &MlxRuntime,
    fused_gate_values: &MlxArray,
    output_width: i32,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let shape = fused_gate_values.shape();
    if shape.len() != 3 || shape[2] != output_width * 2 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "SwiGLU input shape is invalid",
        });
    }
    let gate = runtime.slice(
        fused_gate_values,
        &[0, 0, 0],
        &[shape[0], shape[1], output_width],
        &[1, 1, 1],
    )?;
    let values = runtime.slice(
        fused_gate_values,
        &[0, 0, output_width],
        &[shape[0], shape[1], output_width * 2],
        &[1, 1, 1],
    )?;
    Ok(runtime.multiply(&runtime.silu(&gate)?, &values)?)
}

pub(super) fn timestep_embedding(
    runtime: &MlxRuntime,
    timesteps: &MlxArray,
    embedding_width: usize,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    if timesteps.shape().len() != 1 || !embedding_width.is_multiple_of(2) || embedding_width < 4 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "timestep embedding input is invalid",
        });
    }
    let half_width = embedding_width / 2;
    let frequencies = (0..half_width)
        .map(|frequency_index| {
            (-10_000.0_f32.ln() * frequency_index as f32 / half_width as f32).exp()
        })
        .collect::<Vec<_>>();
    let frequencies = runtime.array_from_f32(&frequencies, &[1, half_width as i32])?;
    let timesteps_f32 = runtime.astype(timesteps, MlxDtype::Float32)?;
    let scaled_timesteps = runtime.multiply_scalar(&timesteps_f32, 1_000.0)?;
    let angles = runtime.multiply(&runtime.expand_dims(&scaled_timesteps, 1)?, &frequencies)?;
    // Timesteps(..., flip_sin_to_cos=true) orders cosine before sine.
    let cosines = runtime.cos(&angles)?;
    let sines = runtime.sin(&angles)?;
    Ok(runtime.concatenate_axis(&[&cosines, &sines], 1)?)
}

pub(super) fn rope_frequencies(
    runtime: &MlxRuntime,
    position_ids: &MlxArray,
    geometry: &Flux2KleinTransformerGeometry,
) -> Result<(MlxArray, MlxArray), Flux2KleinTransformerError> {
    let shape = position_ids.shape();
    if shape.len() != 2 || shape[1] != 4 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "RoPE position IDs must have shape [tokens, 4]",
        });
    }
    let position_ids_f32 = runtime.astype(position_ids, MlxDtype::Float32)?;
    let mut axis_cosines = Vec::with_capacity(4);
    let mut axis_sines = Vec::with_capacity(4);
    for (axis_index, axis_width) in geometry.rope_axis_widths().into_iter().enumerate() {
        let half_axis_width = axis_width / 2;
        let inverse_frequencies = (0..half_axis_width)
            .map(|frequency_index| {
                geometry
                    .rope_theta()
                    .powf(-(frequency_index as f32 / half_axis_width as f32))
            })
            .collect::<Vec<_>>();
        let frequencies =
            runtime.array_from_f32(&inverse_frequencies, &[1, half_axis_width as i32])?;
        let positions = runtime.slice(
            &position_ids_f32,
            &[0, axis_index as i32],
            &[shape[0], axis_index as i32 + 1],
            &[1, 1],
        )?;
        let angles = runtime.multiply(&positions, &frequencies)?;
        // Each frequency coefficient applies to one adjacent real/imaginary pair.
        axis_cosines.push(runtime.repeat_axis(&runtime.cos(&angles)?, 2, 1)?);
        axis_sines.push(runtime.repeat_axis(&runtime.sin(&angles)?, 2, 1)?);
    }
    let cosine_refs = axis_cosines.iter().collect::<Vec<_>>();
    let sine_refs = axis_sines.iter().collect::<Vec<_>>();
    Ok((
        runtime.concatenate_axis(&cosine_refs, 1)?,
        runtime.concatenate_axis(&sine_refs, 1)?,
    ))
}

pub(super) fn apply_rope(
    runtime: &MlxRuntime,
    input: &MlxArray,
    cosines: &MlxArray,
    sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let shape = input.shape();
    if shape.len() != 4
        || cosines.shape() != [shape[1], shape[3]]
        || sines.shape() != cosines.shape()
    {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "RoPE tensors are incompatible",
        });
    }
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    if shape[3] % 2 != 0 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "RoPE feature width must contain complete real/imaginary pairs",
        });
    }
    let real_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    let imaginary_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 1],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    // The official FLUX layout stores each complex component as adjacent real and
    // imaginary values; rotating feature halves would couple unrelated frequencies.
    let negative_imaginary_values = runtime.negative(&imaginary_values)?;
    let rotated_pairs = runtime.concatenate_axis(
        &[
            &runtime.expand_dims(&negative_imaginary_values, 4)?,
            &runtime.expand_dims(&real_values, 4)?,
        ],
        4,
    )?;
    let rotated_pairs = runtime.reshape(&rotated_pairs, &shape)?;
    let cosines = runtime.reshape(cosines, &[1, shape[1], 1, shape[3]])?;
    let sines = runtime.reshape(sines, &[1, shape[1], 1, shape[3]])?;
    let cosine_component = runtime.multiply(&input_f32, &cosines)?;
    let sine_component = runtime.multiply(&rotated_pairs, &sines)?;
    let rotated = runtime.add(&cosine_component, &sine_component)?;
    Ok(runtime.astype(&rotated, input.dtype())?)
}

#[doc(hidden)]
pub fn apply_rope_for_component_oracle(
    runtime: &MlxRuntime,
    input: &MlxArray,
    cosines: &MlxArray,
    sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    apply_rope(runtime, input, cosines, sines)
}
