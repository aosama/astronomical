//! Precision boundaries and dense primitives matching the official reference.

use runtime_integration::MlxRuntime;

use super::{Flux2KleinTransformerError, Flux2KleinTransformerGeometry};
use crate::tensor_store::Flux2KleinWeight;
use mlx_bridge::{MlxArray, MlxDtype};
use model_primitives::fp32_layers::rotate_adjacent_pairs;
pub(super) use model_primitives::fp32_layers::{fp32_layer_norm, fp32_rms_norm};

/// `input @ weightᵀ`, dense or on 4-bit packed weights depending on how the
/// artifact stores the projection.
pub(super) fn linear(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &Flux2KleinWeight,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    Ok(weight.project(runtime, input)?)
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
    if shape[3] % 2 != 0 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "RoPE feature width must contain complete real/imaginary pairs",
        });
    }
    Ok(rotate_adjacent_pairs(runtime, input, cosines, sines)?)
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
