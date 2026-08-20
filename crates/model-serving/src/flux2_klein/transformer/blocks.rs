//! Double-stream and fused single-stream FLUX.2 block equations.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::math::{apply_rope, fp32_layer_norm, fp32_rms_norm, linear, swiglu};
use super::weights::Flux2KleinBlockWeights;
use super::{Flux2KleinTransformerError, Flux2KleinTransformerGeometry};

pub(super) struct DoubleStreamState {
    pub(super) image: MlxArray,
    pub(super) text: MlxArray,
}

pub(super) struct ModulationSet {
    pub(super) image_double: MlxArray,
    pub(super) text_double: MlxArray,
    pub(super) single: MlxArray,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn double_stream_block(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    weights: &Flux2KleinBlockWeights<'_>,
    block_index: usize,
    state: DoubleStreamState,
    modulation: &ModulationSet,
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<DoubleStreamState, Flux2KleinTransformerError> {
    let prefix = format!("transformer_blocks.{block_index}");
    let image_modulations = split_modulation(
        runtime,
        &modulation.image_double,
        geometry.hidden_width(),
        6,
    )?;
    let text_modulations =
        split_modulation(runtime, &modulation.text_double, geometry.hidden_width(), 6)?;
    let normalized_image = modulated_layer_norm(
        runtime,
        &state.image,
        &image_modulations[0],
        &image_modulations[1],
        geometry.normalization_epsilon(),
    )?;
    let normalized_text = modulated_layer_norm(
        runtime,
        &state.text,
        &text_modulations[0],
        &text_modulations[1],
        geometry.normalization_epsilon(),
    )?;
    let (image_attention, text_attention) = joint_attention(
        runtime,
        geometry,
        weights,
        &prefix,
        &normalized_image,
        &normalized_text,
        rope_cosines,
        rope_sines,
    )?;
    let gated_image_attention = runtime.multiply(&image_attention, &image_modulations[2])?;
    let gated_text_attention = runtime.multiply(&text_attention, &text_modulations[2])?;
    let image_after_attention = runtime.add(&state.image, &gated_image_attention)?;
    let text_after_attention = runtime.add(&state.text, &gated_text_attention)?;
    let image_ff_input = modulated_layer_norm(
        runtime,
        &image_after_attention,
        &image_modulations[3],
        &image_modulations[4],
        geometry.normalization_epsilon(),
    )?;
    let text_ff_input = modulated_layer_norm(
        runtime,
        &text_after_attention,
        &text_modulations[3],
        &text_modulations[4],
        geometry.normalization_epsilon(),
    )?;
    let image_ff = feed_forward(
        runtime,
        weights,
        &format!("{prefix}.ff"),
        &image_ff_input,
        geometry.feed_forward_width(),
    )?;
    let text_ff = feed_forward(
        runtime,
        weights,
        &format!("{prefix}.ff_context"),
        &text_ff_input,
        geometry.feed_forward_width(),
    )?;
    Ok(DoubleStreamState {
        image: runtime.add(
            &image_after_attention,
            &runtime.multiply(&image_ff, &image_modulations[5])?,
        )?,
        text: runtime.add(
            &text_after_attention,
            &runtime.multiply(&text_ff, &text_modulations[5])?,
        )?,
    })
}

// The explicit tensors mirror the fused checkpoint boundary and prevent hidden packing copies.
#[allow(clippy::too_many_arguments)]
pub(super) fn single_stream_block(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    weights: &Flux2KleinBlockWeights<'_>,
    block_index: usize,
    hidden_states: &MlxArray,
    modulation: &ModulationSet,
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let prefix = format!("single_transformer_blocks.{block_index}.attn");
    let modulations = split_modulation(runtime, &modulation.single, geometry.hidden_width(), 3)?;
    let normalized = modulated_layer_norm(
        runtime,
        hidden_states,
        &modulations[0],
        &modulations[1],
        geometry.normalization_epsilon(),
    )?;
    let fused = linear(
        runtime,
        &normalized,
        weights.tensor(&format!("{prefix}.to_qkv_mlp_proj.weight"))?,
    )?;
    let hidden_width = geometry.hidden_width() as i32;
    let feed_forward_width = geometry.feed_forward_width() as i32;
    let queries = slice_last(runtime, &fused, 0, hidden_width)?;
    let keys = slice_last(runtime, &fused, hidden_width, hidden_width * 2)?;
    let values = slice_last(runtime, &fused, hidden_width * 2, hidden_width * 3)?;
    let mlp_fused = slice_last(
        runtime,
        &fused,
        hidden_width * 3,
        hidden_width * 3 + feed_forward_width * 2,
    )?;
    let attention = self_attention(
        runtime,
        geometry,
        weights,
        &prefix,
        &queries,
        &keys,
        &values,
        rope_cosines,
        rope_sines,
    )?;
    let mlp = swiglu(runtime, &mlp_fused, feed_forward_width)?;
    let parallel = runtime.concatenate_axis(&[&attention, &mlp], 2)?;
    let projected = linear(
        runtime,
        &parallel,
        weights.tensor(&format!("{prefix}.to_out.weight"))?,
    )?;
    Ok(runtime.add(
        hidden_states,
        &runtime.multiply(&projected, &modulations[2])?,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn joint_attention(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    weights: &Flux2KleinBlockWeights<'_>,
    prefix: &str,
    image: &MlxArray,
    text: &MlxArray,
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<(MlxArray, MlxArray), Flux2KleinTransformerError> {
    let image_q = linear(
        runtime,
        image,
        weights.tensor(&format!("{prefix}.attn.to_q.weight"))?,
    )?;
    let image_k = linear(
        runtime,
        image,
        weights.tensor(&format!("{prefix}.attn.to_k.weight"))?,
    )?;
    let image_v = linear(
        runtime,
        image,
        weights.tensor(&format!("{prefix}.attn.to_v.weight"))?,
    )?;
    let text_q = linear(
        runtime,
        text,
        weights.tensor(&format!("{prefix}.attn.add_q_proj.weight"))?,
    )?;
    let text_k = linear(
        runtime,
        text,
        weights.tensor(&format!("{prefix}.attn.add_k_proj.weight"))?,
    )?;
    let text_v = linear(
        runtime,
        text,
        weights.tensor(&format!("{prefix}.attn.add_v_proj.weight"))?,
    )?;
    // Text and image Q/K scales are separate parameters. Apply them before
    // concatenation by selecting the matching stream rather than broadcasting
    // the concatenated vector over heads.
    let text_tokens = text.shape()[1];
    let image_tokens = image.shape()[1];
    let text_output = joint_attention_with_stream_norms(
        runtime,
        geometry,
        weights,
        prefix,
        (&text_q, &text_k, &text_v),
        (&image_q, &image_k, &image_v),
        rope_cosines,
        rope_sines,
    )?;
    let text_attention = runtime.slice(
        &text_output,
        &[0, 0, 0],
        &[
            text_output.shape()[0],
            text_tokens,
            geometry.hidden_width() as i32,
        ],
        &[1, 1, 1],
    )?;
    let image_attention = runtime.slice(
        &text_output,
        &[0, text_tokens, 0],
        &[
            text_output.shape()[0],
            text_tokens + image_tokens,
            geometry.hidden_width() as i32,
        ],
        &[1, 1, 1],
    )?;
    Ok((
        linear(
            runtime,
            &image_attention,
            weights.tensor(&format!("{prefix}.attn.to_out.0.weight"))?,
        )?,
        linear(
            runtime,
            &text_attention,
            weights.tensor(&format!("{prefix}.attn.to_add_out.weight"))?,
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn joint_attention_with_stream_norms(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    weights: &Flux2KleinBlockWeights<'_>,
    prefix: &str,
    text_qkv: (&MlxArray, &MlxArray, &MlxArray),
    image_qkv: (&MlxArray, &MlxArray, &MlxArray),
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let text_q = shaped_heads(runtime, geometry, text_qkv.0)?;
    let text_k = shaped_heads(runtime, geometry, text_qkv.1)?;
    let image_q = shaped_heads(runtime, geometry, image_qkv.0)?;
    let image_k = shaped_heads(runtime, geometry, image_qkv.1)?;
    let text_q = fp32_rms_norm(
        runtime,
        &text_q,
        weights.tensor(&format!("{prefix}.attn.norm_added_q.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let text_k = fp32_rms_norm(
        runtime,
        &text_k,
        weights.tensor(&format!("{prefix}.attn.norm_added_k.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let image_q = fp32_rms_norm(
        runtime,
        &image_q,
        weights.tensor(&format!("{prefix}.attn.norm_q.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let image_k = fp32_rms_norm(
        runtime,
        &image_k,
        weights.tensor(&format!("{prefix}.attn.norm_k.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let queries = runtime.concatenate_axis(&[&text_q, &image_q], 1)?;
    let keys = runtime.concatenate_axis(&[&text_k, &image_k], 1)?;
    let text_values = shaped_heads(runtime, geometry, text_qkv.2)?;
    let image_values = shaped_heads(runtime, geometry, image_qkv.2)?;
    let values = runtime.concatenate_axis(&[&text_values, &image_values], 1)?;
    attention(
        runtime,
        geometry,
        &queries,
        &keys,
        &values,
        rope_cosines,
        rope_sines,
    )
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    weights: &Flux2KleinBlockWeights<'_>,
    prefix: &str,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let queries = fp32_rms_norm(
        runtime,
        &shaped_heads(runtime, geometry, queries)?,
        weights.tensor(&format!("{prefix}.norm_q.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let keys = fp32_rms_norm(
        runtime,
        &shaped_heads(runtime, geometry, keys)?,
        weights.tensor(&format!("{prefix}.norm_k.weight"))?,
        geometry.normalization_epsilon(),
    )?;
    let values = shaped_heads(runtime, geometry, values)?;
    attention(
        runtime,
        geometry,
        &queries,
        &keys,
        &values,
        rope_cosines,
        rope_sines,
    )
}

fn attention(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    rope_cosines: &MlxArray,
    rope_sines: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let rotated_queries = apply_rope(runtime, queries, rope_cosines, rope_sines)?;
    let rotated_keys = apply_rope(runtime, keys, rope_cosines, rope_sines)?;
    let q = runtime.transpose_axes(&rotated_queries, &[0, 2, 1, 3])?;
    let k = runtime.transpose_axes(&rotated_keys, &[0, 2, 1, 3])?;
    let v = runtime.transpose_axes(values, &[0, 2, 1, 3])?;
    // MLX's fused Metal SDPA tiles scores and avoids an O(tokens^2) score owner.
    let attended = runtime.scaled_dot_product_attention(
        &q,
        &k,
        &v,
        (geometry.attention_head_width() as f32).sqrt().recip(),
    )?;
    let attended = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
    Ok(runtime.reshape(
        &attended,
        &[
            attended.shape()[0],
            attended.shape()[1],
            geometry.hidden_width() as i32,
        ],
    )?)
}

fn shaped_heads(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    input: &MlxArray,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let shape = input.shape();
    Ok(runtime.reshape(
        input,
        &[
            shape[0],
            shape[1],
            geometry.attention_head_count() as i32,
            geometry.attention_head_width() as i32,
        ],
    )?)
}

fn feed_forward(
    runtime: &MlxRuntime,
    weights: &Flux2KleinBlockWeights<'_>,
    prefix: &str,
    input: &MlxArray,
    width: usize,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let fused = linear(
        runtime,
        input,
        weights.tensor(&format!("{prefix}.linear_in.weight"))?,
    )?;
    let activated = swiglu(runtime, &fused, width as i32)?;
    linear(
        runtime,
        &activated,
        weights.tensor(&format!("{prefix}.linear_out.weight"))?,
    )
}

fn modulated_layer_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    shift: &MlxArray,
    scale: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let normalized = fp32_layer_norm(runtime, input, epsilon)?;
    let one_plus_scale = runtime.add(&runtime.full(&[], 1.0, scale.dtype())?, scale)?;
    Ok(runtime.add(&runtime.multiply(&normalized, &one_plus_scale)?, shift)?)
}

pub(super) fn split_modulation(
    runtime: &MlxRuntime,
    modulation: &MlxArray,
    width: usize,
    count: usize,
) -> Result<Vec<MlxArray>, Flux2KleinTransformerError> {
    let shape = modulation.shape();
    if shape.len() != 2 || shape[1] != (width * count) as i32 {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "adaptive modulation shape is invalid",
        });
    }
    (0..count)
        .map(|part_index| {
            let start = (part_index * width) as i32;
            let part = runtime.slice(
                modulation,
                &[0, start],
                &[shape[0], start + width as i32],
                &[1, 1],
            )?;
            Ok(runtime.expand_dims(&part, 1)?)
        })
        .collect()
}

fn slice_last(
    runtime: &MlxRuntime,
    input: &MlxArray,
    start: i32,
    stop: i32,
) -> Result<MlxArray, Flux2KleinTransformerError> {
    let shape = input.shape();
    Ok(runtime.slice(
        input,
        &[0, 0, start],
        &[shape[0], shape[1], stop],
        &[1, 1, 1],
    )?)
}
