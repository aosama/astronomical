//! Qwen3.5 vision self-attention graph assembly.
//!
//! Source lineage: Rust translation of the MLX-VLM Qwen3-VL attention path
//! (MIT License; see third-party license notices). Kernel math is delegated to
//! MLX-C `mlx_fast_scaled_dot_product_attention` declared in `mlx-c/mlx/c/fast.h`.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::{Qwen3_5ExecutionError, Qwen3_5VisionConfig, Qwen3_5VisionWeights};

/// Runs one unmasked, image-segmented vision attention layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn qwen3_5_vision_self_attention(
    runtime: &MlxRuntime,
    vision_config: &Qwen3_5VisionConfig,
    vision_weights: &Qwen3_5VisionWeights,
    normalized_hidden_states: &MlxArray,
    vision_block_prefix: &str,
    attention_sequence_boundaries: &[u32],
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let patch_count = normalized_hidden_states.shape()[0];
    let head_count = u32_to_i32(vision_config.head_count())?;
    let hidden_size = u32_to_i32(vision_config.hidden_size())?;
    let head_dimension = hidden_size / head_count;
    let qkv_projection = linear(
        runtime,
        vision_weights,
        normalized_hidden_states,
        &format!("{vision_block_prefix}.attn.qkv"),
    )?;

    // One affine projection stores Q, K, and V contiguously. The reshape
    // interprets it as [patch, qkv, head, head_dimension] without copying.
    let qkv_by_projection = runtime.reshape(
        &qkv_projection,
        &[patch_count, 3, head_count, head_dimension],
    )?;
    let query_states = qkv_component(runtime, &qkv_by_projection, 0)?;
    let key_states = qkv_component(runtime, &qkv_by_projection, 1)?;
    let value_states = qkv_component(runtime, &qkv_by_projection, 2)?;
    let query_states =
        apply_rotary_embedding(runtime, &query_states, rotary_cosines, rotary_sines)?;
    let key_states = apply_rotary_embedding(runtime, &key_states, rotary_cosines, rotary_sines)?;

    // MLX fast SDPA expects [batch, heads, sequence, head_dimension]. The vision
    // tower has one logical batch; explicit boundaries below isolate images.
    let query_states =
        runtime.expand_dims(&runtime.transpose_axes(&query_states, &[1, 0, 2])?, 0)?;
    let key_states = runtime.expand_dims(&runtime.transpose_axes(&key_states, &[1, 0, 2])?, 0)?;
    let value_states =
        runtime.expand_dims(&runtime.transpose_axes(&value_states, &[1, 0, 2])?, 0)?;

    // The pinned head width is 72. The MLX fused SDPA kernel supports [64, 80,
    // 128], so pad to the next specialization and slice back afterward.
    let fused_head_dimension = [64, 80, 128]
        .into_iter()
        .find(|supported_head_dimension| *supported_head_dimension >= head_dimension)
        .unwrap_or(head_dimension);
    let query_states = pad_attention_head_dimension(
        runtime,
        &query_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;
    let key_states = pad_attention_head_dimension(
        runtime,
        &key_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;
    let value_states = pad_attention_head_dimension(
        runtime,
        &value_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;

    // Preserve Python `head_dim ** -0.5`: double-precision exponentiation then
    // f32 conversion. A direct f32 reciprocal sqrt differs by one ULP here.
    let attention_scale = (head_dimension as f64).powf(-0.5) as f32;
    let mut segmented_attention_outputs = Vec::new();
    // Concatenated images and frames must not attend to each other. Execute SDPA
    // per boundary-delimited segment, then concatenate the outputs.
    for sequence_boundary_pair in attention_sequence_boundaries.windows(2) {
        let sequence_start = u32_to_i32(sequence_boundary_pair[0])?;
        let sequence_end = u32_to_i32(sequence_boundary_pair[1])?;
        let segment_starts = [0, 0, sequence_start, 0];
        let segment_stops = [1, head_count, sequence_end, fused_head_dimension];
        let segment_strides = [1, 1, 1, 1];
        let query_segment = runtime.slice(
            &query_states,
            &segment_starts,
            &segment_stops,
            &segment_strides,
        )?;
        let key_segment = runtime.slice(
            &key_states,
            &segment_starts,
            &segment_stops,
            &segment_strides,
        )?;
        let value_segment = runtime.slice(
            &value_states,
            &segment_starts,
            &segment_stops,
            &segment_strides,
        )?;
        segmented_attention_outputs.push(runtime.scaled_dot_product_attention(
            &query_segment,
            &key_segment,
            &value_segment,
            attention_scale,
        )?);
    }

    let segmented_attention_output_references =
        segmented_attention_outputs.iter().collect::<Vec<_>>();
    let concatenated_attention_output =
        runtime.concatenate_axis(&segmented_attention_output_references, 2)?;
    // Remove neutral padding and restore [patch, hidden] for output projection.
    let concatenated_attention_output = runtime.slice(
        &concatenated_attention_output,
        &[0, 0, 0, 0],
        &[1, head_count, patch_count, head_dimension],
        &[1, 1, 1, 1],
    )?;
    let transposed_attention_output =
        runtime.transpose_axes(&concatenated_attention_output, &[0, 2, 1, 3])?;
    let flattened_attention_output =
        runtime.reshape(&transposed_attention_output, &[patch_count, hidden_size])?;
    linear(
        runtime,
        vision_weights,
        &flattened_attention_output,
        &format!("{vision_block_prefix}.attn.proj"),
    )
}

fn qkv_component(
    runtime: &MlxRuntime,
    qkv_by_projection: &MlxArray,
    projection_index: i32,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let qkv_shape = qkv_by_projection.shape();
    let selected_projection = runtime.slice(
        qkv_by_projection,
        &[0, projection_index, 0, 0],
        &[
            qkv_shape[0],
            projection_index + 1,
            qkv_shape[2],
            qkv_shape[3],
        ],
        &[1, 1, 1, 1],
    )?;
    Ok(runtime.squeeze_axis(&selected_projection, 1)?)
}

fn apply_rotary_embedding(
    runtime: &MlxRuntime,
    attention_states: &MlxArray,
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let attention_shape = attention_states.shape();
    // `rotate_half([x1,x2])=[-x2,x1]`; then x*cos + rotate_half(x)*sin.
    // [patch,1,head_dimension] trigonometric arrays broadcast across heads.
    let half_head_dimension = attention_shape[2] / 2;
    let first_half = runtime.slice(
        attention_states,
        &[0, 0, 0],
        &[attention_shape[0], attention_shape[1], half_head_dimension],
        &[1, 1, 1],
    )?;
    let second_half = runtime.slice(
        attention_states,
        &[0, 0, half_head_dimension],
        &attention_shape,
        &[1, 1, 1],
    )?;
    let negative_second_half = runtime.negative(&second_half)?;
    let rotated_attention_states =
        runtime.concatenate_axis(&[&negative_second_half, &first_half], 2)?;
    let cosine_component = runtime.multiply(attention_states, rotary_cosines)?;
    let sine_component = runtime.multiply(&rotated_attention_states, rotary_sines)?;
    // Reference restores the original dtype after potentially promoted arithmetic.
    let rotated_attention_states_f32 = runtime.add(&cosine_component, &sine_component)?;
    Ok(runtime.astype(&rotated_attention_states_f32, attention_states.dtype())?)
}

fn pad_attention_head_dimension(
    runtime: &MlxRuntime,
    attention_states: &MlxArray,
    head_count: i32,
    patch_count: i32,
    fused_head_dimension: i32,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let head_dimension = attention_states.shape()[3];
    if head_dimension == fused_head_dimension {
        return Ok(runtime.reshape(attention_states, &attention_states.shape())?);
    }
    // Appended Q/K zeros add nothing to QK^T; appended V zeros produce zero
    // output coordinates. Slicing after SDPA therefore recovers original math.
    let padding_states = runtime.zeros(
        &[
            1,
            head_count,
            patch_count,
            fused_head_dimension - head_dimension,
        ],
        attention_states.dtype(),
    )?;
    Ok(runtime.concatenate_axis(&[attention_states, &padding_states], 3)?)
}

fn linear(
    runtime: &MlxRuntime,
    vision_weights: &Qwen3_5VisionWeights,
    input_states: &MlxArray,
    linear_prefix: &str,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let linear_weight = vision_weights.tensor(&format!("{linear_prefix}.weight"))?;
    let transposed_linear_weight = runtime.transpose_axes(linear_weight, &[1, 0])?;
    let linear_bias = vision_weights.tensor(&format!("{linear_prefix}.bias"))?;
    // Fused MLX-C addmm preserves the required BF16 accumulation path.
    Ok(runtime.addmm(
        linear_bias,
        input_states,
        &transposed_linear_weight,
        1.0,
        1.0,
    )?)
}

fn u32_to_i32(dimension_size: u32) -> Result<i32, Qwen3_5ExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| Qwen3_5ExecutionError::InvalidInput {
        description: "vision dimension exceeds the MLX integer range",
    })
}
