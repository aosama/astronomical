//! Qwen3.5 vision self-attention graph assembly.
//!
//! Source lineage: Rust translation of an upstream open-source Qwen3-VL
//! attention path (see third-party/THIRD_PARTY_NOTICES). Kernel math is
//! delegated to MLX-C `mlx_fast_scaled_dot_product_attention` declared in
//! `mlx-c/mlx/c/fast.h`.

use std::cell::RefCell;

use astronomical_runtime_integration::MlxRuntime;

use super::{Qwen3_5ExecutionError, Qwen3_5VisionConfig, Qwen3_5VisionWeights};
use mlx_c_rust::{MlxArray, MlxCompiledElementwiseGraphs, MlxDtype};

/// Retains the attention head-dimension padding zeros so the 27 vision blocks
/// stop re-allocating an identical zero tensor three times per block per image
/// forward (issue #915 item 4i): 81 zero-tensor allocations and writes become
/// one retained array. One slot keyed by the full zeros shape and dtype —
/// the patch count is constant within a request, so a request with a
/// different patch count replaces the retained array and the old one drops,
/// keeping the retained memory bounded to a single padding tensor.
#[derive(Debug, Default)]
pub struct Qwen3_5VisionPaddingZeroCache {
    retained: RefCell<Option<RetainedPaddingZeros>>,
}

#[derive(Debug)]
struct RetainedPaddingZeros {
    zeros_shape: [i32; 4],
    dtype: MlxDtype,
    zeros: MlxArray,
}

impl RetainedPaddingZeros {
    fn matches(&self, zeros_shape: [i32; 4], dtype: MlxDtype) -> bool {
        self.zeros_shape == zeros_shape && self.dtype == dtype
    }
}

impl Qwen3_5VisionPaddingZeroCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            retained: RefCell::new(None),
        }
    }

    /// Pads the last (head) dimension to the fused SDPA width, reusing the
    /// retained zeros when the requested padding shape and dtype match.
    pub fn pad_attention_head_dimension(
        &self,
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
        let zeros_shape = [
            1,
            head_count,
            patch_count,
            fused_head_dimension - head_dimension,
        ];
        let dtype = attention_states.dtype();
        let mut retained_slot = self.retained.borrow_mut();
        if !retained_slot
            .as_ref()
            .is_some_and(|retained| retained.matches(zeros_shape, dtype))
        {
            retained_slot.replace(RetainedPaddingZeros {
                zeros_shape,
                dtype,
                zeros: runtime.zeros(&zeros_shape, dtype)?,
            });
        }
        let Some(retained) = retained_slot.as_ref() else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "vision padding zeros must be retained before reuse",
            });
        };
        Ok(runtime.concatenate_axis(&[attention_states, &retained.zeros], 3)?)
    }
}

/// Runs one unmasked, image-segmented vision attention layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn qwen3_5_vision_self_attention(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    vision_config: &Qwen3_5VisionConfig,
    vision_weights: &Qwen3_5VisionWeights,
    normalized_hidden_states: &MlxArray,
    vision_block_prefix: &str,
    attention_sequence_boundaries: &[u32],
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
    padding_zero_cache: &Qwen3_5VisionPaddingZeroCache,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let patch_count = normalized_hidden_states.shape()[0];
    let head_count = u32_to_i32(vision_config.head_count())?;
    let hidden_size = u32_to_i32(vision_config.hidden_size())?;
    let head_dimension = hidden_size / head_count;
    let (query_states, key_states, value_states) = project_vision_qkv_states(
        runtime,
        vision_weights,
        normalized_hidden_states,
        vision_block_prefix,
        patch_count,
        head_count,
        head_dimension,
    )?;
    let query_states = apply_rotary_embedding(
        runtime,
        compiled_elementwise_graphs,
        &query_states,
        rotary_cosines,
        rotary_sines,
    )?;
    let key_states = apply_rotary_embedding(
        runtime,
        compiled_elementwise_graphs,
        &key_states,
        rotary_cosines,
        rotary_sines,
    )?;

    // MLX fast SDPA expects [batch, heads, sequence, head_dimension]. The vision
    // tower has one logical batch; explicit boundaries below isolate images.
    let (query_states, key_states, value_states) =
        transpose_vision_states_to_sdpa_layout(runtime, query_states, key_states, value_states)?;

    // The pinned head width is 72. The MLX fused SDPA kernel supports [64, 80,
    // 128], so pad to the next specialization and slice back afterward.
    let fused_head_dimension = fused_sdpa_head_dimension(head_dimension);
    let query_states = padding_zero_cache.pad_attention_head_dimension(
        runtime,
        &query_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;
    let key_states = padding_zero_cache.pad_attention_head_dimension(
        runtime,
        &key_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;
    let value_states = padding_zero_cache.pad_attention_head_dimension(
        runtime,
        &value_states,
        head_count,
        patch_count,
        fused_head_dimension,
    )?;

    // Preserve Python `head_dim ** -0.5`: double-precision exponentiation then
    // f32 conversion. A direct f32 reciprocal sqrt differs by one ULP here.
    let attention_scale = (head_dimension as f64).powf(-0.5) as f32;
    let concatenated_attention_output = run_segmented_sdpa(
        runtime,
        &query_states,
        &key_states,
        &value_states,
        attention_sequence_boundaries,
        head_count,
        fused_head_dimension,
        attention_scale,
    )?;

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

/// Projects the normalized patch states into Q, K, and V. One affine
/// projection stores Q, K, and V contiguously; the reshape interprets it as
/// [patch, qkv, head, head_dimension] without copying, and the three
/// components come off that view by their fixed slots.
fn project_vision_qkv_states(
    runtime: &MlxRuntime,
    vision_weights: &Qwen3_5VisionWeights,
    normalized_hidden_states: &MlxArray,
    vision_block_prefix: &str,
    patch_count: i32,
    head_count: i32,
    head_dimension: i32,
) -> Result<(MlxArray, MlxArray, MlxArray), Qwen3_5ExecutionError> {
    let qkv_projection = linear(
        runtime,
        vision_weights,
        normalized_hidden_states,
        &format!("{vision_block_prefix}.attn.qkv"),
    )?;
    let qkv_by_projection = runtime.reshape(
        &qkv_projection,
        &[patch_count, 3, head_count, head_dimension],
    )?;
    let query_states = qkv_component(runtime, &qkv_by_projection, 0)?;
    let key_states = qkv_component(runtime, &qkv_by_projection, 1)?;
    let value_states = qkv_component(runtime, &qkv_by_projection, 2)?;
    Ok((query_states, key_states, value_states))
}

/// Moves the three attention states from [sequence, heads, head_dimension] to
/// MLX's [batch, heads, sequence, head_dimension] SDPA layout. The vision
/// tower has one logical batch; segment boundaries, not batch rows, isolate
/// the concatenated images.
fn transpose_vision_states_to_sdpa_layout(
    runtime: &MlxRuntime,
    query_states: MlxArray,
    key_states: MlxArray,
    value_states: MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray), Qwen3_5ExecutionError> {
    let query_states =
        runtime.expand_dims(&runtime.transpose_axes(&query_states, &[1, 0, 2])?, 0)?;
    let key_states = runtime.expand_dims(&runtime.transpose_axes(&key_states, &[1, 0, 2])?, 0)?;
    let value_states =
        runtime.expand_dims(&runtime.transpose_axes(&value_states, &[1, 0, 2])?, 0)?;
    Ok((query_states, key_states, value_states))
}

/// Selects the padded head width the MLX fused SDPA kernel specializes on.
/// The kernel supports [64, 80, 128]; anything between specializations pads
/// up to the next one (the pinned vision width of 72 pads to 80), and a width
/// beyond every specialization runs unpadded rather than overflowing.
pub(super) fn fused_sdpa_head_dimension(head_dimension: i32) -> i32 {
    [64, 80, 128]
        .into_iter()
        .find(|supported_head_dimension| *supported_head_dimension >= head_dimension)
        .unwrap_or(head_dimension)
}

/// Runs scaled dot-product attention per boundary-delimited segment and
/// concatenates the outputs along the sequence axis. Concatenated images and
/// frames must not attend to each other, so each image's segment is sliced
/// from the padded states and attended alone; neutral padding rides along
/// because appended Q/K zeros add nothing to QK^T and appended V zeros
/// produce zero output coordinates.
#[allow(clippy::too_many_arguments)]
fn run_segmented_sdpa(
    runtime: &MlxRuntime,
    query_states: &MlxArray,
    key_states: &MlxArray,
    value_states: &MlxArray,
    attention_sequence_boundaries: &[u32],
    head_count: i32,
    fused_head_dimension: i32,
    attention_scale: f32,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let mut segmented_attention_outputs = Vec::new();
    for sequence_boundary_pair in attention_sequence_boundaries.windows(2) {
        let sequence_start = u32_to_i32(sequence_boundary_pair[0])?;
        let sequence_end = u32_to_i32(sequence_boundary_pair[1])?;
        let segment_starts = [0, 0, sequence_start, 0];
        let segment_stops = [1, head_count, sequence_end, fused_head_dimension];
        let segment_strides = [1, 1, 1, 1];
        let query_segment = runtime.slice(
            query_states,
            &segment_starts,
            &segment_stops,
            &segment_strides,
        )?;
        let key_segment = runtime.slice(
            key_states,
            &segment_starts,
            &segment_stops,
            &segment_strides,
        )?;
        let value_segment = runtime.slice(
            value_states,
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
    Ok(runtime.concatenate_axis(&segmented_attention_output_references, 2)?)
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
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    attention_states: &MlxArray,
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let attention_shape = attention_states.shape();
    // `rotate_half([x1,x2])=[-x2,x1]`; then x*cos + rotate_half(x)*sin.
    // [patch,1,head_dimension] trigonometric arrays broadcast across heads.
    let half_head_dimension = attention_shape[2] / 2;
    // The head-half slices stay outside the compiled graph: the pinned MLX Slice
    // primitive cannot be shapeless-compiled, so the pre-sliced halves are passed
    // in and the remaining composite fuses its elementwise tail into one kernel.
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
    Ok(runtime.apply_compiled_vision_rope(
        compiled_elementwise_graphs,
        attention_states,
        rotary_cosines,
        rotary_sines,
        &first_half,
        &second_half,
    )?)
}

fn linear(
    runtime: &MlxRuntime,
    vision_weights: &Qwen3_5VisionWeights,
    input_states: &MlxArray,
    linear_prefix: &str,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    vision_weights.linear(runtime, input_states, linear_prefix)
}

fn u32_to_i32(dimension_size: u32) -> Result<i32, Qwen3_5ExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| Qwen3_5ExecutionError::InvalidInput {
        description: "vision dimension exceeds the MLX integer range",
    })
}
