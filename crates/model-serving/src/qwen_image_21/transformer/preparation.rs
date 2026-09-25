//! Request preparation: the joint sequence, rotary tables, shared modulation, and segment masks.
//!
//! Port of everything the reference `QwenImage21Transformer2DModel.forward` computes before its
//! block loop. The heavy index bookkeeping (RoPE frequencies, image ids, target mask, prefix
//! segments, the timestep sinusoid) reuses the family's hermetically-tested pure modules; this
//! file binds their results to MLX arrays and builds the once-per-forward tensors all 32 blocks
//! share.
//!
//! One reference detail drives the joint-sequence layout: every vision-language image slot
//! stands for a 2×2 group of latent tokens, so each `true` in the slot mask expands to four
//! tokens whose values come from `img_in(packed_latents)`, while text tokens copy their VLM
//! embedding one-to-one.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::kv_cache::prefix_length;
use crate::qwen_image_21::mask::{build_image_ids, prefix_segments};
use crate::qwen_image_21::mlx_math::{attention_scale, fp32_zero_center_rms_norm};
use crate::qwen_image_21::modulation::build_target_token_mask;
use crate::qwen_image_21::rope::QwenImage21Rope;
use crate::qwen_image_21::time::SinusoidalTimesteps;

use super::blocks::{AttentionSegment, LAYER_NORM_EPSILON};
use super::weights::{
    CONTEXT_INPUT_WIDTH, HEAD_WIDTH, HIDDEN_WIDTH, LATENT_CHANNEL_COUNT, MODULATION_WIDTH,
    QwenImage21TransformerWeights, TIMESTEP_EMBEDDING_WIDTH,
};

/// Each vision-language image slot represents a 2×2 group of latent tokens.
const IMG_TOKENS_PER_SLOT: usize = 4;
/// The reference sinusoid: `max_period = 10_000`, `time_factor = 1_000`.
const SINUSOIDAL_MAX_PERIOD: f64 = 10_000.0;
const SINUSOIDAL_TIME_FACTOR: f64 = 1_000.0;

/// One transformer forward request.
pub struct QwenImage21TransformerRequest<'a> {
    /// Packed latents `(batch, image_token_count, 64)`: condition-image tokens then target tokens.
    pub packed_latents: &'a MlxArray,
    /// Vision-language embeddings `(batch, vlm_sequence_length, 4096)`, including image-slot
    /// positions (their values are discarded — only their slots' positions matter).
    pub text_embeddings: &'a MlxArray,
    /// Image-slot mask over the vision-language sequence: `true` at condition-image slots.
    pub vlm_image_mask: &'a [bool],
    /// Per-image `(height, width)` latent shapes, condition images first and the target last.
    pub img_shapes: &'a [(usize, usize)],
    /// Current denoising timesteps in `[0, 1]`, one per batch sample.
    pub timesteps: &'a [f32],
}

/// Everything the block loop and output head consume, computed once per forward.
pub(super) struct PreparedForward {
    pub(super) joint_hidden_states: MlxArray,
    pub(super) rope_cosines: MlxArray,
    pub(super) rope_sines: MlxArray,
    pub(super) segments: Vec<AttentionSegment>,
    pub(super) segment_masks: Vec<MlxArray>,
    pub(super) attention_scale: f32,
    pub(super) attention_one_plus_scale: MlxArray,
    pub(super) attention_gate: MlxArray,
    pub(super) feed_forward_one_plus_scale: MlxArray,
    pub(super) feed_forward_gate: MlxArray,
    pub(super) head_one_plus_scale: MlxArray,
    pub(super) prefix_len: usize,
    pub(super) batch: usize,
    pub(super) sequence_length: usize,
}

#[allow(clippy::too_many_lines)]
pub(super) fn prepare_forward(
    runtime: &MlxRuntime,
    weights: &QwenImage21TransformerWeights,
    request: &QwenImage21TransformerRequest<'_>,
    rope: &QwenImage21Rope,
) -> Result<PreparedForward, QwenImage21EngineError> {
    let batch = validate_request(request)?;
    let target_shape = request
        .img_shapes
        .last()
        .copied()
        .expect("validation guarantees a target image shape");
    let target_tokens = target_shape.0 * target_shape.1;
    let image_token_count: usize = request
        .img_shapes
        .iter()
        .map(|&(height, width)| height * width)
        .sum();

    // 1. Project both streams into the hidden width.
    let image_hidden = weights.img_in.forward(runtime, request.packed_latents)?;
    let text_normalized = fp32_zero_center_rms_norm(
        runtime,
        request.text_embeddings,
        &weights.text_norm,
        LAYER_NORM_EPSILON,
    )?;
    let text_projected = weights.txt_in_layer.forward(runtime, &text_normalized)?;
    let text_activated = runtime.gelu_tanh(&text_projected)?;
    let text_hidden = weights.txt_out_layer.forward(runtime, &text_activated)?;

    // 2. Joint sequence: slots expand to 2×2 latent-token groups, text copies one-to-one.
    let mut slot_mask = request.vlm_image_mask.to_vec();
    slot_mask.extend(std::iter::repeat(true).take(target_tokens / IMG_TOKENS_PER_SLOT));
    let joint_mask: Vec<bool> = slot_mask
        .iter()
        .flat_map(|&is_image| {
            std::iter::repeat(is_image).take(if is_image { IMG_TOKENS_PER_SLOT } else { 1 })
        })
        .collect();
    let joint_hidden_states = assemble_joint_sequence(
        runtime,
        &image_hidden,
        &text_hidden,
        &slot_mask,
        batch,
        image_token_count,
    )?;
    let sequence_length = joint_mask.len();

    // 3. Rotary tables from the hermetically-tested pure module, uploaded and pair-repeated.
    let (cosine_values, sine_values) = rope.frequencies(request.img_shapes, &joint_mask);
    let rope_half_width = (HEAD_WIDTH / 2) as i32;
    let rope_cosines =
        runtime.array_from_f32(&cosine_values, &[sequence_length as i32, rope_half_width])?;
    let rope_cosines = runtime.repeat_axis(&rope_cosines, 2, 1)?;
    let rope_sines =
        runtime.array_from_f32(&sine_values, &[sequence_length as i32, rope_half_width])?;
    let rope_sines = runtime.repeat_axis(&rope_sines, 2, 1)?;

    // 4. Attention structure: prefix segments and their additive masks.
    let image_ids = build_image_ids(request.img_shapes, &joint_mask);
    let target_token_mask = build_target_token_mask(request.img_shapes, &joint_mask);
    let prefix_len = prefix_length(&target_token_mask);
    let segments: Vec<AttentionSegment> = prefix_segments(&image_ids, prefix_len)
        .into_iter()
        .map(|(start, end, is_text)| AttentionSegment {
            query_start: start,
            query_end: end,
            key_end: end,
            is_text,
        })
        .collect();
    let segment_masks = build_segment_masks(runtime, &segments)?;

    // 5. Timestep conditioning: the sinusoid rows plus the extra `t = 0` row for
    // `causal_condition`, then the two projection layers.
    let sinusoidal = SinusoidalTimesteps::new(
        TIMESTEP_EMBEDDING_WIDTH,
        SINUSOIDAL_MAX_PERIOD,
        SINUSOIDAL_TIME_FACTOR,
    );
    let mut timestep_rows = Vec::with_capacity((batch + 1) * TIMESTEP_EMBEDDING_WIDTH);
    for &timestep in request.timesteps {
        timestep_rows.extend(sinusoidal.embed(f64::from(timestep)));
    }
    timestep_rows.extend(sinusoidal.embed(0.0));
    let modulation_rows = (batch + 1) as i32;
    let timestep_embeddings = runtime.array_from_f32(
        &timestep_rows,
        &[modulation_rows, TIMESTEP_EMBEDDING_WIDTH as i32],
    )?;
    let timestep_embeddings =
        runtime.astype(&timestep_embeddings, request.packed_latents.dtype())?;
    let timestep_projected = weights
        .time_embed_linear_1
        .forward(runtime, &timestep_embeddings)?;
    let timestep_activated = runtime.silu(&timestep_projected)?;
    let conditioning = weights
        .time_embed_linear_2
        .forward(runtime, &timestep_activated)?;

    // 6. Shared modulation and the AdaLN output scale, row-selected per token.
    let modulation = weights
        .modulation
        .forward(runtime, &runtime.silu(&conditioning)?)?;
    let head_scale = weights
        .norm_out_linear
        .forward(runtime, &runtime.silu(&conditioning)?)?;
    let modulation_width = HIDDEN_WIDTH;
    let attention_one_plus_scale = selected_one_plus_scale(
        runtime,
        &modulation,
        0,
        modulation_width,
        batch,
        prefix_len,
        sequence_length,
    )?;
    let attention_gate = runtime.tanh(&selected_modulation_part(
        runtime,
        &modulation,
        modulation_width,
        2 * modulation_width,
        batch,
        prefix_len,
        sequence_length,
    )?)?;
    let feed_forward_one_plus_scale = selected_one_plus_scale(
        runtime,
        &modulation,
        2 * modulation_width,
        3 * modulation_width,
        batch,
        prefix_len,
        sequence_length,
    )?;
    let feed_forward_gate = runtime.tanh(&selected_modulation_part(
        runtime,
        &modulation,
        3 * modulation_width,
        MODULATION_WIDTH,
        batch,
        prefix_len,
        sequence_length,
    )?)?;
    let head_one_plus_scale = selected_one_plus_scale(
        runtime,
        &head_scale,
        0,
        modulation_width,
        batch,
        prefix_len,
        sequence_length,
    )?;

    Ok(PreparedForward {
        joint_hidden_states,
        rope_cosines,
        rope_sines,
        segments,
        segment_masks,
        attention_scale: attention_scale(HEAD_WIDTH)?,
        attention_one_plus_scale,
        attention_gate,
        feed_forward_one_plus_scale,
        feed_forward_gate,
        head_one_plus_scale,
        prefix_len,
        batch,
        sequence_length,
    })
}

/// The output head: `layer_norm(x) * (1 + scale) → proj_out`, sliced to the target tokens.
pub(super) fn forward_output_head(
    runtime: &MlxRuntime,
    weights: &QwenImage21TransformerWeights,
    prepared: &PreparedForward,
    hidden_states: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let normalized = crate::qwen_image_21::mlx_math::fp32_layer_norm(
        runtime,
        hidden_states,
        LAYER_NORM_EPSILON,
    )?;
    let scaled = runtime.multiply(&normalized, &prepared.head_one_plus_scale)?;
    let projected = weights.proj_out.forward(runtime, &scaled)?;
    Ok(runtime.slice(
        &projected,
        &[0, prepared.prefix_len as i32, 0],
        &[
            prepared.batch as i32,
            prepared.sequence_length as i32,
            LATENT_CHANNEL_COUNT as i32,
        ],
        &[1, 1, 1],
    )?)
}

fn validate_request(
    request: &QwenImage21TransformerRequest<'_>,
) -> Result<usize, QwenImage21EngineError> {
    let latent_shape = request.packed_latents.shape();
    let text_shape = request.text_embeddings.shape();
    if latent_shape.len() != 3 || latent_shape[2] != LATENT_CHANNEL_COUNT as i32 {
        return Err(QwenImage21EngineError::InvalidInput {
            description: format!(
                "packed latents must be (batch, tokens, {LATENT_CHANNEL_COUNT}), received {latent_shape:?}"
            ),
        });
    }
    if text_shape.len() != 3 || text_shape[2] != CONTEXT_INPUT_WIDTH as i32 {
        return Err(QwenImage21EngineError::InvalidInput {
            description: format!(
                "text embeddings must be (batch, tokens, {CONTEXT_INPUT_WIDTH}), received {text_shape:?}"
            ),
        });
    }
    let batch = latent_shape[0] as usize;
    if text_shape[0] != latent_shape[0] {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "latents and text embeddings must share one batch".to_owned(),
        });
    }
    if text_shape[1] as usize != request.vlm_image_mask.len() {
        return Err(QwenImage21EngineError::InvalidInput {
            description: format!(
                "text embeddings have {} tokens but the image-slot mask covers {}",
                text_shape[1],
                request.vlm_image_mask.len()
            ),
        });
    }
    if request.timesteps.len() != batch {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "one timestep per batch sample is required".to_owned(),
        });
    }
    let image_token_count: usize = request
        .img_shapes
        .iter()
        .map(|&(height, width)| height * width)
        .sum();
    if latent_shape[1] as usize != image_token_count {
        return Err(QwenImage21EngineError::InvalidInput {
            description: format!(
                "packed latents hold {} tokens but img_shapes accounts for {image_token_count}",
                latent_shape[1]
            ),
        });
    }
    Ok(batch)
}

fn assemble_joint_sequence(
    runtime: &MlxRuntime,
    image_hidden: &MlxArray,
    text_hidden: &MlxArray,
    slot_mask: &[bool],
    batch: usize,
    image_token_count: usize,
) -> Result<MlxArray, QwenImage21EngineError> {
    let hidden_width = HIDDEN_WIDTH as i32;
    let mut pieces = Vec::new();
    let mut latent_cursor = 0usize;
    let mut index = 0usize;
    while index < slot_mask.len() {
        let is_image = slot_mask[index];
        let run_start = index;
        while index < slot_mask.len() && slot_mask[index] == is_image {
            index += 1;
        }
        let run_length = index - run_start;
        if is_image {
            let joint_tokens = run_length * IMG_TOKENS_PER_SLOT;
            let piece = runtime.slice(
                image_hidden,
                &[0, latent_cursor as i32, 0],
                &[
                    batch as i32,
                    (latent_cursor + joint_tokens) as i32,
                    hidden_width,
                ],
                &[1, 1, 1],
            )?;
            pieces.push(piece);
            latent_cursor += joint_tokens;
        } else {
            let piece = runtime.slice(
                text_hidden,
                &[0, run_start as i32, 0],
                &[batch as i32, (run_start + run_length) as i32, hidden_width],
                &[1, 1, 1],
            )?;
            pieces.push(piece);
        }
    }
    if latent_cursor != image_token_count {
        return Err(QwenImage21EngineError::InvalidInput {
            description: format!(
                "the slot mask expands to {latent_cursor} image tokens but the packed latents hold {image_token_count}"
            ),
        });
    }
    let piece_refs = pieces.iter().collect::<Vec<_>>();
    Ok(runtime.concatenate_axis(&piece_refs, 1)?)
}

/// The reference `_select_modulation_rows`: target tokens read their sample's row, text and
/// condition-image tokens read the trailing `t = 0` row. The target is the trailing contiguous
/// run, so the selection is one concatenate of two broadcasts.
fn selected_modulation_part(
    runtime: &MlxRuntime,
    modulation: &MlxArray,
    column_start: usize,
    column_end: usize,
    batch: usize,
    prefix_len: usize,
    sequence_length: usize,
) -> Result<MlxArray, QwenImage21EngineError> {
    let width = (column_end - column_start) as i32;
    let modulation_rows = modulation.shape()[0];
    let part = runtime.slice(
        modulation,
        &[0, column_start as i32],
        &[modulation_rows, column_end as i32],
        &[1, 1],
    )?;
    let real = runtime.slice(&part, &[0, 0], &[batch as i32, width], &[1, 1])?;
    let zero = runtime.slice(
        &part,
        &[batch as i32, 0],
        &[modulation_rows, width],
        &[1, 1],
    )?;
    let zero_rows = runtime.broadcast_to(
        &runtime.expand_dims(&zero, 1)?,
        &[batch as i32, prefix_len as i32, width],
    )?;
    let target_tokens = sequence_length - prefix_len;
    let real_rows = runtime.broadcast_to(
        &runtime.expand_dims(&real, 1)?,
        &[batch as i32, target_tokens as i32, width],
    )?;
    Ok(runtime.concatenate_axis(&[&zero_rows, &real_rows], 1)?)
}

fn selected_one_plus_scale(
    runtime: &MlxRuntime,
    modulation: &MlxArray,
    column_start: usize,
    column_end: usize,
    batch: usize,
    prefix_len: usize,
    sequence_length: usize,
) -> Result<MlxArray, QwenImage21EngineError> {
    let selected = selected_modulation_part(
        runtime,
        modulation,
        column_start,
        column_end,
        batch,
        prefix_len,
        sequence_length,
    )?;
    let unit = runtime.full(&[], 1.0, selected.dtype())?;
    Ok(runtime.add(&selected, &unit)?)
}

/// Additive `0 / -inf` masks for the text segments: full visibility of everything before the
/// segment and a causal triangle inside it. Image segments need no mask (fused attention).
fn build_segment_masks(
    runtime: &MlxRuntime,
    segments: &[AttentionSegment],
) -> Result<Vec<MlxArray>, QwenImage21EngineError> {
    let mut masks = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.is_text {
            let segment_length = segment.query_end - segment.query_start;
            let key_end = segment.key_end;
            let mut mask = Vec::with_capacity(segment_length * key_end);
            for query_offset in 0..segment_length {
                let query_position = segment.query_start + query_offset;
                for key_position in 0..key_end {
                    let allowed = key_position <= query_position;
                    mask.push(if allowed { 0.0_f32 } else { f32::NEG_INFINITY });
                }
            }
            masks.push(
                runtime.array_from_f32(&mask, &[1, 1, segment_length as i32, key_end as i32])?,
            );
        } else {
            // Placeholder aligned with the segment list; fused attention ignores it.
            masks.push(runtime.array_from_f32(&[0.0], &[1, 1, 1, 1])?);
        }
    }
    Ok(masks)
}
