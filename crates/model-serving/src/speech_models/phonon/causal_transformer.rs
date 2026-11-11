//! The causal transformer Phonon uses twice: as the flow model's backbone
//! (six layers, full history, fed one frame at a time) and inside the Mimi
//! decoder (two layers with layer scale, each position seeing only the last
//! 250 positions, fed in blocks).
//!
//! Layers are pre-norm: `x + attention(norm1(x))`, then
//! `x + feed_forward(norm2(x))`, each branch optionally scaled per channel.
//! Attention projections and the feed-forward have no bias, and the
//! feed-forward uses the exact (erf) GELU, as the Phonon configuration names.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::attention_history::AttentionHistory;
use super::causal_attention_mask::{causal_mask_values, single_query_sees_every_key};
use super::rotary_embedding::rotate_interleaved_pairs;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::layer_norm::LayerNorm;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

const TRANSFORMER_NORM_EPSILON: f32 = 1e-5;

struct TransformerLayer {
    attention_norm: LayerNorm,
    query_key_value: Linear,
    attention_output: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    /// Per-channel scales of the attention and feed-forward branches (Mimi only).
    branch_scales: Option<BranchScales>,
}

struct BranchScales {
    attention_scale: MlxArray,
    feed_forward_scale: MlxArray,
}

/// Where a transformer's weights live and how it attends.
pub(crate) struct TransformerShape<'prefix> {
    pub(crate) prefix: &'prefix str,
    pub(crate) layer_count: usize,
    pub(crate) head_count: i32,
    pub(crate) rotary_max_period: f32,
    /// `None` for full history; otherwise how many positions, the current one
    /// included, each position may attend to.
    pub(crate) attention_window: Option<usize>,
    pub(crate) has_branch_scales: bool,
}

pub(crate) struct CausalTransformer {
    layers: Vec<TransformerLayer>,
    head_count: i32,
    rotary_max_period: f32,
    attention_window: Option<usize>,
}

/// One forward call's view of where the new positions sit.
struct PositionContext<'mask> {
    absolute_first_position: usize,
    mask: Option<&'mask MlxArray>,
}

impl CausalTransformer {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        shape: &TransformerShape<'_>,
    ) -> Result<Self, SpeechModelError> {
        let mut layers = Vec::with_capacity(shape.layer_count);
        for layer_index in 0..shape.layer_count {
            let layer_prefix = format!("{}.layers.{layer_index}", shape.prefix);
            layers.push(TransformerLayer::load(
                checkpoint,
                &layer_prefix,
                shape.has_branch_scales,
            )?);
        }
        Ok(Self {
            layers,
            head_count: shape.head_count,
            rotary_max_period: shape.rotary_max_period,
            attention_window: shape.attention_window,
        })
    }

    pub(crate) fn empty_history(&self) -> AttentionHistory {
        AttentionHistory::empty(self.layers.len())
    }

    /// `[new positions, width]` in, the same shape out. The history grows by
    /// the new positions, and a windowed transformer then forgets keys no
    /// later position can see.
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        history: &mut AttentionHistory,
    ) -> Result<MlxArray, SpeechModelError> {
        let new_count = input.shape()[0] as usize;
        let mask = self.attention_mask(runtime, history.stored_count(), new_count)?;
        let position_context = PositionContext {
            absolute_first_position: history.absolute_position_count(),
            mask: mask.as_ref(),
        };
        let mut states = input.retain()?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            states = self.layer_forward(
                runtime,
                layer,
                layer_index,
                &states,
                history,
                &position_context,
            )?;
        }
        let kept_count = self.attention_window.map(|window| window - 1);
        history.advance(runtime, new_count, kept_count)?;
        Ok(states)
    }

    fn layer_forward(
        &self,
        runtime: &MlxRuntime,
        layer: &TransformerLayer,
        layer_index: usize,
        states: &MlxArray,
        history: &mut AttentionHistory,
        position_context: &PositionContext<'_>,
    ) -> Result<MlxArray, SpeechModelError> {
        let normalized = layer.attention_norm.forward(runtime, states)?;
        let attended = self.attend(
            runtime,
            layer,
            layer_index,
            &normalized,
            history,
            position_context,
        )?;
        let attention_scale = layer
            .branch_scales
            .as_ref()
            .map(|scales| &scales.attention_scale);
        let after_attention =
            runtime.add(states, &scaled_branch(runtime, &attended, attention_scale)?)?;
        let feed_forward_input = layer.feed_forward_norm.forward(runtime, &after_attention)?;
        let expanded = runtime.gelu(
            &layer
                .feed_forward_in
                .forward(runtime, &feed_forward_input)?,
        )?;
        let fed_forward = layer.feed_forward_out.forward(runtime, &expanded)?;
        let feed_forward_scale = layer
            .branch_scales
            .as_ref()
            .map(|scales| &scales.feed_forward_scale);
        Ok(runtime.add(
            &after_attention,
            &scaled_branch(runtime, &fed_forward, feed_forward_scale)?,
        )?)
    }

    fn attend(
        &self,
        runtime: &MlxRuntime,
        layer: &TransformerLayer,
        layer_index: usize,
        normalized: &MlxArray,
        history: &mut AttentionHistory,
        position_context: &PositionContext<'_>,
    ) -> Result<MlxArray, SpeechModelError> {
        let [length, width] = normalized.shape()[..] else {
            return Err(SpeechModelError::Input(
                "transformer expects [length, width]".to_owned(),
            ));
        };
        let (queries, new_keys, new_values) = self.rotated_queries_and_keys(
            runtime,
            layer,
            normalized,
            position_context.absolute_first_position,
        )?;
        let (keys, values) = history.append(runtime, layer_index, new_keys, new_values)?;
        let attended_heads =
            scaled_dot_product(runtime, &queries, &keys, &values, position_context.mask)?;
        let heads_merged = runtime.reshape(
            &runtime.transpose_axes(&attended_heads, &[1, 0, 2])?,
            &[length, width],
        )?;
        layer.attention_output.forward(runtime, &heads_merged)
    }

    /// Query, key and value heads of `normalized`, `[heads, length, head
    /// width]` each, with queries and keys rotated to their absolute positions.
    fn rotated_queries_and_keys(
        &self,
        runtime: &MlxRuntime,
        layer: &TransformerLayer,
        normalized: &MlxArray,
        first_position: usize,
    ) -> Result<(MlxArray, MlxArray, MlxArray), SpeechModelError> {
        let [length, width] = normalized.shape()[..] else {
            return Err(SpeechModelError::Input(
                "transformer expects [length, width]".to_owned(),
            ));
        };
        let head_width = width / self.head_count;
        let packed_projections = layer.query_key_value.forward(runtime, normalized)?;
        let per_projection_heads = runtime.reshape(
            &packed_projections,
            &[length, 3, self.head_count, head_width],
        )?;
        let projection_first = runtime.transpose_axes(&per_projection_heads, &[1, 2, 0, 3])?;
        let queries = self.rotated(
            runtime,
            &self.projection(runtime, &projection_first, 0)?,
            first_position,
        )?;
        let keys = self.rotated(
            runtime,
            &self.projection(runtime, &projection_first, 1)?,
            first_position,
        )?;
        let values = self.projection(runtime, &projection_first, 2)?;
        Ok((queries, keys, values))
    }

    /// Query, key or value heads, `[heads, length, head width]`.
    fn projection(
        &self,
        runtime: &MlxRuntime,
        projection_first: &MlxArray,
        projection_index: i32,
    ) -> Result<MlxArray, SpeechModelError> {
        let [_, head_count, length, head_width] = projection_first.shape()[..] else {
            return Err(SpeechModelError::Input(
                "projections expect [3, heads, length, width]".to_owned(),
            ));
        };
        let selected = runtime.slice(
            projection_first,
            &[projection_index, 0, 0, 0],
            &[projection_index + 1, head_count, length, head_width],
            &[1, 1, 1, 1],
        )?;
        Ok(runtime.squeeze_axis(&selected, 0)?)
    }

    fn rotated(
        &self,
        runtime: &MlxRuntime,
        heads: &MlxArray,
        first_position: usize,
    ) -> Result<MlxArray, SpeechModelError> {
        rotate_interleaved_pairs(runtime, heads, first_position, self.rotary_max_period)
    }

    /// `None` when nothing is hidden: one new query that may see every key.
    fn attention_mask(
        &self,
        runtime: &MlxRuntime,
        stored_count: usize,
        new_count: usize,
    ) -> Result<Option<MlxArray>, SpeechModelError> {
        if new_count == 1 && single_query_sees_every_key(stored_count, self.attention_window) {
            return Ok(None);
        }
        let mask_values = causal_mask_values(stored_count, new_count, self.attention_window);
        let key_count = (stored_count + new_count) as i32;
        Ok(Some(runtime.array_from_f32(
            &mask_values,
            &[1, new_count as i32, key_count],
        )?))
    }
}

impl TransformerLayer {
    fn load(
        checkpoint: &CheckpointTensors<'_>,
        layer_prefix: &str,
        has_branch_scales: bool,
    ) -> Result<Self, SpeechModelError> {
        let branch_scales = if has_branch_scales {
            Some(BranchScales {
                attention_scale: checkpoint
                    .tensor(&format!("{layer_prefix}.layer_scale_1.scale"))?,
                feed_forward_scale: checkpoint
                    .tensor(&format!("{layer_prefix}.layer_scale_2.scale"))?,
            })
        } else {
            None
        };
        Ok(Self {
            attention_norm: LayerNorm::load(
                checkpoint,
                &format!("{layer_prefix}.norm1"),
                TRANSFORMER_NORM_EPSILON,
            )?,
            query_key_value: Linear::load(
                checkpoint,
                &format!("{layer_prefix}.self_attn.in_proj"),
            )?,
            attention_output: Linear::load(
                checkpoint,
                &format!("{layer_prefix}.self_attn.out_proj"),
            )?,
            feed_forward_norm: LayerNorm::load(
                checkpoint,
                &format!("{layer_prefix}.norm2"),
                TRANSFORMER_NORM_EPSILON,
            )?,
            feed_forward_in: Linear::load(checkpoint, &format!("{layer_prefix}.linear1"))?,
            feed_forward_out: Linear::load(checkpoint, &format!("{layer_prefix}.linear2"))?,
            branch_scales,
        })
    }
}

/// `softmax(q k^T / sqrt(d) + mask) v` over `[heads, length, head width]`.
fn scaled_dot_product(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    mask: Option<&MlxArray>,
) -> Result<MlxArray, SpeechModelError> {
    let head_width = queries.shape()[2];
    let scale = 1.0 / (head_width as f32).sqrt();
    let scores = runtime.multiply_scalar(
        &runtime.matmul(queries, &runtime.transpose_axes(keys, &[0, 2, 1])?)?,
        scale,
    )?;
    let masked_scores = match mask {
        Some(mask) => runtime.add(&scores, mask)?,
        None => scores,
    };
    let attention_weights = runtime.softmax_axis(&masked_scores, -1)?;
    Ok(runtime.matmul(&attention_weights, values)?)
}

fn scaled_branch(
    runtime: &MlxRuntime,
    branch_output: &MlxArray,
    scale: Option<&MlxArray>,
) -> Result<MlxArray, SpeechModelError> {
    match scale {
        Some(scale) => Ok(runtime.multiply(branch_output, scale)?),
        None => Ok(branch_output.retain()?),
    }
}

#[cfg(test)]
#[path = "causal_transformer_tests.rs"]
mod tests;
