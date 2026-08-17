//! Descriptor-driven Laguna attention using the family-neutral primitives.
//!
//! Laguna owns projection binding, per-layer geometry, rotary policy, cache
//! selection, and optional output gating here. The underlying attention masks
//! and cache mechanisms remain neutral so another model family does not have to
//! import Laguna policy to reuse identical mathematics.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::attention::build_causal_sliding_window_mask;
use crate::laguna::artifacts::{LagunaAttentionProjection, LagunaLayerTensorRole};
use crate::laguna::normalization::{
    LagunaAttentionDescriptor, LagunaAttentionKind, LagunaCacheDescriptor, LagunaGatingKind,
};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::bound_linear::LagunaBoundLinear;
use super::decoder_state::LagunaDecoderState;
use super::error::LagunaExecutionError;
use super::rope_application::apply_layer_rope;
use super::weights::LagunaNativeWeights;

/// Reuses the two causal masks shared by equivalent attention layers during
/// one model forward. The cache is intentionally forward-scoped because its
/// arrays describe that forward's query and retained key positions only.
#[derive(Default)]
pub(super) struct LagunaAttentionMaskCache {
    entries: Vec<LagunaAttentionMaskEntry>,
}

struct LagunaAttentionMaskEntry {
    key: LagunaAttentionMaskKey,
    mask: MlxArray,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct LagunaAttentionMaskKey {
    first_query_absolute_position: i32,
    query_token_count: i32,
    first_key_absolute_position: i32,
    key_token_count: i32,
    window_size: i32,
}

impl LagunaAttentionMaskCache {
    fn mask_for<'cache>(
        &'cache mut self,
        runtime: &MlxRuntime,
        key: LagunaAttentionMaskKey,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<&'cache MlxArray, LagunaExecutionError> {
        if let Some(entry_index) = self.entries.iter().position(|entry| entry.key == key) {
            return Ok(&self.entries[entry_index].mask);
        }
        let mask = build_causal_sliding_window_mask(
            runtime,
            key.first_query_absolute_position,
            key.query_token_count,
            key.first_key_absolute_position,
            key.key_token_count,
            key.window_size,
            performance_attribution,
        )?;
        let inserted_entry_index = self.entries.len();
        self.entries.push(LagunaAttentionMaskEntry { key, mask });
        Ok(&self.entries[inserted_entry_index].mask)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    attention: &LagunaAttentionDescriptor,
    layer_index: usize,
    decoder_state: &mut LagunaDecoderState,
    attention_mask_cache: &mut LagunaAttentionMaskCache,
    rms_norm_epsilon: f32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    performance_attribution.measure_operation(
        PerformanceOperation::AttentionForwardSpan,
        |attribution| {
            forward_attention_inner(
                runtime,
                hidden_states,
                weights,
                attention,
                layer_index,
                decoder_state,
                attention_mask_cache,
                rms_norm_epsilon,
                attribution,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_attention_inner(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    attention: &LagunaAttentionDescriptor,
    layer_index: usize,
    decoder_state: &mut LagunaDecoderState,
    attention_mask_cache: &mut LagunaAttentionMaskCache,
    rms_norm_epsilon: f32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let hidden_shape = hidden_states.shape();
    if hidden_shape.len() != 3 {
        return Err(LagunaExecutionError::invalid_geometry(
            "Laguna hidden states must have rank three",
        ));
    }
    let batch_size = hidden_shape[0];
    let token_count = hidden_shape[1];
    let query_head_count = attention.query_head_count() as i32;
    let key_value_head_count = attention.key_value_head_count() as i32;
    let head_dimension = attention.head_dimension() as i32;
    let queries = project_heads(
        runtime,
        hidden_states,
        weights.linear(
            layer_index,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
        )?,
        batch_size,
        token_count,
        query_head_count,
        head_dimension,
    )?;
    let keys = project_heads(
        runtime,
        hidden_states,
        weights.linear(
            layer_index,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
        )?,
        batch_size,
        token_count,
        key_value_head_count,
        head_dimension,
    )?;
    let values = project_heads(
        runtime,
        hidden_states,
        weights.linear(
            layer_index,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
        )?,
        batch_size,
        token_count,
        key_value_head_count,
        head_dimension,
    )?;
    let normalized_queries = runtime.rms_norm(
        &queries,
        weights.layer(
            layer_index,
            LagunaLayerTensorRole::AttentionQueryNormalization,
        )?,
        rms_norm_epsilon,
    )?;
    let normalized_keys = runtime.rms_norm(
        &keys,
        weights.layer(
            layer_index,
            LagunaLayerTensorRole::AttentionKeyNormalization,
        )?,
        rms_norm_epsilon,
    )?;
    let transposed_queries = runtime.transpose_axes(&normalized_queries, &[0, 2, 1, 3])?;
    let transposed_keys = runtime.transpose_axes(&normalized_keys, &[0, 2, 1, 3])?;
    let transposed_values = runtime.transpose_axes(&values, &[0, 2, 1, 3])?;
    let rope_offset = decoder_state.absolute_position(layer_index).unwrap_or(0);
    let rotated_queries = apply_layer_rope(
        runtime,
        &transposed_queries,
        attention.rope(),
        rope_offset,
        performance_attribution,
    )?;
    let rotated_keys = apply_layer_rope(
        runtime,
        &transposed_keys,
        attention.rope(),
        rope_offset,
        performance_attribution,
    )?;
    let (active_keys, active_values, _) = decoder_state.update_and_fetch(
        runtime,
        layer_index,
        &rotated_keys,
        &transposed_values,
        performance_attribution,
    )?;
    let attention_scale = (head_dimension as f32).sqrt().recip();
    let attention_output = attend(
        runtime,
        &rotated_queries,
        &active_keys,
        &active_values,
        attention,
        token_count,
        rope_offset,
        attention_scale,
        attention_mask_cache,
        performance_attribution,
    )?;
    let token_major = runtime.transpose_axes(&attention_output, &[0, 2, 1, 3])?;
    let flattened = runtime.reshape(
        &token_major,
        &[batch_size, token_count, query_head_count * head_dimension],
    )?;
    let gated = apply_output_gate(
        runtime,
        hidden_states,
        &flattened,
        weights,
        attention,
        layer_index,
        query_head_count,
        head_dimension,
        performance_attribution,
    )?;
    weights
        .linear(
            layer_index,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
        )?
        .project(runtime, &gated)
}

fn attend(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    attention: &LagunaAttentionDescriptor,
    query_token_count: i32,
    first_query_absolute_position: i32,
    attention_scale: f32,
    attention_mask_cache: &mut LagunaAttentionMaskCache,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    if query_token_count <= 1 {
        return Ok(runtime.scaled_dot_product_attention(queries, keys, values, attention_scale)?);
    }
    if attention.kind() == LagunaAttentionKind::Full {
        // MLX's native causal mode derives the cached-prefix offset as
        // `key_length - query_length`. Avoid supplying an equivalent array
        // mask, which MLX otherwise broadcasts across every query head.
        return Ok(runtime.causal_scaled_dot_product_attention(
            queries,
            keys,
            values,
            attention_scale,
        )?);
    }
    let LagunaCacheDescriptor::Rotating { window_size } = attention.cache() else {
        return Err(LagunaExecutionError::invalid_geometry(
            "sliding Laguna attention requires a rotating cache descriptor",
        ));
    };
    attend_sliding_in_window_sized_query_blocks(
        runtime,
        queries,
        keys,
        values,
        attention_scale,
        first_query_absolute_position,
        i32::try_from(*window_size).map_err(|_| {
            LagunaExecutionError::invalid_geometry("sliding attention window exceeds i32")
        })?,
        attention_mask_cache,
        performance_attribution,
    )
}

#[allow(clippy::too_many_arguments)]
fn attend_sliding_in_window_sized_query_blocks(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    attention_scale: f32,
    first_query_absolute_position: i32,
    window_size: i32,
    attention_mask_cache: &mut LagunaAttentionMaskCache,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    if query_shape.len() != 4 || key_shape.len() != 4 || window_size <= 0 {
        return Err(LagunaExecutionError::invalid_geometry(
            "sliding attention requires rank-four arrays and a positive window",
        ));
    }
    let query_token_count = query_shape[2];
    let key_token_count = key_shape[2];
    let first_key_absolute_position = first_query_absolute_position
        .checked_add(query_token_count)
        .and_then(|query_end| query_end.checked_sub(key_token_count))
        .unwrap_or(0)
        .max(0);
    let mut query_block_start = 0_i32;
    let mut block_outputs = Vec::new();
    while query_block_start < query_token_count {
        let query_block_end = query_block_start
            .saturating_add(window_size)
            .min(query_token_count);
        let query_block_token_count = query_block_end - query_block_start;
        let query_block_first_absolute_position = first_query_absolute_position
            .checked_add(query_block_start)
            .ok_or_else(|| {
                LagunaExecutionError::invalid_geometry(
                    "sliding query block absolute position overflowed",
                )
            })?;
        let required_first_key_absolute_position = query_block_first_absolute_position
            .saturating_sub(window_size.saturating_sub(1))
            .max(first_key_absolute_position);
        let required_key_end_absolute_position = query_block_first_absolute_position
            .checked_add(query_block_token_count)
            .ok_or_else(|| {
                LagunaExecutionError::invalid_geometry(
                    "sliding query block key boundary overflowed",
                )
            })?;
        let key_block_start =
            required_first_key_absolute_position.saturating_sub(first_key_absolute_position);
        let key_block_end = required_key_end_absolute_position
            .saturating_sub(first_key_absolute_position)
            .min(key_token_count);
        let query_block = runtime.slice(
            queries,
            &[0, 0, query_block_start, 0],
            &[
                query_shape[0],
                query_shape[1],
                query_block_end,
                query_shape[3],
            ],
            &[1, 1, 1, 1],
        )?;
        let key_block = runtime.slice(
            keys,
            &[0, 0, key_block_start, 0],
            &[key_shape[0], key_shape[1], key_block_end, key_shape[3]],
            &[1, 1, 1, 1],
        )?;
        let value_block = runtime.slice(
            values,
            &[0, 0, key_block_start, 0],
            &[key_shape[0], key_shape[1], key_block_end, key_shape[3]],
            &[1, 1, 1, 1],
        )?;
        let mask_key = LagunaAttentionMaskKey {
            first_query_absolute_position: query_block_first_absolute_position,
            query_token_count: query_block_token_count,
            first_key_absolute_position: required_first_key_absolute_position,
            key_token_count: key_block_end - key_block_start,
            window_size,
        };
        let mask = attention_mask_cache.mask_for(runtime, mask_key, performance_attribution)?;
        block_outputs.push(runtime.masked_scaled_dot_product_attention(
            &query_block,
            &key_block,
            &value_block,
            attention_scale,
            mask,
        )?);
        query_block_start = query_block_end;
    }
    if block_outputs.len() == 1 {
        return block_outputs.pop().ok_or_else(|| {
            LagunaExecutionError::invalid_geometry("sliding attention output is missing")
        });
    }
    let output_references = block_outputs.iter().collect::<Vec<_>>();
    Ok(runtime.concatenate_axis(&output_references, 2)?)
}

fn project_heads(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weight: &LagunaBoundLinear,
    batch_size: i32,
    token_count: i32,
    head_count: i32,
    head_dimension: i32,
) -> Result<MlxArray, LagunaExecutionError> {
    let projected = weight.project(runtime, hidden_states)?;
    Ok(runtime.reshape(
        &projected,
        &[batch_size, token_count, head_count, head_dimension],
    )?)
}

fn apply_output_gate(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    attention_output: &MlxArray,
    weights: &LagunaNativeWeights,
    attention: &LagunaAttentionDescriptor,
    layer_index: usize,
    query_head_count: i32,
    head_dimension: i32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    match attention.gating_kind() {
        LagunaGatingKind::None => Ok(attention_output.retain()?),
        // The projection exists only to apply the optional gate, so keep its
        // cost inside the same boundary. Timing only softplus would understate
        // the user-visible cost of selecting a gated descriptor.
        gating_kind => performance_attribution.measure_operation(
            PerformanceOperation::SoftplusAttentionGateApplication,
            |_| {
                let gate_logits = weights
                    .linear(
                        layer_index,
                        LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                    )?
                    .project(runtime, hidden_states)?;
                let hidden_shape = hidden_states.shape();
                let shaped_gate = match gating_kind {
                    LagunaGatingKind::PerHead => runtime.reshape(
                        &gate_logits,
                        &[hidden_shape[0], hidden_shape[1], query_head_count, 1],
                    )?,
                    LagunaGatingKind::PerElement => runtime.reshape(
                        &gate_logits,
                        &[
                            hidden_shape[0],
                            hidden_shape[1],
                            query_head_count,
                            head_dimension,
                        ],
                    )?,
                    LagunaGatingKind::None => gate_logits,
                };
                let shaped_output = runtime.reshape(
                    attention_output,
                    &[
                        hidden_shape[0],
                        hidden_shape[1],
                        query_head_count,
                        head_dimension,
                    ],
                )?;
                let gated = runtime.apply_softplus_attention_gate(&shaped_output, &shaped_gate)?;
                Ok(runtime.reshape(&gated, &attention_output.shape())?)
            },
        ),
    }
}
