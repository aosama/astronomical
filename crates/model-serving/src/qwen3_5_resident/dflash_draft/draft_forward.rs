//! The DFlash draft trunk forward: one parallel pass over a speculation block.
//!
//! The block is `[bonus token, mask token × speculation count]`. Layer zero
//! consumes the block's embeddings (looked up from the target's table); each
//! layer projects its own query/key/value for the block tokens, appends the
//! block's keys and values into its key-value state beside the target-derived
//! context entries, and attends according to its layer kind (causal windowed
//! or fully bidirectional — the block-diffusion parallelism). The returned
//! normalized hidden states are `[1, block size, hidden]`; the caller scores
//! them with the target's language-model head, which the draft does not ship.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::MlxArray;

use crate::attention::build_causal_sliding_window_mask;
use crate::performance_attribution::PerformanceAttribution;

use super::draft_attention_masks::{DFlashDraftAttentionRoute, attention_route_for};
use super::draft_configuration::{DFlashDraftConfiguration, DFlashDraftLayerAttentionKind};
use super::draft_context_kv_projection::{projected_rotated_keys, projected_values};
use super::draft_key_value_state::DFlashDraftKeyValueStates;
use super::draft_weights::{DFlashDraftLayerWeights, DFlashDraftWeights};

/// Runs the trunk over one block and returns final-normalized hidden states.
///
/// `block_embeddings` must be `[1, block size, hidden]` at absolute positions
/// `first_block_absolute_position ..+ block size` — which must equal every
/// layer's current key-value offset. Every layer's key-value state grows by
/// the block size; the caller rolls back to
/// `first_block_absolute_position` when the verification round rejects a
/// suffix.
pub fn forward_draft_block(
    runtime: &MlxRuntime,
    weights: &DFlashDraftWeights,
    configuration: &DFlashDraftConfiguration,
    key_value_states: &mut DFlashDraftKeyValueStates,
    block_embeddings: &MlxArray,
    first_block_absolute_position: i32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    let block_token_count = block_embeddings.shape()[1];
    let mut hidden_states: MlxArray = block_embeddings.retain()?;
    for (layer_index, layer_weights) in weights.layers.iter().enumerate() {
        let attention_kind = configuration.layer_attention_kinds()[layer_index];
        hidden_states = forward_one_draft_layer(
            runtime,
            layer_weights,
            attention_kind,
            configuration,
            key_value_states.layer_state_mut(layer_index),
            &hidden_states,
            first_block_absolute_position,
            block_token_count,
            performance_attribution,
        )?;
    }
    Ok(runtime.rms_norm(
        &hidden_states,
        &weights.final_normalization,
        configuration.rms_norm_epsilon(),
    )?)
}

#[allow(clippy::too_many_arguments)]
fn forward_one_draft_layer(
    runtime: &MlxRuntime,
    layer_weights: &DFlashDraftLayerWeights,
    attention_kind: DFlashDraftLayerAttentionKind,
    configuration: &DFlashDraftConfiguration,
    layer_key_value_state: Option<&mut crate::decoder_cache::FullAttentionKeyValueState>,
    layer_input: &MlxArray,
    first_block_absolute_position: i32,
    block_token_count: i32,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    let layer_key_value_state = layer_key_value_state.ok_or(MlxRuntimeError::RuntimeOperation {
        operation: "run one DFlash draft layer",
        description: "the draft key-value layer for this trunk layer is missing".to_owned(),
    })?;

    let residual_before_attention = layer_input;
    let normalized_input = runtime.rms_norm(
        residual_before_attention,
        &layer_weights.input_normalization,
        configuration.rms_norm_epsilon(),
    )?;

    let new_keys = projected_rotated_keys(
        runtime,
        layer_weights,
        configuration,
        &normalized_input,
        first_block_absolute_position,
    )?;
    let new_values = projected_values(runtime, layer_weights, configuration, &normalized_input)?;

    let transposed_query_projection =
        runtime.transpose_axes(&layer_weights.query_projection, &[1, 0])?;
    let queries = runtime.matmul(&normalized_input, &transposed_query_projection)?;
    let queries = runtime.reshape(
        &queries,
        &[
            1,
            block_token_count,
            configuration.attention_head_count(),
            configuration.head_dimension(),
        ],
    )?;
    let queries = runtime.rms_norm(
        &queries,
        &layer_weights.query_head_normalization,
        configuration.rms_norm_epsilon(),
    )?;
    let queries = runtime.transpose_axes(&queries, &[0, 2, 1, 3])?;
    let queries = runtime.rope(
        &queries,
        configuration.head_dimension(),
        configuration.rope_theta(),
        first_block_absolute_position,
    )?;

    let (active_keys, active_values) = layer_key_value_state.update_and_fetch(
        runtime,
        &new_keys,
        &new_values,
        first_block_absolute_position,
    )?;

    // After the block append, every layer's key-value view spans the context
    // plus the block: the mask grid's key axis is exactly that length.
    let key_view_token_count = first_block_absolute_position + block_token_count;
    let attention_scale = (configuration.head_dimension() as f32).recip().sqrt();
    let attention_output = match attention_route_for(attention_kind) {
        DFlashDraftAttentionRoute::SlidingWindowMasked { window_size } => {
            let sliding_window_mask = build_causal_sliding_window_mask(
                runtime,
                first_block_absolute_position,
                block_token_count,
                0,
                key_view_token_count,
                window_size,
                performance_attribution,
            )?;
            runtime.masked_scaled_dot_product_attention(
                &queries,
                &active_keys,
                &active_values,
                attention_scale,
                &sliding_window_mask,
            )?
        }
        DFlashDraftAttentionRoute::Unmasked => runtime.scaled_dot_product_attention(
            &queries,
            &active_keys,
            &active_values,
            attention_scale,
        )?,
    };
    let attention_output = runtime.transpose_axes(&attention_output, &[0, 2, 1, 3])?;
    let attention_output = runtime.reshape(
        &attention_output,
        &[
            1,
            block_token_count,
            configuration.attention_head_count() * configuration.head_dimension(),
        ],
    )?;
    let transposed_output_projection =
        runtime.transpose_axes(&layer_weights.output_projection, &[1, 0])?;
    let attended_hidden = runtime.matmul(&attention_output, &transposed_output_projection)?;
    let hidden_after_attention = runtime.add(residual_before_attention, &attended_hidden)?;

    let residual_before_mlp = &hidden_after_attention;
    let normalized_hidden = runtime.rms_norm(
        residual_before_mlp,
        &layer_weights.post_attention_normalization,
        configuration.rms_norm_epsilon(),
    )?;
    let transposed_gate_projection =
        runtime.transpose_axes(&layer_weights.gate_projection, &[1, 0])?;
    let gate = runtime.matmul(&normalized_hidden, &transposed_gate_projection)?;
    let gated = runtime.silu(&gate)?;
    let transposed_up_projection = runtime.transpose_axes(&layer_weights.up_projection, &[1, 0])?;
    let up = runtime.matmul(&normalized_hidden, &transposed_up_projection)?;
    let activated = runtime.multiply(&gated, &up)?;
    let transposed_down_projection =
        runtime.transpose_axes(&layer_weights.down_projection, &[1, 0])?;
    let down = runtime.matmul(&activated, &transposed_down_projection)?;
    Ok(runtime.add(residual_before_mlp, &down)?)
}
