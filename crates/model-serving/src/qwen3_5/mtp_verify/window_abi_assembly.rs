//! Assembling the verification window's frozen compile-input vector and
//! reading its ordered outputs.
//!
//! The unsafe graph-builder half lives in runtime-integration; this module is
//! its safe model-side counterpart. Assembly walks the runtime's frozen slot
//! plan and resolves every position against the loaded model's weights and
//! this attempt's state leaves, so the two sides of the ABI cannot disagree.
//! Output reading mirrors the frozen output order: all-position logits first,
//! then per decoder layer — gated-delta layers contribute `[next rolling
//! convolution, next recurrent state]`, full-attention layers contribute
//! `[rotated new keys, new values]`.
//!
use astronomical_runtime_integration::{
    VerifyWindowAffineSlot, VerifyWindowFeedForwardWeightSlot, VerifyWindowFullAttentionWeightSlot,
    VerifyWindowGatedDeltaWeightSlot, VerifyWindowGeometry, VerifyWindowInputSlot,
    VerifyWindowLayerWeightSlot, VerifyWindowTrunkWeightSlot, verify_window_input_slots,
};

use crate::qwen3_5::model::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderLayerWeights,
};

use super::window_state_leaves::{WindowStateLeaves, WindowStateUpdate};
use astronomical_mlx_c_rust::MlxArray;

/// Everything one apply needs beside the model's resident weights.
pub(crate) struct VerifyWindowAttemptInputs<'a> {
    pub(crate) token_indices: &'a MlxArray,
    pub(crate) position_offsets: &'a MlxArray,
    pub(crate) key_value_base_offset: &'a MlxArray,
    pub(crate) query_normalization_scale: &'a MlxArray,
    pub(crate) key_normalization_scale: &'a MlxArray,
    pub(crate) state_leaves: &'a [WindowStateLeaves],
}

/// One apply's typed results before they are installed into live state.
pub(crate) struct VerifyWindowAttemptOutputs {
    pub(crate) all_position_logits: MlxArray,
    pub(crate) pre_final_normalization_hidden_states: MlxArray,
    pub(crate) layer_updates: Vec<WindowStateUpdate>,
}

/// Assembles the flat input vector in the frozen slot order.
pub(crate) fn assemble_window_inputs(
    geometry: &VerifyWindowGeometry,
    decoder_layer_weights: &[Qwen3_5DecoderLayerWeights],
    embedding_weights: &Qwen3_5AffineWeights,
    final_normalization_weight: &MlxArray,
    language_model_head_weights: &Qwen3_5AffineWeights,
    attempt_inputs: &VerifyWindowAttemptInputs<'_>,
) -> Result<Vec<MlxArray>, String> {
    let slot_plan = verify_window_input_slots(geometry);
    let mut input_vector = Vec::with_capacity(slot_plan.len());
    for slot in slot_plan {
        let resolved = match slot {
            VerifyWindowInputSlot::TokenIndices => {
                attempt_inputs.token_indices.retain().map_err(|e| e.to_string())
            }
            VerifyWindowInputSlot::PositionOffsets => {
                attempt_inputs.position_offsets.retain().map_err(|e| e.to_string())
            }
            VerifyWindowInputSlot::KeyValueBaseOffset => attempt_inputs
                .key_value_base_offset
                .retain()
                .map_err(|e| e.to_string()),
            VerifyWindowInputSlot::QueryNormalizationScale => attempt_inputs
                .query_normalization_scale
                .retain()
                .map_err(|e| e.to_string()),
            VerifyWindowInputSlot::KeyNormalizationScale => attempt_inputs
                .key_normalization_scale
                .retain()
                .map_err(|e| e.to_string()),
            VerifyWindowInputSlot::GatedDeltaRollingState { layer_index } => match attempt_inputs
                .state_leaves
                .get(layer_index)
            {
                Some(WindowStateLeaves::GatedDelta { convolution, .. }) => {
                    convolution.retain().map_err(|e| e.to_string())
                }
                _ => Err(format!(
                    "layer {layer_index} has no extracted rolling-convolution leaf"
                )),
            },
            VerifyWindowInputSlot::GatedDeltaRecurrentState { layer_index } => {
                match attempt_inputs.state_leaves.get(layer_index) {
                    Some(WindowStateLeaves::GatedDelta { recurrent, .. }) => {
                        recurrent.retain().map_err(|e| e.to_string())
                    }
                    _ => Err(format!(
                        "layer {layer_index} has no extracted recurrent-state leaf"
                    )),
                }
            }
            VerifyWindowInputSlot::FullAttentionKeysSlab { layer_index } => match attempt_inputs
                .state_leaves
                .get(layer_index)
            {
                Some(WindowStateLeaves::FullAttention { keys, .. }) => {
                    keys.retain().map_err(|e| e.to_string())
                }
                _ => Err(format!("layer {layer_index} has no extracted key-slab leaf")),
            },
            VerifyWindowInputSlot::FullAttentionValuesSlab { layer_index } => match attempt_inputs
                .state_leaves
                .get(layer_index)
            {
                Some(WindowStateLeaves::FullAttention { values, .. }) => {
                    values.retain().map_err(|e| e.to_string())
                }
                _ => Err(format!("layer {layer_index} has no extracted value-slab leaf")),
            },
            VerifyWindowInputSlot::LayerWeight {
                layer_index,
                slot: layer_weight_slot,
            } => {
                let layer_weights = decoder_layer_weights.get(layer_index).ok_or_else(|| {
                    format!("decoder layer {layer_index} has no bound weights")
                })?;
                layer_weight_array(layer_weights, layer_weight_slot)
            }
            VerifyWindowInputSlot::TrunkWeight(trunk_weight_slot) => match trunk_weight_slot {
                VerifyWindowTrunkWeightSlot::Embedding(affine_slot) => affine_slot_array_owned(
                    embedding_weights,
                    affine_slot,
                )
                .map_err(|_| {
                    "the embedding weights are not uniformly quantized for the compiled window"
                        .to_owned()
                }),
                VerifyWindowTrunkWeightSlot::FinalNormalization => final_normalization_weight
                    .retain()
                    .map_err(|e| e.to_string()),
                VerifyWindowTrunkWeightSlot::LanguageModelHead(affine_slot) => {
                    affine_slot_array_owned(language_model_head_weights, affine_slot).map_err(
                        |_| {
                            "the language-model head is not uniformly quantized for the compiled window"
                                .to_owned()
                        },
                    )
                }
            },
        };
        input_vector.push(resolved?);
    }
    Ok(input_vector)
}

/// Reads the ordered output vector into typed results.
pub(crate) fn read_window_outputs(
    geometry: &VerifyWindowGeometry,
    window_outputs: Vec<MlxArray>,
) -> Result<VerifyWindowAttemptOutputs, String> {
    let boundary_pair_count = geometry.row_count().saturating_sub(1).max(0) as usize;
    let gated_delta_layer_count = geometry
        .layer_kinds()
        .iter()
        .filter(|layer_kind| {
            **layer_kind == astronomical_runtime_integration::VerifyWindowLayerKind::GatedDelta
        })
        .count();
    let expected_output_count =
        2 + 2 * geometry.layer_kinds().len() + 2 * boundary_pair_count * gated_delta_layer_count;
    if window_outputs.len() != expected_output_count {
        return Err(format!(
            "the compiled window returned {} outputs for {} layers plus logits",
            window_outputs.len(),
            geometry.layer_kinds().len()
        ));
    }
    let mut output_iterator = window_outputs.into_iter();
    let all_position_logits = output_iterator
        .next()
        .ok_or_else(|| "the compiled window returned no all-position logits".to_owned())?;
    let pre_final_normalization_hidden_states = output_iterator.next().ok_or_else(|| {
        "the compiled window returned no pre-final-normalization hidden states".to_owned()
    })?;
    let mut layer_updates = Vec::with_capacity(geometry.layer_kinds().len());
    for layer_index in 0..geometry.layer_kinds().len() {
        match geometry.layer_kinds()[layer_index] {
            astronomical_runtime_integration::VerifyWindowLayerKind::GatedDelta => {
                let convolution = output_iterator.next().ok_or_else(|| {
                    format!("layer {layer_index} is missing its rolling-convolution output")
                })?;
                let recurrent = output_iterator.next().ok_or_else(|| {
                    format!("layer {layer_index} is missing its recurrent-state output")
                })?;
                let boundary_pair_count = geometry.row_count().saturating_sub(1).max(0) as usize;
                let mut boundary_convolution_states = Vec::with_capacity(boundary_pair_count);
                let mut boundary_recurrent_states = Vec::with_capacity(boundary_pair_count);
                for _ in 0..boundary_pair_count {
                    boundary_convolution_states.push(output_iterator.next().ok_or_else(|| {
                        format!("layer {layer_index} is missing a boundary convolution snapshot")
                    })?);
                    boundary_recurrent_states.push(output_iterator.next().ok_or_else(|| {
                        format!("layer {layer_index} is missing a boundary recurrent snapshot")
                    })?);
                }
                layer_updates.push(WindowStateUpdate::GatedDelta {
                    convolution,
                    recurrent,
                    boundary_convolution_states,
                    boundary_recurrent_states,
                });
            }
            astronomical_runtime_integration::VerifyWindowLayerKind::FullAttention => {
                let keys = output_iterator.next().ok_or_else(|| {
                    format!("layer {layer_index} is missing its rotated-keys output")
                })?;
                let values = output_iterator
                    .next()
                    .ok_or_else(|| format!("layer {layer_index} is missing its values output"))?;
                layer_updates.push(WindowStateUpdate::FullAttention { keys, values });
            }
        }
    }
    Ok(VerifyWindowAttemptOutputs {
        all_position_logits,
        pre_final_normalization_hidden_states,
        layer_updates,
    })
}

fn layer_weight_array(
    layer_weights: &Qwen3_5DecoderLayerWeights,
    slot: VerifyWindowLayerWeightSlot,
) -> Result<MlxArray, String> {
    match slot {
        VerifyWindowLayerWeightSlot::InputNormalization => layer_weights
            .input_normalization_weight
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowLayerWeightSlot::PostAttentionNormalization => layer_weights
            .post_attention_normalization_weight
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowLayerWeightSlot::FeedForward(feed_forward_slot) => {
            let crate::qwen3_5::model::decoder_layer_weights::Qwen3_5DecoderFeedForwardWeights::Dense(
                dense_mlp_weights,
            ) = &layer_weights.mlp_weights
            else {
                return Err(
                    "the compiled verification window requires dense feed-forward weights"
                        .to_owned(),
                );
            };
            match feed_forward_slot {
                VerifyWindowFeedForwardWeightSlot::Gate(affine_slot) => {
                    affine_slot_array_owned(&dense_mlp_weights.gate_projection, affine_slot)
                }
                VerifyWindowFeedForwardWeightSlot::Up(affine_slot) => {
                    affine_slot_array_owned(&dense_mlp_weights.up_projection, affine_slot)
                }
                VerifyWindowFeedForwardWeightSlot::Down(affine_slot) => {
                    affine_slot_array_owned(&dense_mlp_weights.down_projection, affine_slot)
                }
            }
        }
        VerifyWindowLayerWeightSlot::GatedDelta(gated_delta_slot) => {
            match &layer_weights.attention_weights {
                Qwen3_5AttentionWeights::Linear(linear_attention_weights) => {
                    linear_attention_weight_array(linear_attention_weights, gated_delta_slot)
                }
                _ => Err(
                    "the decoder layer is not a gated-delta layer for its compiled slot".to_owned(),
                ),
            }
        }
        VerifyWindowLayerWeightSlot::FullAttention(full_attention_slot) => {
            match &layer_weights.attention_weights {
                Qwen3_5AttentionWeights::Full(full_attention_weights) => {
                    full_attention_weight_array(full_attention_weights, full_attention_slot)
                }
                _ => Err(
                    "the decoder layer is not a full-attention layer for its compiled slot"
                        .to_owned(),
                ),
            }
        }
    }
}

fn linear_attention_weight_array(
    weights: &crate::qwen3_5::model::decoder_layer_weights::Qwen3_5LinearAttentionWeights,
    slot: VerifyWindowGatedDeltaWeightSlot,
) -> Result<MlxArray, String> {
    match slot {
        VerifyWindowGatedDeltaWeightSlot::InputQueriesKeysValues(affine_slot) => {
            affine_slot_array_owned(&weights.input_queries_keys_values_projection, affine_slot)
        }
        VerifyWindowGatedDeltaWeightSlot::OutputGate(affine_slot) => {
            affine_slot_array_owned(&weights.output_gate_projection, affine_slot)
        }
        VerifyWindowGatedDeltaWeightSlot::UpdateRate(affine_slot) => {
            affine_slot_array_owned(&weights.update_rate_projection, affine_slot)
        }
        VerifyWindowGatedDeltaWeightSlot::DecayInterval(affine_slot) => {
            affine_slot_array_owned(&weights.decay_interval_projection, affine_slot)
        }
        VerifyWindowGatedDeltaWeightSlot::OutputProjection(affine_slot) => {
            affine_slot_array_owned(&weights.output_projection, affine_slot)
        }
        VerifyWindowGatedDeltaWeightSlot::ConvolutionWeight => weights
            .convolution_weight
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowGatedDeltaWeightSlot::DecayIntervalBias => weights
            .decay_interval_bias
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowGatedDeltaWeightSlot::DecayRateLogarithm => weights
            .decay_rate_logarithm
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowGatedDeltaWeightSlot::NormalizationWeight => weights
            .normalization_weight
            .retain()
            .map_err(|e| e.to_string()),
    }
}

fn full_attention_weight_array(
    weights: &crate::qwen3_5::model::decoder_layer_weights::Qwen3_5FullAttentionWeights,
    slot: VerifyWindowFullAttentionWeightSlot,
) -> Result<MlxArray, String> {
    match slot {
        VerifyWindowFullAttentionWeightSlot::Query(affine_slot) => {
            affine_slot_array_owned(&weights.query_projection, affine_slot)
        }
        VerifyWindowFullAttentionWeightSlot::Key(affine_slot) => {
            affine_slot_array_owned(&weights.key_projection, affine_slot)
        }
        VerifyWindowFullAttentionWeightSlot::Value(affine_slot) => {
            affine_slot_array_owned(&weights.value_projection, affine_slot)
        }
        VerifyWindowFullAttentionWeightSlot::Output(affine_slot) => {
            affine_slot_array_owned(&weights.output_projection, affine_slot)
        }
        VerifyWindowFullAttentionWeightSlot::QueryNormalization => weights
            .query_normalization_weight
            .retain()
            .map_err(|e| e.to_string()),
        VerifyWindowFullAttentionWeightSlot::KeyNormalization => weights
            .key_normalization_weight
            .retain()
            .map_err(|e| e.to_string()),
    }
}

fn affine_slot_array<'weights>(
    affine_weights: &'weights Qwen3_5AffineWeights,
    affine_slot: VerifyWindowAffineSlot,
) -> Option<&'weights MlxArray> {
    let Qwen3_5AffineWeights::Quantized {
        packed_weight,
        quantization_scales,
        quantization_biases,
        ..
    } = affine_weights
    else {
        return None;
    };
    match affine_slot {
        VerifyWindowAffineSlot::PackedWeight => Some(packed_weight),
        VerifyWindowAffineSlot::Scales => Some(quantization_scales),
        VerifyWindowAffineSlot::Biases => Some(quantization_biases),
    }
}

fn affine_slot_array_owned(
    affine_weights: &Qwen3_5AffineWeights,
    affine_slot: VerifyWindowAffineSlot,
) -> Result<MlxArray, String> {
    affine_slot_array(affine_weights, affine_slot)
        .ok_or_else(|| "a compiled-window affine module is not uniformly quantized".to_owned())?
        .retain()
        .map_err(|error| error.to_string())
}
