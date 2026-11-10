//! Where each quantizable module's affine weights live inside the typed
//! Qwen3.5 weights, resolved once from the artifact's canonical module
//! names so a probe swap never string-matches on the hot path.

use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;

#[cfg(test)]
#[path = "module_locations_tests.rs"]
mod module_locations_tests;

/// Where a module's weight lives inside the typed weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModuleLocation {
    Embedding,
    LanguageModelHead,
    /// `decoder_layer_index`, then which affine module inside the layer.
    DecoderLayerAttention {
        decoder_layer_index: usize,
        projection: AttentionProjection,
    },
    DecoderLayerDenseFeedForward {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
    /// Fused resident expert arrays for one layer and projection.
    DecoderLayerRoutedExperts {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
    DecoderLayerSharedExpert {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
    /// The shared expert's one-row output gate (`mlp.shared_expert_gate`).
    DecoderLayerSharedExpertOutputGate {
        decoder_layer_index: usize,
    },
    /// The mixture-of-experts router (`mlp.gate`), which picks the experts
    /// each token visits.
    DecoderLayerRouter {
        decoder_layer_index: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionProjection {
    Query,
    Key,
    Value,
    Output,
    LinearInputQueriesKeysValues,
    LinearOutputGate,
    LinearUpdateRate,
    LinearDecayInterval,
    LinearOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardProjection {
    Gate,
    Up,
    Down,
}

/// Resolves a canonical artifact module name (as the quantization crate's
/// inventory spells it, for example
/// `language_model.model.layers.3.linear_attn.in_proj_qkv`) to its location,
/// or `None` for a name outside the Qwen3.5 language model.
#[must_use]
pub fn module_location_for(module_name: &str) -> Option<ModuleLocation> {
    if module_name.ends_with("embed_tokens") {
        return Some(ModuleLocation::Embedding);
    }
    if module_name.ends_with("lm_head") {
        return Some(ModuleLocation::LanguageModelHead);
    }
    let (decoder_layer_index, layer_suffix) = split_decoder_layer(module_name)?;
    if let Some(projection) = attention_projection_for(layer_suffix) {
        return Some(ModuleLocation::DecoderLayerAttention {
            decoder_layer_index,
            projection,
        });
    }
    feed_forward_location_for(decoder_layer_index, layer_suffix)
}

/// The affine weights at a location, or `None` where the layer has another
/// shape (a full-attention location on a linear-attention layer) or the
/// location is a routed expert or the router, whose arrays live in their own
/// owners and are swapped by the probe's expert swaps.
pub(crate) fn affine_weights_at_mut(
    weights: &mut Qwen3_5Weights,
    location: ModuleLocation,
) -> Option<&mut Qwen3_5AffineWeights> {
    match location {
        ModuleLocation::Embedding => Some(&mut weights.embedding_weights),
        ModuleLocation::LanguageModelHead => Some(&mut weights.language_model_head_weights),
        ModuleLocation::DecoderLayerAttention {
            decoder_layer_index,
            projection,
        } => {
            let layer_weights = weights.decoder_layer_weights.get_mut(decoder_layer_index)?;
            attention_weights_mut(&mut layer_weights.attention_weights, projection)
        }
        ModuleLocation::DecoderLayerRoutedExperts { .. }
        | ModuleLocation::DecoderLayerRouter { .. } => None,
        feed_forward_location => feed_forward_weights_mut(weights, feed_forward_location),
    }
}

fn attention_weights_mut(
    attention_weights: &mut Qwen3_5AttentionWeights,
    projection: AttentionProjection,
) -> Option<&mut Qwen3_5AffineWeights> {
    match (attention_weights, projection) {
        (Qwen3_5AttentionWeights::Full(full), AttentionProjection::Query) => {
            Some(&mut full.query_projection)
        }
        (Qwen3_5AttentionWeights::Full(full), AttentionProjection::Key) => {
            Some(&mut full.key_projection)
        }
        (Qwen3_5AttentionWeights::Full(full), AttentionProjection::Value) => {
            Some(&mut full.value_projection)
        }
        (Qwen3_5AttentionWeights::Full(full), AttentionProjection::Output) => {
            Some(&mut full.output_projection)
        }
        (
            Qwen3_5AttentionWeights::Linear(linear),
            AttentionProjection::LinearInputQueriesKeysValues,
        ) => Some(&mut linear.input_queries_keys_values_projection),
        (Qwen3_5AttentionWeights::Linear(linear), AttentionProjection::LinearOutputGate) => {
            Some(&mut linear.output_gate_projection)
        }
        (Qwen3_5AttentionWeights::Linear(linear), AttentionProjection::LinearUpdateRate) => {
            Some(&mut linear.update_rate_projection)
        }
        (Qwen3_5AttentionWeights::Linear(linear), AttentionProjection::LinearDecayInterval) => {
            Some(&mut linear.decay_interval_projection)
        }
        (Qwen3_5AttentionWeights::Linear(linear), AttentionProjection::LinearOutput) => {
            Some(&mut linear.output_projection)
        }
        _ => None,
    }
}

fn feed_forward_weights_mut(
    weights: &mut Qwen3_5Weights,
    location: ModuleLocation,
) -> Option<&mut Qwen3_5AffineWeights> {
    let (decoder_layer_index, is_shared_expert, projection) = match location {
        ModuleLocation::DecoderLayerDenseFeedForward {
            decoder_layer_index,
            projection,
        } => (decoder_layer_index, false, Some(projection)),
        ModuleLocation::DecoderLayerSharedExpert {
            decoder_layer_index,
            projection,
        } => (decoder_layer_index, true, Some(projection)),
        ModuleLocation::DecoderLayerSharedExpertOutputGate {
            decoder_layer_index,
        } => (decoder_layer_index, true, None),
        _ => return None,
    };
    let layer_weights = weights.decoder_layer_weights.get_mut(decoder_layer_index)?;
    match (&mut layer_weights.mlp_weights, is_shared_expert, projection) {
        (Qwen3_5DecoderFeedForwardWeights::Dense(dense), false, Some(projection)) => {
            Some(match projection {
                FeedForwardProjection::Gate => &mut dense.gate_projection,
                FeedForwardProjection::Up => &mut dense.up_projection,
                FeedForwardProjection::Down => &mut dense.down_projection,
            })
        }
        (Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(moe), true, Some(projection)) => {
            Some(match projection {
                FeedForwardProjection::Gate => &mut moe.shared_expert_gate_projection,
                FeedForwardProjection::Up => &mut moe.shared_expert_up_projection,
                FeedForwardProjection::Down => &mut moe.shared_expert_down_projection,
            })
        }
        (Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(moe), true, None) => {
            Some(&mut moe.shared_expert_output_gate_projection)
        }
        _ => None,
    }
}

/// `(layer index, suffix after the index)` from `...layers.N.suffix`.
fn split_decoder_layer(module_name: &str) -> Option<(usize, &str)> {
    let (_, after_layers) = module_name.split_once("layers.")?;
    let (index_text, layer_suffix) = after_layers.split_once('.')?;
    Some((index_text.parse().ok()?, layer_suffix))
}

fn attention_projection_for(layer_suffix: &str) -> Option<AttentionProjection> {
    Some(match layer_suffix {
        "self_attn.q_proj" => AttentionProjection::Query,
        "self_attn.k_proj" => AttentionProjection::Key,
        "self_attn.v_proj" => AttentionProjection::Value,
        "self_attn.o_proj" => AttentionProjection::Output,
        "linear_attn.in_proj_qkv" => AttentionProjection::LinearInputQueriesKeysValues,
        "linear_attn.in_proj_z" => AttentionProjection::LinearOutputGate,
        "linear_attn.in_proj_b" => AttentionProjection::LinearUpdateRate,
        "linear_attn.in_proj_a" => AttentionProjection::LinearDecayInterval,
        "linear_attn.out_proj" => AttentionProjection::LinearOutput,
        _ => return None,
    })
}

/// The two one-matrix gates of a mixture-of-experts layer: the router that
/// picks the experts, and the shared expert's output gate.
fn gate_location_for(decoder_layer_index: usize, layer_suffix: &str) -> Option<ModuleLocation> {
    match layer_suffix {
        "mlp.gate" => Some(ModuleLocation::DecoderLayerRouter {
            decoder_layer_index,
        }),
        "mlp.shared_expert_gate" => Some(ModuleLocation::DecoderLayerSharedExpertOutputGate {
            decoder_layer_index,
        }),
        _ => None,
    }
}

fn feed_forward_location_for(
    decoder_layer_index: usize,
    layer_suffix: &str,
) -> Option<ModuleLocation> {
    if let Some(gate_location) = gate_location_for(decoder_layer_index, layer_suffix) {
        return Some(gate_location);
    }
    let (container, projection_name) = layer_suffix.rsplit_once('.')?;
    let projection = match projection_name {
        "gate_proj" => FeedForwardProjection::Gate,
        "up_proj" => FeedForwardProjection::Up,
        "down_proj" => FeedForwardProjection::Down,
        _ => return None,
    };
    Some(match container {
        "mlp" => ModuleLocation::DecoderLayerDenseFeedForward {
            decoder_layer_index,
            projection,
        },
        "mlp.shared_expert" => ModuleLocation::DecoderLayerSharedExpert {
            decoder_layer_index,
            projection,
        },
        "mlp.switch_mlp" | "mlp.experts" => ModuleLocation::DecoderLayerRoutedExperts {
            decoder_layer_index,
            projection,
        },
        _ => return None,
    })
}
