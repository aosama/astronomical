//! Unit tests for resolving Qwen3.5 artifact module names to weight
//! locations.

use super::*;

#[test]
fn should_resolve_linear_and_full_attention_projections() {
    assert_eq!(
        module_location_for("language_model.model.layers.3.linear_attn.in_proj_qkv"),
        Some(ModuleLocation::DecoderLayerAttention {
            decoder_layer_index: 3,
            projection: AttentionProjection::LinearInputQueriesKeysValues,
        })
    );
    assert_eq!(
        module_location_for("language_model.model.layers.11.self_attn.o_proj"),
        Some(ModuleLocation::DecoderLayerAttention {
            decoder_layer_index: 11,
            projection: AttentionProjection::Output,
        })
    );
}

#[test]
fn should_resolve_dense_shared_and_routed_feed_forward_projections() {
    assert_eq!(
        module_location_for("language_model.model.layers.0.mlp.down_proj"),
        Some(ModuleLocation::DecoderLayerDenseFeedForward {
            decoder_layer_index: 0,
            projection: FeedForwardProjection::Down,
        })
    );
    assert_eq!(
        module_location_for("language_model.model.layers.2.mlp.shared_expert.up_proj"),
        Some(ModuleLocation::DecoderLayerSharedExpert {
            decoder_layer_index: 2,
            projection: FeedForwardProjection::Up,
        })
    );
    assert_eq!(
        module_location_for("language_model.model.layers.2.mlp.shared_expert_gate"),
        Some(ModuleLocation::DecoderLayerSharedExpertOutputGate {
            decoder_layer_index: 2
        })
    );
    assert_eq!(
        module_location_for("language_model.model.layers.2.mlp.switch_mlp.gate_proj"),
        Some(ModuleLocation::DecoderLayerRoutedExperts {
            decoder_layer_index: 2,
            projection: FeedForwardProjection::Gate,
        })
    );
}

#[test]
fn should_resolve_the_embedding_and_the_head() {
    assert_eq!(
        module_location_for("language_model.model.embed_tokens"),
        Some(ModuleLocation::Embedding)
    );
    assert_eq!(
        module_location_for("language_model.lm_head"),
        Some(ModuleLocation::LanguageModelHead)
    );
}

#[test]
fn should_not_resolve_names_outside_the_language_model() {
    assert_eq!(module_location_for("vision_tower.blocks.0.attn.qkv"), None);
    assert_eq!(
        module_location_for("language_model.model.layers.x.mlp.up_proj"),
        None
    );
    assert_eq!(
        module_location_for("language_model.model.layers.0.mlp.router"),
        None
    );
    assert_eq!(module_location_for(""), None);
}

#[test]
fn should_resolve_the_router_apart_from_the_swiglu_gate_projection() {
    assert_eq!(
        module_location_for("language_model.model.layers.5.mlp.gate"),
        Some(ModuleLocation::DecoderLayerRouter {
            decoder_layer_index: 5,
        })
    );
    assert_eq!(
        module_location_for("language_model.model.layers.5.mlp.gate_proj"),
        Some(ModuleLocation::DecoderLayerDenseFeedForward {
            decoder_layer_index: 5,
            projection: FeedForwardProjection::Gate,
        })
    );
}
