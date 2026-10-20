//! Unit tests for the name-based module taxonomy. Every expectation below is
//! taken from a name OptiQ documents as having bitten a real conversion.

use super::*;

#[test]
#[ignore = "pending implementation of transformer_block_index"]
fn should_extract_the_block_index_from_a_layers_segment() {
    assert_eq!(
        transformer_block_index("model.layers.7.mlp.up_proj"),
        Some(7)
    );
    assert_eq!(
        transformer_block_index("language_model.model.layers.31.self_attn.q_proj"),
        Some(31)
    );
}

#[test]
#[ignore = "pending implementation of transformer_block_index"]
fn should_return_none_for_modules_outside_transformer_blocks() {
    assert_eq!(transformer_block_index("model.embed_tokens"), None);
    assert_eq!(transformer_block_index("lm_head"), None);
    assert_eq!(transformer_block_index("model.norm"), None);
}

#[test]
#[ignore = "pending implementation of classify_module"]
fn should_classify_the_qwen3_moe_router_but_not_the_swiglu_gate_projection() {
    // `mlp.gate` is the router; `mlp.gate_proj` is the activation gate inside
    // each expert. Confusing the two quantizes the router.
    assert_eq!(
        classify_module("model.layers.3.mlp.gate"),
        ModuleRole::MixtureOfExpertsRouter
    );
    assert_eq!(
        classify_module("model.layers.3.mlp.gate_proj"),
        ModuleRole::DenseFeedForward
    );
}

#[test]
#[ignore = "pending implementation of classify_module"]
fn should_classify_gemma_style_router_projections() {
    assert_eq!(
        classify_module("model.layers.3.mlp.router"),
        ModuleRole::MixtureOfExpertsRouter
    );
    assert_eq!(
        classify_module("model.layers.3.mlp.router.proj"),
        ModuleRole::MixtureOfExpertsRouter
    );
}

#[test]
#[ignore = "pending implementation of classify_module"]
fn should_classify_shared_experts_before_routed_experts() {
    assert_eq!(
        classify_module("model.layers.3.mlp.shared_experts.gate_proj"),
        ModuleRole::SharedExpert
    );
    assert_eq!(
        classify_module("model.layers.3.mlp.switch_mlp.gate_proj"),
        ModuleRole::RoutedExpert
    );
    assert_eq!(
        classify_module("model.layers.3.mlp.experts.17.down_proj"),
        ModuleRole::RoutedExpert
    );
}

#[test]
#[ignore = "pending implementation of classify_module"]
fn should_classify_both_attention_families() {
    assert_eq!(
        classify_module("model.layers.23.self_attn.o_proj"),
        ModuleRole::FullAttentionProjection
    );
    assert_eq!(
        classify_module("model.layers.22.linear_attn.in_proj_qkv"),
        ModuleRole::LinearAttentionProjection
    );
}

#[test]
#[ignore = "pending implementation of classify_module"]
fn should_protect_embeddings_heads_and_modules_outside_blocks() {
    assert_eq!(
        classify_module("model.embed_tokens"),
        ModuleRole::EmbeddingOrOutputHead
    );
    assert_eq!(
        classify_module("lm_head"),
        ModuleRole::EmbeddingOrOutputHead
    );
    assert_eq!(
        classify_module("mtp.fc"),
        ModuleRole::OutsideTransformerBlocks
    );
    assert!(is_always_protected(classify_module("lm_head")));
}

#[test]
#[ignore = "pending implementation of boundary_block_modules"]
fn should_collect_the_first_and_last_blocks_as_boundary_modules() {
    let module_names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.3.self_attn.q_proj",
        "lm_head",
    ];

    let boundary_modules = boundary_block_modules(&module_names, 1);

    assert_eq!(
        boundary_modules,
        vec![
            "model.layers.0.self_attn.q_proj",
            "model.layers.3.self_attn.q_proj"
        ]
    );
}
