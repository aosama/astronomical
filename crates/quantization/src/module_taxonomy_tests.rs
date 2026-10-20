//! Unit tests for the name-based module taxonomy. Every expectation below is
//! taken from a name OptiQ documents as having bitten a real conversion.

use super::*;

#[test]
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
fn should_return_none_for_modules_outside_transformer_blocks() {
    assert_eq!(transformer_block_index("model.embed_tokens"), None);
    assert_eq!(transformer_block_index("lm_head"), None);
    assert_eq!(transformer_block_index("model.norm"), None);
}

#[test]
fn should_skip_a_non_numeric_segment_after_an_early_layers_and_find_the_real_block() {
    // A stray non-numeric segment after one `layers` must not shadow a later
    // real block index: the windowed scan keeps looking for the next `layers`
    // whose successor is all digits.
    assert_eq!(
        transformer_block_index("backbone.layers.norm.layers.7.attn.q_proj"),
        Some(7)
    );
}

#[test]
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
fn should_classify_the_dense_feed_forward_and_the_other_inside_block_fallthrough() {
    // `up_proj`/`down_proj` are the SwiGLU dense MLP suffixes; a name still
    // inside a block that matches no suffix table (here a router bias) is the
    // `OtherInsideBlock` fall-through, not a router under the tight taxonomy.
    assert_eq!(
        classify_module("model.layers.0.mlp.up_proj"),
        ModuleRole::DenseFeedForward
    );
    assert_eq!(
        classify_module("model.layers.9.mlp.gate.expert_bias"),
        ModuleRole::OtherInsideBlock
    );
}

#[test]
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

#[test]
fn should_return_no_boundary_modules_for_a_zero_count() {
    // A zero count protects nothing here, deliberately diverging from OptiQ's
    // `sorted[-0:]` slice, which would select every block.
    let module_names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.2.self_attn.q_proj",
    ];

    assert!(boundary_block_modules(&module_names, 0).is_empty());
}
