//! Unit tests for the structural priority rules. Expected values are the
//! arithmetic of OptiQ's `_structural_priority` worked by hand.

use super::*;

#[test]
#[ignore = "pending implementation of structural_priority"]
fn should_rank_embeddings_and_the_head_at_the_top() {
    assert_eq!(structural_priority("model.embed_tokens", 32), 1.0);
    assert_eq!(structural_priority("lm_head", 32), 1.0);
}

#[test]
#[ignore = "pending implementation of structural_priority"]
fn should_protect_the_first_and_last_blocks() {
    // 0.90 for the edge block plus 0.30 for attention, clamped to 1.0.
    assert_eq!(
        structural_priority("model.layers.0.self_attn.q_proj", 32),
        1.0
    );
    // 0.90 for the edge block, dense feed-forward adds nothing.
    assert_eq!(structural_priority("model.layers.31.mlp.up_proj", 32), 0.90);
}

#[test]
#[ignore = "pending implementation of structural_priority"]
fn should_rank_routed_experts_below_the_dense_feed_forward_in_the_middle() {
    let dense_priority = structural_priority("model.layers.16.mlp.up_proj", 32);
    let expert_priority = structural_priority("model.layers.16.mlp.switch_mlp.up_proj", 32);
    let router_priority = structural_priority("model.layers.16.mlp.gate.weight", 32);

    assert!(expert_priority < dense_priority);
    assert!(router_priority > dense_priority);
}

#[test]
#[ignore = "pending implementation of structural_priority"]
fn should_never_fall_below_the_floor() {
    assert!(structural_priority("model.layers.16.mlp.switch_mlp.up_proj", 32) >= 0.05);
}
