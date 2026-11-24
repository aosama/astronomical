use std::cmp::Ordering;

use super::compare_in_mlx_lm_traversal_order;

fn sorted_in_traversal_order(mut module_names: Vec<&str>) -> Vec<&str> {
    module_names.sort_by(compare_names);
    module_names
}

fn compare_names(left_name: &&str, right_name: &&str) -> Ordering {
    compare_in_mlx_lm_traversal_order(left_name, right_name)
}

#[test]
fn should_visit_a_dense_block_feed_forward_first_then_attention_in_reverse_assignment_order() {
    let visited = sorted_in_traversal_order(vec![
        "language_model.model.layers.3.self_attn.q_proj",
        "language_model.model.layers.3.mlp.gate_proj",
        "language_model.model.layers.3.self_attn.o_proj",
        "language_model.model.layers.3.mlp.up_proj",
        "language_model.model.layers.3.self_attn.k_proj",
        "language_model.model.layers.3.mlp.down_proj",
        "language_model.model.layers.3.self_attn.v_proj",
    ]);

    assert_eq!(
        visited,
        vec![
            "language_model.model.layers.3.mlp.up_proj",
            "language_model.model.layers.3.mlp.down_proj",
            "language_model.model.layers.3.mlp.gate_proj",
            "language_model.model.layers.3.self_attn.o_proj",
            "language_model.model.layers.3.self_attn.v_proj",
            "language_model.model.layers.3.self_attn.k_proj",
            "language_model.model.layers.3.self_attn.q_proj",
        ]
    );
}

#[test]
fn should_visit_the_highest_block_first_and_compare_block_indices_as_numbers() {
    let visited = sorted_in_traversal_order(vec![
        "language_model.model.layers.2.mlp.up_proj",
        "language_model.model.layers.10.mlp.up_proj",
        "language_model.model.layers.9.mlp.up_proj",
    ]);

    assert_eq!(
        visited,
        vec![
            "language_model.model.layers.10.mlp.up_proj",
            "language_model.model.layers.9.mlp.up_proj",
            "language_model.model.layers.2.mlp.up_proj",
        ]
    );
}

#[test]
fn should_visit_a_linear_attention_block_in_reverse_assignment_order() {
    let visited = sorted_in_traversal_order(vec![
        "language_model.model.layers.0.linear_attn.in_proj_qkv",
        "language_model.model.layers.0.linear_attn.in_proj_z",
        "language_model.model.layers.0.linear_attn.in_proj_b",
        "language_model.model.layers.0.linear_attn.in_proj_a",
        "language_model.model.layers.0.linear_attn.out_proj",
    ]);

    assert_eq!(
        visited,
        vec![
            "language_model.model.layers.0.linear_attn.out_proj",
            "language_model.model.layers.0.linear_attn.in_proj_a",
            "language_model.model.layers.0.linear_attn.in_proj_b",
            "language_model.model.layers.0.linear_attn.in_proj_z",
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        ]
    );
}

#[test]
fn should_visit_a_mixture_of_experts_block_as_mlx_lm_does() {
    let visited = sorted_in_traversal_order(vec![
        "language_model.model.layers.1.mlp.gate",
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj",
        "language_model.model.layers.1.mlp.switch_mlp.up_proj",
        "language_model.model.layers.1.mlp.switch_mlp.down_proj",
        "language_model.model.layers.1.mlp.shared_expert.gate_proj",
        "language_model.model.layers.1.mlp.shared_expert.down_proj",
        "language_model.model.layers.1.mlp.shared_expert.up_proj",
        "language_model.model.layers.1.mlp.shared_expert_gate",
    ]);

    assert_eq!(
        visited,
        vec![
            "language_model.model.layers.1.mlp.shared_expert_gate",
            "language_model.model.layers.1.mlp.shared_expert.up_proj",
            "language_model.model.layers.1.mlp.shared_expert.down_proj",
            "language_model.model.layers.1.mlp.shared_expert.gate_proj",
            "language_model.model.layers.1.mlp.switch_mlp.down_proj",
            "language_model.model.layers.1.mlp.switch_mlp.up_proj",
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj",
            "language_model.model.layers.1.mlp.gate",
        ]
    );
}

#[test]
fn should_visit_an_untied_head_before_the_blocks_and_the_embedding_last() {
    let visited = sorted_in_traversal_order(vec![
        "language_model.model.embed_tokens",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.lm_head",
    ]);

    assert_eq!(
        visited,
        vec![
            "language_model.lm_head",
            "language_model.model.layers.0.mlp.up_proj",
            "language_model.model.embed_tokens",
        ]
    );
}

#[test]
fn should_order_unknown_segments_deterministically_after_known_ones() {
    assert_eq!(
        compare_in_mlx_lm_traversal_order(
            "language_model.model.layers.0.mlp.up_proj",
            "language_model.model.layers.0.mlp.novel_projection"
        ),
        Ordering::Less
    );
    assert_eq!(
        compare_in_mlx_lm_traversal_order("decoder.beta", "decoder.alpha"),
        Ordering::Less
    );
    assert_eq!(
        compare_in_mlx_lm_traversal_order("decoder.alpha", "decoder.alpha"),
        Ordering::Equal
    );
}

#[test]
fn should_order_a_parent_before_its_own_child() {
    assert_eq!(
        compare_in_mlx_lm_traversal_order("language_model.model", "language_model.model.norm"),
        Ordering::Less
    );
}
