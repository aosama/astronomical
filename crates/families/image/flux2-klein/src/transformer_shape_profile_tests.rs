//! Beside-source coverage for the config-driven transformer shape table. The table is a
//! name-sorted map, so these tests pin its size, its ordering and the shapes at both ends and in
//! the middle of the official 5 double-stream and 20 single-stream block profile.

use std::collections::BTreeMap;

use crate::Flux2KleinTransformerConfig;
use crate::test_fixtures::transformer_config_json;

fn official_config() -> Flux2KleinTransformerConfig {
    Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the exact profile configuration should validate")
}

fn official_shape_table() -> BTreeMap<String, Vec<usize>> {
    official_config().expected_weight_shapes().collect()
}

#[test]
fn should_list_the_169_official_weights_once_each_in_name_order() {
    let tensor_names = official_config()
        .expected_weight_shapes()
        .map(|(tensor_name, _)| tensor_name)
        .collect::<Vec<_>>();

    assert_eq!(tensor_names.len(), 169);
    assert_eq!(
        tensor_names.first().map(String::as_str),
        Some("context_embedder.weight")
    );
    assert_eq!(
        tensor_names.last().map(String::as_str),
        Some("x_embedder.weight")
    );
    assert!(tensor_names.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn should_size_the_shared_weights_from_the_hidden_and_conditioning_widths() {
    let shape_table = official_shape_table();

    assert_eq!(shape_table["x_embedder.weight"], vec![3_072, 128]);
    assert_eq!(shape_table["context_embedder.weight"], vec![3_072, 7_680]);
    assert_eq!(
        shape_table["time_guidance_embed.timestep_embedder.linear_1.weight"],
        vec![3_072, 256]
    );
    assert_eq!(
        shape_table["double_stream_modulation_img.linear.weight"],
        vec![18_432, 3_072]
    );
    assert_eq!(
        shape_table["single_stream_modulation.linear.weight"],
        vec![9_216, 3_072]
    );
    assert_eq!(shape_table["norm_out.linear.weight"], vec![6_144, 3_072]);
    assert_eq!(shape_table["proj_out.weight"], vec![128, 3_072]);
}

#[test]
fn should_size_a_double_stream_block_from_hidden_head_and_feed_forward_widths() {
    let shape_table = official_shape_table();

    assert_eq!(
        shape_table["transformer_blocks.4.attn.to_out.0.weight"],
        vec![3_072, 3_072]
    );
    assert_eq!(
        shape_table["transformer_blocks.2.attn.norm_added_k.weight"],
        vec![128]
    );
    assert_eq!(
        shape_table["transformer_blocks.0.ff.linear_in.weight"],
        vec![18_432, 3_072]
    );
    assert_eq!(
        shape_table["transformer_blocks.4.ff_context.linear_out.weight"],
        vec![3_072, 9_216]
    );
}

#[test]
fn should_size_a_single_stream_block_around_its_fused_projection() {
    let shape_table = official_shape_table();

    assert_eq!(
        shape_table["single_transformer_blocks.0.attn.norm_q.weight"],
        vec![128]
    );
    assert_eq!(
        shape_table["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"],
        vec![27_648, 3_072]
    );
    assert_eq!(
        shape_table["single_transformer_blocks.19.attn.to_out.weight"],
        vec![3_072, 12_288]
    );
}

#[test]
fn should_not_name_blocks_beyond_the_configured_counts() {
    let shape_table = official_shape_table();

    assert!(!shape_table.contains_key("transformer_blocks.5.attn.to_q.weight"));
    assert!(!shape_table.contains_key("single_transformer_blocks.20.attn.to_out.weight"));
}

#[test]
fn should_key_the_nine_shared_weights_by_name() {
    let shared_shapes = official_config().shared_weight_shapes();

    assert_eq!(shared_shapes.len(), 9);
    assert_eq!(shared_shapes["proj_out.weight"], vec![128, 3_072]);
    assert!(!shared_shapes.contains_key("transformer_blocks.0.attn.to_q.weight"));
}

#[test]
fn should_insert_the_sixteen_weights_of_one_double_block_beside_existing_entries() {
    let mut shapes = BTreeMap::from([("already_listed.weight".to_owned(), vec![1])]);

    official_config().insert_double_stream_block_shapes(&mut shapes, 3);

    assert_eq!(shapes.len(), 17);
    assert_eq!(shapes["already_listed.weight"], vec![1]);
    assert_eq!(
        shapes["transformer_blocks.3.attn.to_q.weight"],
        vec![3_072, 3_072]
    );
    assert_eq!(
        shapes["transformer_blocks.3.ff_context.linear_in.weight"],
        vec![18_432, 3_072]
    );
}

#[test]
fn should_keep_four_weights_when_a_single_block_is_inserted_twice() {
    let config = official_config();
    let mut shapes = BTreeMap::new();

    config.insert_single_stream_block_shapes(&mut shapes, 19);
    config.insert_single_stream_block_shapes(&mut shapes, 19);

    assert_eq!(shapes.len(), 4);
    assert_eq!(
        shapes["single_transformer_blocks.19.attn.to_out.weight"],
        vec![3_072, 12_288]
    );
}
