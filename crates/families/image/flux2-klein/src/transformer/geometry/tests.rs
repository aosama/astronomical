use std::collections::BTreeMap;

use super::{Flux2KleinTransformerGeometry, Flux2KleinTransformerGeometryError};
use crate::configuration::Flux2KleinTransformerConfig;

use crate::test_fixtures::transformer_config_json;

#[test]
fn should_expose_the_exact_official_klein_4b_transformer_geometry() {
    let config = Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the exact profile configuration should validate first");
    let geometry = Flux2KleinTransformerGeometry::from_config(&config)
        .expect("validated model geometry should construct the execution profile");

    assert_eq!(geometry.hidden_width(), 3_072);
    assert_eq!(geometry.feed_forward_width(), 9_216);
    assert_eq!(geometry.rope_axis_widths(), [32, 32, 32, 32]);
    assert_eq!(geometry.rope_theta(), 2_000.0);
    assert_eq!(geometry.double_stream_block_count(), 5);
    assert_eq!(geometry.single_stream_block_count(), 20);
    assert_eq!(geometry.output_width(), 128);
}

#[test]
fn should_reject_geometry_that_cannot_preserve_four_axis_rope() {
    let invalid_geometry =
        Flux2KleinTransformerGeometry::new(6, 3, 2, 4, 6, [2, 2, 2, 4], 2_000.0, 1, 1, 4, 1.0e-6);

    assert!(matches!(
        invalid_geometry,
        Err(Flux2KleinTransformerGeometryError::RopeWidthMismatch { .. })
    ));
}

fn official_geometry() -> Flux2KleinTransformerGeometry {
    let config = Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the exact profile configuration should validate first");
    Flux2KleinTransformerGeometry::from_config(&config)
        .expect("validated model geometry should construct the execution profile")
}

fn small_geometry(
    double_stream_block_count: usize,
    single_stream_block_count: usize,
) -> Flux2KleinTransformerGeometry {
    Flux2KleinTransformerGeometry::new(
        16,
        2,
        8,
        4,
        6,
        [2, 2, 2, 2],
        2_000.0,
        double_stream_block_count,
        single_stream_block_count,
        4,
        1.0e-6,
    )
    .expect("a small geometry with eight-wide heads should construct")
}

#[test]
fn should_list_the_shared_weights_then_double_blocks_then_single_blocks_in_order() {
    let shapes = official_geometry()
        .expected_weight_shapes()
        .collect::<Vec<_>>();

    assert_eq!(shapes.len(), 169);
    assert_eq!(
        shapes[0],
        ("x_embedder.weight".to_owned(), vec![3_072, 128])
    );
    assert_eq!(
        shapes[3],
        (
            "time_guidance_embed.timestep_embedder.linear_2.weight".to_owned(),
            vec![3_072, 3_072]
        )
    );
    assert_eq!(shapes[8], ("proj_out.weight".to_owned(), vec![128, 3_072]));
    assert_eq!(shapes[9].0, "transformer_blocks.0.attn.add_k_proj.weight");
    assert_eq!(
        shapes[89].0,
        "single_transformer_blocks.0.attn.norm_k.weight"
    );
    assert_eq!(
        shapes[168],
        (
            "single_transformer_blocks.19.attn.to_out.weight".to_owned(),
            vec![3_072, 12_288]
        )
    );
}

#[test]
fn should_describe_a_double_block_as_matrices_then_head_norms_then_both_feed_forwards() {
    let shapes = official_geometry()
        .expected_weight_shapes()
        .collect::<Vec<_>>();

    assert_eq!(
        shapes[16],
        (
            "transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![3_072, 3_072]
        )
    );
    assert_eq!(
        shapes[17],
        (
            "transformer_blocks.0.attn.norm_added_k.weight".to_owned(),
            vec![128]
        )
    );
    assert_eq!(
        shapes[21],
        (
            "transformer_blocks.0.ff.linear_in.weight".to_owned(),
            vec![18_432, 3_072]
        )
    );
    assert_eq!(
        shapes[24],
        (
            "transformer_blocks.0.ff_context.linear_out.weight".to_owned(),
            vec![3_072, 9_216]
        )
    );
    assert_eq!(shapes[25].0, "transformer_blocks.1.attn.add_k_proj.weight");
}

#[test]
fn should_describe_a_single_block_as_norms_then_the_fused_projection_then_the_output() {
    let shapes = official_geometry()
        .expected_weight_shapes()
        .collect::<Vec<_>>();

    assert_eq!(
        shapes[90].0,
        "single_transformer_blocks.0.attn.norm_q.weight"
    );
    assert_eq!(
        shapes[91],
        (
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".to_owned(),
            vec![27_648, 3_072]
        )
    );
    assert_eq!(shapes[92].1, vec![3_072, 12_288]);
    assert_eq!(
        shapes[93].0,
        "single_transformer_blocks.1.attn.norm_k.weight"
    );
}

#[test]
fn should_not_name_blocks_beyond_the_configured_counts() {
    let names = official_geometry()
        .expected_weight_shapes()
        .map(|(tensor_name, _)| tensor_name)
        .collect::<Vec<_>>();

    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("transformer_blocks.5."))
    );
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("single_transformer_blocks.20."))
    );
}

#[test]
fn should_emit_one_tensor_set_per_block_even_at_the_minimum_block_counts() {
    let minimal = small_geometry(1, 1)
        .expected_weight_shapes()
        .collect::<Vec<_>>();
    let larger = small_geometry(3, 2)
        .expected_weight_shapes()
        .collect::<Vec<_>>();

    assert_eq!(minimal.len(), 9 + 16 + 4);
    assert_eq!(larger.len(), 9 + 3 * 16 + 2 * 4);
    assert_eq!(minimal[0], ("x_embedder.weight".to_owned(), vec![16, 4]));
    assert_eq!(
        minimal[6],
        (
            "single_stream_modulation.linear.weight".to_owned(),
            vec![48, 16]
        )
    );
    assert_eq!(
        minimal[25].0,
        "single_transformer_blocks.0.attn.norm_k.weight"
    );
    assert_eq!(
        minimal[27],
        (
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".to_owned(),
            vec![144, 16]
        )
    );
}

#[test]
fn should_agree_with_the_config_shape_table_for_the_official_profile() {
    let config = Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the exact profile configuration should validate first");

    let geometry_table = official_geometry()
        .expected_weight_shapes()
        .collect::<BTreeMap<_, _>>();
    let config_table = config.expected_weight_shapes().collect::<BTreeMap<_, _>>();

    assert_eq!(geometry_table, config_table);
}

#[test]
fn should_size_the_nine_shared_weights_from_the_geometry_widths() {
    let shared_shapes = small_geometry(1, 1).shared_weight_shapes();

    assert_eq!(shared_shapes.len(), 9);
    assert_eq!(
        shared_shapes[1],
        ("context_embedder.weight".to_owned(), vec![16, 6])
    );
    assert_eq!(
        shared_shapes[2],
        (
            "time_guidance_embed.timestep_embedder.linear_1.weight".to_owned(),
            vec![16, 256]
        )
    );
    assert_eq!(
        shared_shapes[7],
        ("norm_out.linear.weight".to_owned(), vec![32, 16])
    );
    assert_eq!(
        shared_shapes[8],
        ("proj_out.weight".to_owned(), vec![4, 16])
    );
}

#[test]
fn should_push_the_sixteen_weights_of_one_double_block_under_its_own_prefix() {
    let mut shapes = Vec::new();

    small_geometry(8, 1).push_double_stream_block_shapes(&mut shapes, 7);

    assert_eq!(shapes.len(), 16);
    assert!(
        shapes
            .iter()
            .all(|(tensor_name, _)| tensor_name.starts_with("transformer_blocks.7."))
    );
    assert_eq!(
        shapes[7],
        (
            "transformer_blocks.7.attn.to_v.weight".to_owned(),
            vec![16, 16]
        )
    );
    assert_eq!(
        shapes[11],
        (
            "transformer_blocks.7.attn.norm_q.weight".to_owned(),
            vec![8]
        )
    );
    assert_eq!(
        shapes[12],
        (
            "transformer_blocks.7.ff.linear_in.weight".to_owned(),
            vec![96, 16]
        )
    );
    assert_eq!(
        shapes[15],
        (
            "transformer_blocks.7.ff_context.linear_out.weight".to_owned(),
            vec![16, 48]
        )
    );
}

#[test]
fn should_push_the_four_weights_of_one_single_block_after_the_shapes_already_listed() {
    let mut shapes = vec![("already_listed.weight".to_owned(), vec![1])];

    small_geometry(1, 3).push_single_stream_block_shapes(&mut shapes, 2);

    assert_eq!(shapes.len(), 5);
    assert_eq!(shapes[0].0, "already_listed.weight");
    assert_eq!(
        shapes[1].0,
        "single_transformer_blocks.2.attn.norm_k.weight"
    );
    assert_eq!(
        shapes[3],
        (
            "single_transformer_blocks.2.attn.to_qkv_mlp_proj.weight".to_owned(),
            vec![144, 16]
        )
    );
    assert_eq!(shapes[4].1, vec![16, 64]);
}
