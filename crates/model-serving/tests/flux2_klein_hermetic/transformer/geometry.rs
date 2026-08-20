use astronomical_model_serving::{
    Flux2KleinTransformerConfig, Flux2KleinTransformerGeometry, Flux2KleinTransformerGeometryError,
};

use super::super::support::transformer_config_json;

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
