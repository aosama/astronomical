use astronomical_model_serving::{
    DecoderCacheLayerLayout, LagunaAttentionKind, LagunaCacheDescriptor,
    LagunaFeedForwardDescriptor, LagunaTargetNormalizer, laguna_decoder_cache_layout,
};
use serde_json::json;

use super::support::{
    LagunaQualificationSize, config_value, normalize, qualification_config_value,
};

#[test]
fn should_build_cache_layout_from_a_synthetic_non_modulo_schedule() {
    let mut config = config_value(3);
    config["layer_types"] = json!(["sliding", "full", "sliding"]);
    config["sliding_window"] = json!(16);
    config["mlp_layer_types"] = json!(["dense", "sparse", "dense"]);
    config["num_attention_heads_per_layer"] = json!([8, 12, 8]);
    config["num_experts"] = json!(16);
    config["num_experts_per_tok"] = json!(4);
    config["moe_intermediate_size"] = json!(768);
    config["shared_expert_intermediate_size"] = json!(512);
    let contract = normalize(config);
    assert_eq!(contract.layers().len(), 3);
    assert_eq!(
        contract.layers()[0].attention().kind(),
        LagunaAttentionKind::Sliding
    );
    assert_eq!(
        contract.layers()[1].attention().kind(),
        LagunaAttentionKind::Full
    );
    assert!(matches!(
        contract.layers()[0].attention().cache(),
        LagunaCacheDescriptor::Rotating { window_size: 16 }
    ));
    assert!(matches!(
        contract.layers()[1].feed_forward(),
        LagunaFeedForwardDescriptor::Moe(_)
    ));
    assert!(matches!(
        contract.layers()[2].feed_forward(),
        LagunaFeedForwardDescriptor::Dense(_)
    ));

    let decoder_cache_layout = laguna_decoder_cache_layout(&contract)
        .expect("a mixed descriptor schedule should produce a cache layout");
    assert_eq!(decoder_cache_layout.layer_count(), 3);
    assert!(matches!(
        decoder_cache_layout.layer(0),
        Some(DecoderCacheLayerLayout::RotatingWindowAttention {
            window_size: 16,
            ..
        })
    ));
    assert!(matches!(
        decoder_cache_layout.layer(1),
        Some(DecoderCacheLayerLayout::AppendOnlyAttention { .. })
    ));
}

#[test]
fn should_treat_xs_and_s_layer_schedules_as_named_rows() {
    for (
        row_name,
        fixture_size,
        expected_layer_count,
        expected_full_count,
        expected_sliding_heads,
    ) in [
        ("xs", LagunaQualificationSize::ExtraSmall, 40, 10, 64_u32),
        ("s", LagunaQualificationSize::Small, 48, 12, 72),
    ] {
        let contract = LagunaTargetNormalizer::normalize(
            &serde_json::to_vec(&qualification_config_value(fixture_size))
                .expect("qualification config should serialize"),
        )
        .unwrap_or_else(|_| panic!("{row_name} should normalize"));
        assert_eq!(contract.layers().len(), expected_layer_count, "{row_name}");
        let full_count = contract
            .layers()
            .iter()
            .filter(|layer| layer.attention().kind() == LagunaAttentionKind::Full)
            .count();
        assert_eq!(full_count, expected_full_count, "{row_name}");
        let sliding_layer = contract
            .layers()
            .iter()
            .find(|layer| layer.attention().kind() == LagunaAttentionKind::Sliding)
            .unwrap_or_else(|| panic!("{row_name} should contain sliding layers"));
        assert_eq!(
            sliding_layer.attention().query_head_count(),
            expected_sliding_heads,
            "{row_name}"
        );
        assert_eq!(
            sliding_layer.attention().key_value_head_count(),
            8,
            "{row_name}"
        );
        assert_eq!(
            sliding_layer.attention().head_dimension(),
            128,
            "{row_name}"
        );
        laguna_decoder_cache_layout(&contract)
            .unwrap_or_else(|_| panic!("{row_name} cache layout should build"));
    }
}
