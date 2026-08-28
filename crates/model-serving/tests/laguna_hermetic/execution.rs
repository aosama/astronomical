use astronomical_model_serving::{
    DecoderCacheLayerLayout, LagunaAttentionKind, LagunaCacheDescriptor,
    LagunaFeedForwardDescriptor, laguna_decoder_cache_layout,
};
use serde_json::json;

use super::support::{config_value, normalize};

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
