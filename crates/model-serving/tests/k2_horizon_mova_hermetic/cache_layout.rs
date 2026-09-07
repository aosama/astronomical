use astronomical_model_serving::{
    DecoderCacheTensorDtype, K2HorizonMoVAConfig, k2_horizon_mova_decoder_cache_layout,
};

use super::support::family_member_config_json;

#[test]
fn k2_horizon_mova_prompt_cache_layout_is_append_only_attention_from_family_config() {
    let config =
        K2HorizonMoVAConfig::from_json_bytes(family_member_config_json(4, &[0], 4, 2).as_bytes())
            .expect("tiny family config should parse");
    let layout = k2_horizon_mova_decoder_cache_layout(&config)
        .expect("K2 Horizon MoVA cache layout should resolve from family config");
    let sequence_tensors = layout.sequence_tensor_layouts();
    assert_eq!(sequence_tensors.len(), config.num_hidden_layers() * 2);
    assert!(layout.boundary_tensor_layouts().is_empty());
    for sequence_tensor in sequence_tensors {
        assert_eq!(
            sequence_tensor.tensor_layout().dtype(),
            DecoderCacheTensorDtype::BFloat16
        );
        assert_eq!(sequence_tensor.tensor_layout().sequence_axis(), Some(2));
        assert_eq!(
            sequence_tensor.tensor_layout().dimensions(),
            &[1, config.num_key_value_heads(), 0, config.head_dim()]
        );
    }
}
