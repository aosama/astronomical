use astronomical_model_serving::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};

use super::support::family_member_config_json;

#[test]
fn should_parse_family_knobs_without_baking_in_one_artifact_geometry() {
    let config = K2HorizonMoVAConfig::from_json_bytes(
        family_member_config_json(3, &[0, 1], 8, 4).as_bytes(),
    )
    .expect("family config should parse");
    assert_eq!(config.num_hidden_layers(), 3);
    assert_eq!(config.layer_kind(0), K2HorizonMoVALayerKind::Dense);
    assert_eq!(config.layer_kind(1), K2HorizonMoVALayerKind::Dense);
    assert_eq!(
        config.layer_kind(2),
        K2HorizonMoVALayerKind::SparseMixtureOfValues
    );
    assert_eq!(config.num_experts(), 8);
    assert_eq!(config.mova_num_experts(), 4);
    assert_eq!(
        config
            .affine_profile_for_module("model.layers.2.mlp.gate")
            .bits(),
        4
    );
}

#[test]
fn should_treat_zero_mova_experts_as_sparse_feed_forward_without_value_experts() {
    let config =
        K2HorizonMoVAConfig::from_json_bytes(family_member_config_json(2, &[0], 4, 0).as_bytes())
            .expect("family config should parse");
    assert_eq!(
        config.layer_kind(1),
        K2HorizonMoVALayerKind::SparseFeedForward
    );
}

#[test]
fn should_reject_non_affine_quantization_and_unknown_model_type() {
    let mut document: serde_json::Value =
        serde_json::from_str(&family_member_config_json(2, &[0], 4, 2)).expect("json");
    document["model_type"] = serde_json::json!("llama");
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
    document["model_type"] = serde_json::json!("k2_horizon_mova");
    document["quantization"]["mode"] = serde_json::json!("mxfp4");
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
}
