use astronomical_model_serving::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};

use super::support;

#[test]
fn should_parse_family_knobs_without_baking_in_one_artifact_geometry() {
    let config = K2HorizonMoVAConfig::from_json_bytes(
        support::family_member_config_json(3, &[0, 1], 8, 4).as_bytes(),
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
    let config = K2HorizonMoVAConfig::from_json_bytes(
        support::family_member_config_json(2, &[0], 4, 0).as_bytes(),
    )
    .expect("family config should parse");
    assert_eq!(
        config.layer_kind(1),
        K2HorizonMoVALayerKind::SparseFeedForward
    );
}

#[test]
fn should_reject_non_affine_quantization_and_unknown_model_type() {
    let mut document: serde_json::Value =
        serde_json::from_str(&support::family_member_config_json(2, &[0], 4, 2)).expect("json");
    document["model_type"] = serde_json::json!("llama");
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
    document["model_type"] = serde_json::json!("k2_horizon_mova");
    document["quantization"]["mode"] = serde_json::json!("mxfp4");
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
}

#[test]
fn should_reject_declared_knobs_the_serving_path_does_not_implement() {
    for (knob_name, knob_value) in [
        ("attention_bias", serde_json::json!(true)),
        ("query_key_norm", serde_json::json!(true)),
    ] {
        let mut document: serde_json::Value =
            serde_json::from_str(&support::family_member_config_json(2, &[0], 4, 2)).expect("json");
        document[knob_name] = knob_value;
        let parse_error = K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes())
            .expect_err("a declared-but-unimplemented knob must fail closed");
        assert!(
            parse_error.to_string().contains(knob_name),
            "the rejection must name the offending knob: {parse_error}"
        );
    }
    let mut document: serde_json::Value =
        serde_json::from_str(&support::family_member_config_json(2, &[0], 4, 2)).expect("json");
    document["num_shared_experts"] = serde_json::json!(2);
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
    document["num_shared_experts"] = serde_json::json!(0);
    let zero_error = K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes())
        .expect_err(
            "zero shared experts must fail closed because weight binding and \
                     the decoder execute exactly one shared expert",
        );
    assert!(
        zero_error.to_string().contains("num_shared_experts"),
        "the rejection must name the offending knob: {zero_error}"
    );
}

#[test]
fn should_reject_a_member_that_declares_no_rope_theta() {
    let mut document: serde_json::Value =
        serde_json::from_str(&support::family_member_config_json(2, &[0], 4, 2)).expect("json");
    document
        .as_object_mut()
        .expect("config document")
        .remove("rope_parameters");
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_err());
    document["rope_theta"] = serde_json::json!(10_000.0);
    assert!(K2HorizonMoVAConfig::from_json_bytes(document.to_string().as_bytes()).is_ok());
}
