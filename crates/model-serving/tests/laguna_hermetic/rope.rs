use astronomical_model_serving::{
    LagunaNormalizationError, LagunaRopeDescriptor, LagunaTargetNormalizer,
};
use serde_json::json;

use super::support::{config_bytes, config_value, normalize};

#[test]
fn should_normalize_flat_default_and_yarn_rope_without_pinned_context_equality() {
    let default_contract = normalize(config_value(2));
    assert!(matches!(
        default_contract.layers()[0].attention().rope(),
        LagunaRopeDescriptor::Default(_)
    ));
    assert_eq!(
        default_contract.layers()[0]
            .attention()
            .rope()
            .rotary_dimension(),
        64
    );

    let mut yarn_config = config_value(2);
    yarn_config["max_position_embeddings"] = json!(50_000);
    yarn_config["rope_parameters"] = yarn_parameters(2.0, 8_192, 0.5, 500_000.0);
    let yarn_contract = normalize(yarn_config);
    let LagunaRopeDescriptor::Yarn(yarn) = yarn_contract.layers()[0].attention().rope() else {
        panic!("flat YaRN parameters should normalize");
    };
    assert_eq!(yarn.factor(), 2.0);
    assert_eq!(yarn.original_maximum_position_count(), 8_192);
    assert_eq!(yarn.rotary_dimension(), 64);
}

#[test]
fn should_select_nested_per_kind_rope_for_the_active_attention_kind() {
    let mut config = config_value(2);
    config["layer_types"] = json!(["full", "sliding"]);
    config["sliding_window"] = json!(512);
    config["rope_parameters"] = json!({
        "full_attention": yarn_parameters(3.0, 4_096, 0.5, 500_000.0),
        "sliding_attention": {
            "type": "default",
            "rope_theta": 20_000.0,
            "partial_rotary_factor": 1.0
        }
    });
    let contract = normalize(config);

    assert!(matches!(
        contract.layers()[0].attention().rope(),
        LagunaRopeDescriptor::Yarn(_)
    ));
    assert_eq!(
        contract.layers()[1].attention().rope().rope_theta(),
        20_000.0
    );
    assert_eq!(
        contract.layers()[1].attention().rope().rotary_dimension(),
        128
    );
}

#[test]
fn should_apply_sliding_override_before_flat_and_flat_before_legacy_rope() {
    let mut config = config_value(2);
    config["layer_types"] = json!(["full", "sliding"]);
    config["sliding_window"] = json!(512);
    config["rope_parameters"] = json!({
        "type": "default",
        "rope_theta": 11_000.0,
        "partial_rotary_factor": 0.5
    });
    config["swa_rope_parameters"] = json!({
        "rope_type": "default",
        "rope_theta": 22_000.0,
        "partial_rotary_factor": 1.0
    });
    config["rope_scaling"] = yarn_parameters(9.0, 1_024, 0.25, 33_000.0);
    let contract = normalize(config);

    assert_eq!(
        contract.layers()[0].attention().rope().rope_theta(),
        11_000.0
    );
    assert_eq!(
        contract.layers()[1].attention().rope().rope_theta(),
        22_000.0
    );
}

#[test]
fn should_use_legacy_scaling_with_top_level_theta_and_partial_fallbacks() {
    let mut config = config_value(1);
    assert!(
        config
            .as_object_mut()
            .expect("fixture object")
            .remove("rope_parameters")
            .is_some()
    );
    config["rope_theta"] = json!(700_000.0);
    config["partial_rotary_factor"] = json!(0.5);
    config["rope_scaling"] = json!({
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 4_096,
        "beta_slow": 1.0,
        "beta_fast": 32.0,
        "attention_factor": 1.2
    });
    let contract = normalize(config);

    let LagunaRopeDescriptor::Yarn(yarn) = contract.layers()[0].attention().rope() else {
        panic!("legacy YaRN should normalize");
    };
    assert_eq!(yarn.rope_theta(), 700_000.0);
    assert_eq!(yarn.rotary_dimension(), 64);
}

#[test]
fn should_reject_non_integral_odd_zero_and_unsupported_rotary_declarations() {
    let invalid_rows = [
        ("partial_rotary_factor", json!(0.0)),
        ("partial_rotary_factor", json!(0.3)),
        ("partial_rotary_factor", json!(0.0078125)),
        ("rope_theta", json!(-1.0)),
    ];
    for (field_name, invalid_value) in invalid_rows {
        let mut config = config_value(1);
        config["rope_parameters"][field_name] = invalid_value;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::InvalidRopeValue { .. })
        ));
    }

    let mut unsupported = config_value(1);
    unsupported["rope_parameters"]["rope_type"] = json!("longrope");
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&unsupported)),
        Err(LagunaNormalizationError::UnsupportedValue { .. })
    ));
}

fn yarn_parameters(
    factor: f64,
    original_position_count: u32,
    partial_rotary_factor: f64,
    rope_theta: f64,
) -> serde_json::Value {
    json!({
        "rope_type": "yarn",
        "rope_theta": rope_theta,
        "factor": factor,
        "original_max_position_embeddings": original_position_count,
        "beta_slow": 1.0,
        "beta_fast": 32.0,
        "attention_factor": 1.2,
        "partial_rotary_factor": partial_rotary_factor
    })
}
