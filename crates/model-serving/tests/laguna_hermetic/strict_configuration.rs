use astronomical_model_serving::{LagunaNormalizationError, LagunaTargetNormalizer};
use serde_json::json;

use super::support::{config_bytes, config_value, normalize};

#[test]
fn should_reject_duplicate_fields_at_every_configuration_depth() {
    let duplicate_root_field = append_raw_root_field(config_value(1), r#""hidden_size":2048"#);
    let duplicate_nested_field = append_raw_root_field(
        config_value(1),
        r#""future_metadata":{"mode":"first","mode":"second"}"#,
    );
    let duplicate_quantization_field = append_raw_root_field(
        config_value(1),
        r#""quantization_config":{"quant_method":"compressed-tensors","format":"float-quantized","format":"pack-quantized"}"#,
    );

    for config_json_bytes in [
        duplicate_root_field,
        duplicate_nested_field,
        duplicate_quantization_field,
    ] {
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_json_bytes),
            Err(LagunaNormalizationError::DuplicateConfigField)
        ));
    }
}

#[test]
fn should_accept_only_the_execution_semantics_represented_by_the_contract() {
    let baseline_contract = normalize(config_value(1));
    let mut explicit_supported_config = config_value(1);
    explicit_supported_config["qkv_bias"] = json!(false);
    explicit_supported_config["rope_style"] = json!("rotate-half");
    explicit_supported_config["swa_attention_sink_enabled"] = json!(false);
    explicit_supported_config["moe_router_use_sigmoid"] = json!(true);
    explicit_supported_config["hidden_act"] = json!("silu");
    explicit_supported_config["use_bidirectional_attention"] = json!(false);

    assert_eq!(normalize(explicit_supported_config), baseline_contract);

    for (field_name, unsupported_value) in [
        ("qkv_bias", json!(true)),
        ("rope_style", json!("interleaved")),
        ("swa_attention_sink_enabled", json!(true)),
        ("moe_router_use_sigmoid", json!(false)),
        ("hidden_act", json!("gelu")),
        ("use_bidirectional_attention", json!(true)),
    ] {
        let mut unsupported_config = config_value(1);
        unsupported_config[field_name] = unsupported_value;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&unsupported_config)),
            Err(LagunaNormalizationError::UnsupportedValue {
                field_name: rejected_field_name,
                ..
            }) if rejected_field_name == field_name
        ));
    }
}

#[test]
fn should_bound_user_controlled_values_in_normalization_errors() {
    let mut unsupported_config = config_value(1);
    unsupported_config["rope_style"] = json!("x".repeat(4_096));

    let error = LagunaTargetNormalizer::normalize(&config_bytes(&unsupported_config))
        .expect_err("the unsupported rope style should fail");
    let LagunaNormalizationError::UnsupportedValue { actual_value, .. } = error else {
        panic!("the unsupported rope style should retain a typed error");
    };
    assert!(actual_value.chars().count() <= 257);
}

fn append_raw_root_field(config: serde_json::Value, raw_field: &str) -> Vec<u8> {
    let mut config_json_bytes = config_bytes(&config);
    assert_eq!(config_json_bytes.pop(), Some(b'}'));
    config_json_bytes.push(b',');
    config_json_bytes.extend_from_slice(raw_field.as_bytes());
    config_json_bytes.push(b'}');
    config_json_bytes
}
