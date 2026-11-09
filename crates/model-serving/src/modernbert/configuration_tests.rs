use super::{ModernBertConfiguration, ModernBertConfigurationError};

/// ModernBERT-base geometry with the 8-bit quantization the Library ships.
fn modernbert_base_config() -> serde_json::Value {
    serde_json::json!({
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "max_position_embeddings": 8192,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160_000.0,
        "local_rope_theta": 10_000.0,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 50_283,
        "cls_token_id": 50_281,
        "sep_token_id": 50_282,
        "quantization": { "group_size": 64, "bits": 8 },
    })
}

fn parse_error_of(config_value: &serde_json::Value) -> ModernBertConfigurationError {
    ModernBertConfiguration::from_config_value(config_value)
        .expect_err("the configuration must be refused")
}

#[test]
fn should_parse_the_base_geometry_and_derive_its_head_dimension() {
    let configuration = ModernBertConfiguration::from_config_value(&modernbert_base_config())
        .expect("the base geometry is supported");
    assert_eq!(configuration.head_dimension(), 64);
    assert_eq!(
        (
            configuration.quantization_group_size,
            configuration.quantization_bits
        ),
        (64, 8)
    );
}

#[test]
fn should_report_an_absent_quantization_block_or_field_as_missing_quantization() {
    let mut unquantized = modernbert_base_config();
    unquantized
        .as_object_mut()
        .expect("the fixture is an object")
        .remove("quantization");
    assert_eq!(
        parse_error_of(&unquantized),
        ModernBertConfigurationError::MissingQuantization
    );
    let mut bits_only = modernbert_base_config();
    bits_only["quantization"] = serde_json::json!({ "bits": 8 });
    assert_eq!(
        parse_error_of(&bits_only),
        ModernBertConfigurationError::MissingQuantization
    );
}

#[test]
fn should_name_a_zero_quantization_field_rather_than_report_it_missing() {
    let mut zero_bits = modernbert_base_config();
    zero_bits["quantization"]["bits"] = serde_json::json!(0);
    assert_eq!(
        parse_error_of(&zero_bits),
        ModernBertConfigurationError::MissingField { field_name: "bits" }
    );
}

#[test]
fn should_accept_the_largest_supported_geometry_and_refuse_one_more() {
    let mut largest_geometry = modernbert_base_config();
    largest_geometry["hidden_size"] = serde_json::json!(4_096);
    largest_geometry["num_hidden_layers"] = serde_json::json!(64);
    largest_geometry["num_attention_heads"] = serde_json::json!(32);
    assert!(ModernBertConfiguration::from_config_value(&largest_geometry).is_ok());

    let mut one_layer_more = largest_geometry.clone();
    one_layer_more["num_hidden_layers"] = serde_json::json!(65);
    assert!(matches!(
        parse_error_of(&one_layer_more),
        ModernBertConfigurationError::LayerCountExceeded {
            actual_layer_count: 65,
            ..
        }
    ));
}

#[test]
fn should_refuse_heads_that_do_not_tile_the_hidden_size() {
    let mut eleven_heads = modernbert_base_config();
    eleven_heads["num_attention_heads"] = serde_json::json!(11);
    assert!(matches!(
        parse_error_of(&eleven_heads),
        ModernBertConfigurationError::HiddenSizeNotDivisibleByHeads { .. }
    ));
}

#[test]
fn should_refuse_quantization_beyond_eight_bits_or_misaligned_groups() {
    let mut sixteen_bits = modernbert_base_config();
    sixteen_bits["quantization"]["bits"] = serde_json::json!(16);
    assert!(matches!(
        parse_error_of(&sixteen_bits),
        ModernBertConfigurationError::UnsupportedQuantization {
            quantization_bits: 16,
            ..
        }
    ));
    let mut misaligned_group = modernbert_base_config();
    misaligned_group["quantization"]["group_size"] = serde_json::json!(80);
    assert!(matches!(
        parse_error_of(&misaligned_group),
        ModernBertConfigurationError::QuantizationGroupMisaligned { .. }
    ));
}

#[test]
fn should_refuse_a_vanishing_layer_norm_epsilon() {
    let mut vanishing_epsilon = modernbert_base_config();
    vanishing_epsilon["layer_norm_eps"] = serde_json::json!(1e-12);
    assert!(matches!(
        parse_error_of(&vanishing_epsilon),
        ModernBertConfigurationError::UnsupportedLayerNormEpsilon { .. }
    ));
}
