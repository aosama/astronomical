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

/// The base geometry with its rotary bases in the nested form newer configs use.
fn config_with_nested_rope_parameters() -> serde_json::Value {
    let mut config_value = modernbert_base_config();
    let config_fields = config_value
        .as_object_mut()
        .expect("the fixture is an object");
    config_fields.remove("global_rope_theta");
    config_fields.remove("local_rope_theta");
    config_fields.insert(
        "rope_parameters".to_owned(),
        serde_json::json!({
            "full_attention": { "rope_theta": 160_000.0, "rope_type": "default" },
            "sliding_attention": { "rope_theta": 10_000.0, "rope_type": "default" },
        }),
    );
    config_value
}

#[test]
fn should_read_rotary_bases_from_the_nested_rope_parameters_form() {
    let configuration =
        ModernBertConfiguration::from_config_value(&config_with_nested_rope_parameters())
            .expect("the nested form is supported");
    assert_eq!(configuration.global_rope_theta, 160_000.0);
    assert_eq!(configuration.local_rope_theta, 10_000.0);
}

#[test]
fn should_prefer_the_flat_rotary_fields_when_both_forms_are_present() {
    let mut config_value = config_with_nested_rope_parameters();
    config_value["global_rope_theta"] = serde_json::json!(5.0);
    config_value["local_rope_theta"] = serde_json::json!(7.0);
    let configuration = ModernBertConfiguration::from_config_value(&config_value)
        .expect("both forms together are supported");
    assert_eq!(configuration.global_rope_theta, 5.0);
    assert_eq!(configuration.local_rope_theta, 7.0);
}

#[test]
fn should_refuse_a_config_with_neither_rotary_form() {
    let mut config_value = config_with_nested_rope_parameters();
    config_value
        .as_object_mut()
        .expect("the fixture is an object")
        .remove("rope_parameters");
    assert_eq!(
        parse_error_of(&config_value),
        ModernBertConfigurationError::MissingField {
            field_name: "global_rope_theta"
        }
    );
}

#[test]
fn should_refuse_a_nested_rotary_base_that_is_missing_or_not_positive() {
    let mut missing_layer_type = config_with_nested_rope_parameters();
    missing_layer_type["rope_parameters"]
        .as_object_mut()
        .expect("an object")
        .remove("sliding_attention");
    assert_eq!(
        parse_error_of(&missing_layer_type),
        ModernBertConfigurationError::MissingField {
            field_name: "local_rope_theta"
        }
    );
    let mut zero_base = config_with_nested_rope_parameters();
    zero_base["rope_parameters"]["full_attention"]["rope_theta"] = serde_json::json!(0.0);
    assert_eq!(
        parse_error_of(&zero_base),
        ModernBertConfigurationError::MissingField {
            field_name: "global_rope_theta"
        }
    );
}

#[test]
fn should_refuse_rope_parameters_that_are_not_an_object() {
    let mut config_value = config_with_nested_rope_parameters();
    config_value["rope_parameters"] = serde_json::json!("default");
    assert_eq!(
        parse_error_of(&config_value),
        ModernBertConfigurationError::MissingField {
            field_name: "global_rope_theta"
        }
    );
}
