//! Unit tests for reading a Laya artifact's `config.json`.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration::MINIMUM_APPLIED_TEMPERATURE;

use super::{LayaConfiguration, LayaConfigurationError};
use crate::modernbert::configuration::ModernBertConfigurationError;

/// A config shaped like the published English checkpoint's.
fn english_checkpoint_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "laya",
        "encoder": {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "max_position_embeddings": 8192,
            "local_attention": 128,
            "global_attn_every_n_layers": 3,
            "rope_parameters": {
                "full_attention": { "rope_theta": 160_000.0 },
                "sliding_attention": { "rope_theta": 10_000.0 },
            },
            "layer_norm_eps": 1e-5,
            "pad_token_id": 50_283,
            "cls_token_id": 50_281,
            "sep_token_id": 50_282,
        },
        "head_layers": 2,
        "max_len": 512,
        "head_max_len": 192,
        "temperature": [1.6, 1.25, 2.0],
        "temperature_by_options": { "choice:3-5": 1.75, "choice:11+": 0.1 },
        "quantization": { "group_size": 32, "bits": 8 },
    })
}

fn parse_error_of(config_value: &serde_json::Value) -> LayaConfigurationError {
    LayaConfiguration::from_config_value(config_value).expect_err("the config must be refused")
}

#[test]
fn should_read_the_published_checkpoint_shape() {
    let configuration = LayaConfiguration::from_config_value(&english_checkpoint_config())
        .expect("the published shape is supported");
    assert_eq!(configuration.encoder.hidden_size, 1024);
    assert_eq!(configuration.encoder.quantization_group_size, 32);
    assert_eq!(configuration.head_layer_count, 2);
    assert_eq!(configuration.sequence_limits.maximum_sequence_tokens, 512);
    assert_eq!(configuration.sequence_limits.head_token_budget, 192);
    assert_eq!(configuration.head_attention_head_count(), 16);
    let temperatures = &configuration.temperature_table;
    assert_eq!(
        temperatures.temperature_for(DecisionQuestionKind::Choice, 4),
        1.75
    );
    assert_eq!(
        temperatures.temperature_for(DecisionQuestionKind::Choice, 2),
        1.6
    );
    assert_eq!(
        temperatures.temperature_for(DecisionQuestionKind::Score, 3),
        1.25
    );
    assert_eq!(
        temperatures.temperature_for(DecisionQuestionKind::Noul, 2),
        2.0
    );
}

#[test]
fn should_clamp_a_sharpening_temperature_a_checkpoint_ships() {
    let configuration = LayaConfiguration::from_config_value(&english_checkpoint_config())
        .expect("the published shape is supported");
    assert_eq!(
        configuration
            .temperature_table
            .temperature_for(DecisionQuestionKind::Choice, 12),
        MINIMUM_APPLIED_TEMPERATURE
    );
}

#[test]
fn should_fall_back_to_defaults_for_omitted_head_settings() {
    let mut config_value = english_checkpoint_config();
    let config_fields = config_value.as_object_mut().expect("an object");
    for omitted_field in [
        "head_layers",
        "max_len",
        "head_max_len",
        "temperature",
        "temperature_by_options",
    ] {
        config_fields.remove(omitted_field);
    }
    let configuration =
        LayaConfiguration::from_config_value(&config_value).expect("defaults are supported");
    assert_eq!(configuration.head_layer_count, 2);
    assert_eq!(configuration.sequence_limits.maximum_sequence_tokens, 512);
    assert_eq!(configuration.sequence_limits.head_token_budget, 192);
    assert_eq!(
        configuration
            .temperature_table
            .temperature_for(DecisionQuestionKind::Noul, 2),
        1.0
    );
}

#[test]
fn should_accept_the_sequential_option_layout_and_refuse_any_other() {
    let mut sequential = english_checkpoint_config();
    sequential["option_layout"] = serde_json::json!("sequential");
    assert!(LayaConfiguration::from_config_value(&sequential).is_ok());
    let mut parallel = english_checkpoint_config();
    parallel["option_layout"] = serde_json::json!("parallel");
    assert!(matches!(
        parse_error_of(&parallel),
        LayaConfigurationError::UnsupportedOptionLayout { .. }
    ));
}

#[test]
fn should_accept_the_deepest_supported_head_and_refuse_one_layer_more() {
    let mut deepest = english_checkpoint_config();
    deepest["head_layers"] = serde_json::json!(8);
    assert!(LayaConfiguration::from_config_value(&deepest).is_ok());
    let mut too_deep = english_checkpoint_config();
    too_deep["head_layers"] = serde_json::json!(9);
    assert_eq!(
        parse_error_of(&too_deep),
        LayaConfigurationError::TooManyHeadLayers {
            actual: 9,
            maximum: 8
        }
    );
}

#[test]
fn should_accept_a_headless_checkpoint() {
    let mut headless = english_checkpoint_config();
    headless["head_layers"] = serde_json::json!(0);
    let configuration =
        LayaConfiguration::from_config_value(&headless).expect("zero head layers is supported");
    assert_eq!(configuration.head_layer_count, 0);
}

#[test]
fn should_refuse_sequence_limits_that_cannot_hold_a_question() {
    for (maximum_sequence_tokens, head_token_budget) in
        [(7, 4), (512, 512), (512, 600), (512, 0), (8193, 192)]
    {
        let mut config_value = english_checkpoint_config();
        config_value["max_len"] = serde_json::json!(maximum_sequence_tokens);
        config_value["head_max_len"] = serde_json::json!(head_token_budget);
        assert!(
            matches!(
                parse_error_of(&config_value),
                LayaConfigurationError::UnusableSequenceLimits { .. }
            ),
            "max_len {maximum_sequence_tokens} with head_max_len {head_token_budget} must be refused"
        );
    }
}

#[test]
fn should_accept_sequence_limits_at_both_boundaries() {
    let mut shortest = english_checkpoint_config();
    shortest["max_len"] = serde_json::json!(8);
    shortest["head_max_len"] = serde_json::json!(7);
    assert!(LayaConfiguration::from_config_value(&shortest).is_ok());
    let mut longest = english_checkpoint_config();
    longest["max_len"] = serde_json::json!(8192);
    assert!(LayaConfiguration::from_config_value(&longest).is_ok());
}

#[test]
fn should_refuse_a_limit_that_is_not_an_unsigned_integer() {
    let mut config_value = english_checkpoint_config();
    config_value["max_len"] = serde_json::json!("512");
    assert_eq!(
        parse_error_of(&config_value),
        LayaConfigurationError::InvalidField {
            field_name: "max_len"
        }
    );
}

#[test]
fn should_refuse_malformed_temperatures() {
    for malformed_temperatures in [
        serde_json::json!([1.0, 1.0]),
        serde_json::json!([1.0, "warm", 1.0]),
        serde_json::json!(1.5),
    ] {
        let mut config_value = english_checkpoint_config();
        config_value["temperature"] = malformed_temperatures;
        assert_eq!(
            parse_error_of(&config_value),
            LayaConfigurationError::MalformedTemperatures
        );
    }
    for malformed_buckets in [
        serde_json::json!([1.0]),
        serde_json::json!({ "choice:2": "warm" }),
    ] {
        let mut config_value = english_checkpoint_config();
        config_value["temperature_by_options"] = malformed_buckets;
        assert_eq!(
            parse_error_of(&config_value),
            LayaConfigurationError::MalformedBucketTemperatures
        );
    }
}

#[test]
fn should_refuse_a_config_without_an_encoder_or_without_quantization() {
    let mut without_encoder = english_checkpoint_config();
    without_encoder
        .as_object_mut()
        .expect("an object")
        .remove("encoder");
    assert_eq!(
        parse_error_of(&without_encoder),
        LayaConfigurationError::MissingEncoder
    );
    let mut without_quantization = english_checkpoint_config();
    without_quantization
        .as_object_mut()
        .expect("an object")
        .remove("quantization");
    assert_eq!(
        parse_error_of(&without_quantization),
        LayaConfigurationError::Encoder(ModernBertConfigurationError::MissingQuantization)
    );
}

#[test]
fn should_refuse_a_hidden_size_the_head_cannot_split_into_equal_heads() {
    // 200 hidden channels give three heads of 64, which leaves a remainder.
    let mut config_value = english_checkpoint_config();
    config_value["encoder"]["hidden_size"] = serde_json::json!(200);
    config_value["encoder"]["num_attention_heads"] = serde_json::json!(8);
    config_value["quantization"] = serde_json::json!({ "group_size": 8, "bits": 8 });
    assert_eq!(
        parse_error_of(&config_value),
        LayaConfigurationError::UnevenHeadAttentionWidth {
            hidden_size: 200,
            head_count: 3
        }
    );
}

#[test]
fn should_give_a_narrow_encoder_one_head() {
    let mut config_value = english_checkpoint_config();
    config_value["encoder"]["hidden_size"] = serde_json::json!(32);
    config_value["encoder"]["num_attention_heads"] = serde_json::json!(2);
    let configuration =
        LayaConfiguration::from_config_value(&config_value).expect("a narrow encoder is supported");
    assert_eq!(configuration.head_attention_head_count(), 1);
}
