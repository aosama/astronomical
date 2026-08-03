use astronomical_model_serving::Qwen3_5VisionConfig;

use super::vision_config_test_support::CERTIFIED_VISION_CONFIG_JSON;

#[test]
fn should_parse_the_certified_ornith_vision_config() {
    let vision_config =
        Qwen3_5VisionConfig::from_json_bytes(CERTIFIED_VISION_CONFIG_JSON.as_bytes())
            .expect("the certified Ornith config should parse vision_config");

    assert_eq!(vision_config.depth(), 27);
    assert_eq!(vision_config.hidden_size(), 1152);
    assert_eq!(vision_config.in_channels(), 3);
    assert_eq!(vision_config.intermediate_size(), 4304);
    assert_eq!(vision_config.head_count(), 16);
    assert_eq!(vision_config.position_embedding_count(), 2304);
    assert_eq!(vision_config.patch_size(), 16);
    assert_eq!(vision_config.spatial_merge_size(), 2);
    assert_eq!(vision_config.temporal_patch_size(), 2);
    assert_eq!(vision_config.out_hidden_size(), 2048);
    assert_eq!(vision_config.hidden_activation(), "gelu_pytorch_tanh");
}

#[test]
fn should_allow_a_text_only_qwen_config_without_vision_config() {
    let text_only_config_json = r#"{
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text"
        }
    }"#;

    let vision_config =
        Qwen3_5VisionConfig::from_optional_json_bytes(text_only_config_json.as_bytes())
            .expect("a text-only Qwen config should not require vision metadata");

    assert!(vision_config.is_none());
}

#[test]
fn should_accept_an_ornith_vision_config_with_a_different_depth() {
    let config_bytes = CERTIFIED_VISION_CONFIG_JSON.replace("\"depth\": 27", "\"depth\": 32");
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(config_bytes.as_bytes())
        .expect("vision config with a different depth should be accepted");
    assert_eq!(vision_config.depth(), 32);
}

#[test]
fn should_accept_an_ornith_vision_config_with_a_different_hidden_size() {
    let config_bytes =
        CERTIFIED_VISION_CONFIG_JSON.replace("\"hidden_size\": 1152", "\"hidden_size\": 1024");
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(config_bytes.as_bytes())
        .expect("vision config with a different hidden_size should be accepted");
    assert_eq!(vision_config.hidden_size(), 1024);
}

#[test]
fn should_accept_an_ornith_vision_config_with_a_different_patch_size() {
    let config_bytes =
        CERTIFIED_VISION_CONFIG_JSON.replace("\"patch_size\": 16", "\"patch_size\": 14");
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(config_bytes.as_bytes())
        .expect("vision config with a different patch_size should be accepted");
    assert_eq!(vision_config.patch_size(), 14);
}

#[test]
fn should_reject_an_ornith_vision_config_with_the_wrong_model_type() {
    let config_bytes = CERTIFIED_VISION_CONFIG_JSON.replace(
        "\"model_type\": \"qwen3_5_moe_vision\"",
        "\"model_type\": \"wrong_vision\"",
    );
    let result = Qwen3_5VisionConfig::from_json_bytes(config_bytes.as_bytes());
    assert!(
        result.is_err(),
        "vision config with wrong model_type should be rejected"
    );
}
