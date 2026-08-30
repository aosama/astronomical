use astronomical_model_serving::Qwen3_5VisionConfig;

use crate::qwen3_5_hermetic::vision_config_test_support::FROZEN_VISION_CONFIG_JSON;

#[test]
fn should_accept_the_dense_qwen3_5_vision_model_type() {
    let config_bytes = FROZEN_VISION_CONFIG_JSON.replace(
        "\"model_type\": \"qwen3_5_moe_vision\"",
        "\"model_type\": \"qwen3_5_vision\"",
    );

    let vision_config = Qwen3_5VisionConfig::from_json_bytes(config_bytes.as_bytes())
        .expect("the dense Qwen3.5 vision model type should parse");

    assert_eq!(vision_config.out_hidden_size(), 2048);
}
