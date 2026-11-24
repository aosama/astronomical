use serde_json::json;

use super::*;

#[test]
fn should_read_a_multimodal_config() {
    let summary = ModelConfigSummary::from_config_value(&json!({
        "model_type": "qwen3_5",
        "hidden_size": 1,
        "image_token_id": 10,
        "video_token_id": 11,
        "vision_start_token_id": 12,
        "vision_end_token_id": 13,
        "text_config": {"hidden_size": 4096},
        "vision_config": {"model_type": "qwen3_5_vision", "depth": 27, "hidden_size": 1152,
                          "out_hidden_size": 4096, "patch_size": 16, "spatial_merge_size": 2},
    }));
    assert_eq!(
        summary.text_hidden_size,
        Some(4096),
        "the nested width wins"
    );
    assert!(summary.declares_image_tokens());
    let vision = summary.vision.unwrap();
    assert_eq!(vision.block_count, Some(27));
    assert_eq!(vision.output_hidden_size, Some(4096));
}

#[test]
fn should_fall_back_to_the_root_width_for_a_plain_text_config() {
    let summary = ModelConfigSummary::from_config_value(&json!({"hidden_size": 512}));
    assert_eq!(summary.text_hidden_size, Some(512));
    assert!(!summary.declares_image_tokens());
    assert_eq!(summary.vision, None);
}

#[test]
fn should_ignore_fields_of_the_wrong_type() {
    let summary = ModelConfigSummary::from_config_value(&json!({
        "hidden_size": "big", "image_token_id": -1, "vision_config": "none",
    }));
    assert_eq!(summary, ModelConfigSummary::default());
}
