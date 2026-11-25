use serde_json::json;

use super::{recognizes_model_type, serves_vision};

#[test]
fn should_recognize_flash_next_beside_the_qwen_3_5_model_types() {
    assert!(recognizes_model_type(Some("qwen4_exp")));
    assert!(recognizes_model_type(Some("qwen3_5_moe")));
}

#[test]
fn should_not_recognize_a_missing_or_neighbouring_model_type() {
    assert!(!recognizes_model_type(None));
    assert!(!recognizes_model_type(Some("qwen4_exp_text")));
    assert!(!recognizes_model_type(Some("")));
}

#[test]
fn should_serve_a_qwen_3_5_vision_tower() {
    assert!(serves_vision(true, &json!({"model_type": "qwen3_5_moe"})));
}

#[test]
fn should_not_serve_flash_next_images_even_with_a_vision_tower() {
    assert!(!serves_vision(true, &json!({"model_type": "qwen4_exp"})));
}

#[test]
fn should_not_serve_vision_without_a_tower_or_a_model_type() {
    assert!(!serves_vision(false, &json!({"model_type": "qwen3_5_moe"})));
    assert!(serves_vision(true, &json!({})));
}
