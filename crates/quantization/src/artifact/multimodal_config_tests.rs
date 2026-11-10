use super::{register_vision_sidecar, strip_multimodal_claims};

fn written_config(config_text: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, config_text).expect("the config should be writable");
    (directory, config_path)
}

fn read_config(config_path: &std::path::Path) -> serde_json::Value {
    let config_text = std::fs::read_to_string(config_path).expect("the config should be readable");
    serde_json::from_str(&config_text).expect("the config should parse")
}

#[test]
fn should_register_the_sidecar_and_keep_the_vision_config() {
    let (_directory, config_path) =
        written_config(r#"{"model_type": "qwen3_5", "vision_config": {"depth": 12}}"#);

    register_vision_sidecar(
        &config_path,
        "optiq_vision.safetensors",
        153,
        "Qwen/Qwen3.5-0.8B",
    )
    .expect("the registration should be written");

    let config = read_config(&config_path);
    assert_eq!(
        config["optiq_vision"]["sidecar"],
        "optiq_vision.safetensors"
    );
    assert_eq!(config["optiq_vision"]["dtype"], "bfloat16");
    assert_eq!(config["optiq_vision"]["n_tensors"], 153);
    assert_eq!(config["optiq_vision"]["base_model"], "Qwen/Qwen3.5-0.8B");
    assert_eq!(config["vision_config"]["depth"], 12);
}

#[test]
fn should_strip_every_multimodal_claim_and_keep_the_rest() {
    let (_directory, config_path) = written_config(
        r#"{"model_type": "qwen3_5", "vision_config": {}, "image_token_id": 7,
            "video_token_id": 8, "vision_start_token_id": 9, "vision_end_token_id": 10}"#,
    );

    let removed_field_count =
        strip_multimodal_claims(&config_path).expect("the claims should be stripped");

    let config = read_config(&config_path);
    assert_eq!(removed_field_count, 5);
    assert_eq!(config, serde_json::json!({"model_type": "qwen3_5"}));
}

#[test]
fn should_leave_a_text_only_config_untouched() {
    let config_text = r#"{"model_type": "qwen3_5"}"#;
    let (_directory, config_path) = written_config(config_text);

    let removed_field_count =
        strip_multimodal_claims(&config_path).expect("a text-only config should be accepted");

    assert_eq!(removed_field_count, 0);
    let unchanged_text =
        std::fs::read_to_string(&config_path).expect("the config should be readable");
    assert_eq!(unchanged_text, config_text);
}

#[test]
fn should_reject_a_config_that_is_not_json() {
    let (_directory, config_path) = written_config("not json");

    let registration = register_vision_sidecar(&config_path, "optiq_vision.safetensors", 1, "base");

    assert!(matches!(
        registration,
        Err(crate::error::QuantizationError::ModelConfigUnreadable { .. })
    ));
}
