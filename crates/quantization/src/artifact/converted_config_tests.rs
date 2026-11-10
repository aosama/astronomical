use super::normalize_converted_config;

fn normalized(config_text: &str) -> serde_json::Value {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, config_text).expect("the config should be writable");
    normalize_converted_config(&config_path).expect("the config should normalize");
    let normalized_text =
        std::fs::read_to_string(&config_path).expect("the config should be readable");
    serde_json::from_str(&normalized_text).expect("the config should parse")
}

#[test]
fn should_spell_the_rope_kind_as_type_for_the_qwen_family() {
    let config = normalized(
        r#"{"model_type": "qwen3_5",
            "text_config": {"rope_parameters": {"rope_type": "default", "rope_theta": 10000000}}}"#,
    );

    let rope_parameters = &config["text_config"]["rope_parameters"];
    assert_eq!(rope_parameters["type"], "default");
    assert!(rope_parameters.get("rope_type").is_none());
    assert_eq!(rope_parameters["rope_theta"], 10_000_000);
}

#[test]
fn should_keep_an_existing_type_and_leave_the_old_key_beside_it() {
    let config = normalized(
        r#"{"model_type": "qwen3_5_text",
            "rope_parameters": {"type": "yarn", "rope_type": "default"}}"#,
    );

    assert_eq!(config["rope_parameters"]["type"], "yarn");
    assert_eq!(config["rope_parameters"]["rope_type"], "default");
}

#[test]
fn should_not_rename_for_another_family_but_still_drop_the_loaded_from_path() {
    let config = normalized(
        r#"{"model_type": "gemma4", "_name_or_path": "somewhere/local",
            "rope_parameters": {"rope_type": "default"}}"#,
    );

    assert_eq!(config["rope_parameters"]["rope_type"], "default");
    assert!(config.get("_name_or_path").is_none());
}

#[test]
fn should_reject_a_config_that_is_not_an_object() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, "[1, 2]").expect("the config should be writable");

    let normalization = normalize_converted_config(&config_path);

    assert!(matches!(
        normalization,
        Err(crate::error::QuantizationError::ModelConfigUnreadable { .. })
    ));
}
