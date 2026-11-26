use serde_json::json;

use super::serves_vision;

#[test]
fn should_serve_vision_when_the_pack_declares_its_tower_and_settings() {
    let config_value = json!({
        "components": {"text": true, "vision": true, "mtp": false},
        "vision_config": {"depth": 27, "hidden_size": 1152}
    });

    assert!(serves_vision(&config_value));
}

#[test]
fn should_not_serve_vision_when_the_pack_turns_its_tower_off() {
    let config_value = json!({
        "components": {"text": true, "vision": false},
        "vision_config": {"depth": 27}
    });

    assert!(!serves_vision(&config_value));
}

#[test]
fn should_not_serve_vision_without_the_towers_settings_or_components() {
    assert!(!serves_vision(&json!({"components": {"vision": true}})));
    assert!(!serves_vision(&json!({"vision_config": {"depth": 27}})));
    assert!(!serves_vision(
        &json!({"components": {"vision": "yes"}, "vision_config": {}})
    ));
}
