//! Unit tests for reading `joint_head_config.json`.

use super::ClefHeadSettings;

const CLEF_FLASH_SETTINGS: &str = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}"#;

#[test]
fn should_read_the_released_head_shape() {
    let head_settings = ClefHeadSettings::from_json_bytes(CLEF_FLASH_SETTINGS.as_bytes())
        .expect("the released settings read");

    assert_eq!(head_settings.hidden_size, 4096);
    assert_eq!(head_settings.width, 1024);
    assert_eq!(head_settings.routing_layer_count, 2);
    assert_eq!(head_settings.decoder_layer_count, 4);
    assert_eq!(head_settings.attention_head_count, 16);
    assert_eq!(head_settings.feedforward_width, 4096);
}

#[test]
fn should_refuse_a_width_the_heads_do_not_divide() {
    let uneven = CLEF_FLASH_SETTINGS.replace("\"heads\": 16", "\"heads\": 3");

    assert!(ClefHeadSettings::from_json_bytes(uneven.as_bytes()).is_err());
}

#[test]
fn should_refuse_a_missing_zero_or_malformed_size() {
    let missing = CLEF_FLASH_SETTINGS.replace("\"layers\": 4, ", "");
    let zero = CLEF_FLASH_SETTINGS.replace("\"width\": 1024", "\"width\": 0");

    assert!(ClefHeadSettings::from_json_bytes(missing.as_bytes()).is_err());
    assert!(ClefHeadSettings::from_json_bytes(zero.as_bytes()).is_err());
    assert!(ClefHeadSettings::from_json_bytes(b"not json").is_err());
}
