//! Unit contracts for reading `kev_decision.json`.

use super::KevSettings;

#[test]
fn should_read_the_temperature_and_pointer_dimension() {
    let settings = KevSettings::from_json_bytes(
        br#"{"model_type": "kev", "temperature": 2.3510958125672174, "pointer_dimension": 256, "adapter_rank": 16}"#,
    )
    .expect("complete settings should read");

    assert_eq!(settings.temperature, 2.3510958125672174);
    assert_eq!(settings.pointer_dimension, 256);
}

#[test]
fn should_refuse_settings_of_another_family_or_without_usable_values() {
    for settings_json in [
        r#"{"model_type": "laya", "temperature": 1.0, "pointer_dimension": 256}"#,
        r#"{"model_type": "kev", "temperature": 0.0, "pointer_dimension": 256}"#,
        r#"{"model_type": "kev", "temperature": -1.0, "pointer_dimension": 256}"#,
        r#"{"model_type": "kev", "temperature": 1.0, "pointer_dimension": 0}"#,
        r#"{"model_type": "kev", "pointer_dimension": 256}"#,
        "not json",
    ] {
        assert!(
            KevSettings::from_json_bytes(settings_json.as_bytes()).is_err(),
            "{settings_json} must be refused"
        );
    }
}
