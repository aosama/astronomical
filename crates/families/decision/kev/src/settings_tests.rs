//! Unit contracts for reading a Kev folder's settings in either layout.

use decision_contract::kev_folder_layout::KevFolderLayout;

use super::KevSettings;

const CONVERTED_SETTINGS: &str = r#"{"model_type": "kev", "temperature": 2.3510958125672174, "pointer_dimension": 256, "adapter_rank": 16}"#;
const PUBLISHED_SETTINGS: &str = r#"{"format": "abstractcore-decision/1", "kind": "kev", "head_dim": 256, "temperature": 2.3510958125672174, "head_file": "head.safetensors", "backbone": "merged"}"#;

// --- positive ---

#[test]
fn should_read_the_temperature_and_pointer_dimension() {
    let settings =
        KevSettings::from_json_bytes(KevFolderLayout::Converted, CONVERTED_SETTINGS.as_bytes())
            .expect("complete settings should read");

    assert_eq!(settings.temperature, 2.3510958125672174);
    assert_eq!(settings.pointer_dimension, 256);
}

#[test]
fn should_find_the_layout_a_folder_uses() {
    for (layout, settings_text) in [
        (KevFolderLayout::Converted, CONVERTED_SETTINGS),
        (KevFolderLayout::Published, PUBLISHED_SETTINGS),
    ] {
        let model_directory = tempfile::tempdir().unwrap();
        std::fs::write(
            model_directory.path().join(layout.settings_file_name()),
            settings_text,
        )
        .unwrap();

        let settings = KevSettings::read(model_directory.path()).unwrap();

        assert_eq!(settings.layout, layout);
        assert_eq!(settings.pointer_dimension, 256);
    }
}

// --- edge / boundary ---

#[test]
fn should_pass_over_a_decision_json_of_another_model() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join("decision.json"),
        PUBLISHED_SETTINGS.replace(r#""kind": "kev""#, r#""kind": "laya""#),
    )
    .unwrap();

    let failure = KevSettings::read(model_directory.path()).unwrap_err();

    assert!(format!("{failure:?}").contains("describes a Kev model"));
}

// --- negative ---

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
            KevSettings::from_json_bytes(KevFolderLayout::Converted, settings_json.as_bytes())
                .is_err(),
            "{settings_json} must be refused"
        );
    }
}

#[test]
fn should_refuse_a_folder_without_settings() {
    let model_directory = tempfile::tempdir().unwrap();

    assert!(KevSettings::read(model_directory.path()).is_err());
}
