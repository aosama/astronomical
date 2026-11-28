//! Unit contracts for the two Kev folder layouts.

use super::{KevFolderLayout, KevFolderSettings};

const CONVERTED_SETTINGS: &str = r#"{"model_type":"kev","temperature":2.35,"pointer_dimension":256,"base_model":"Qwen/Qwen3.5-0.8B-Base"}"#;
const PUBLISHED_SETTINGS: &str = r#"{"format":"abstractcore-decision/1","kind":"kev","head_dim":256,"temperature":2.35,"head_file":"head.safetensors","backbone":"merged","base":"Qwen/Qwen3.5-0.8B-Base"}"#;

fn expected_settings() -> KevFolderSettings {
    KevFolderSettings {
        temperature: 2.35,
        pointer_dimension: 256,
        base_model: Some("Qwen/Qwen3.5-0.8B-Base".to_owned()),
    }
}

// --- positive ---

#[test]
fn should_read_the_same_settings_from_both_layouts() {
    assert_eq!(
        KevFolderLayout::Converted.read_settings(CONVERTED_SETTINGS.as_bytes()),
        Ok(expected_settings())
    );
    assert_eq!(
        KevFolderLayout::Published.read_settings(PUBLISHED_SETTINGS.as_bytes()),
        Ok(expected_settings())
    );
}

#[test]
fn should_name_each_layouts_own_files() {
    assert_eq!(
        KevFolderLayout::Converted.settings_file_name(),
        "kev_decision.json"
    );
    assert_eq!(
        KevFolderLayout::Converted.head_file_name(),
        "kev_head.safetensors"
    );
    assert_eq!(
        KevFolderLayout::Published.settings_file_name(),
        "decision.json"
    );
    assert_eq!(
        KevFolderLayout::Published.head_file_name(),
        "head.safetensors"
    );
}

// --- edge / boundary ---

#[test]
fn should_check_the_converted_layout_first() {
    assert_eq!(
        KevFolderLayout::ALL,
        [KevFolderLayout::Converted, KevFolderLayout::Published]
    );
}

#[test]
fn should_read_a_converted_folder_that_names_no_base_model() {
    let settings = KevFolderLayout::Converted
        .read_settings(br#"{"model_type":"kev","temperature":1.0,"pointer_dimension":1}"#)
        .unwrap();

    assert_eq!(settings.base_model, None);
    assert_eq!(settings.pointer_dimension, 1);
}

#[test]
fn should_not_take_one_layouts_settings_for_the_other() {
    assert!(!KevFolderLayout::Published.marks_kev(CONVERTED_SETTINGS.as_bytes()));
    assert!(!KevFolderLayout::Converted.marks_kev(PUBLISHED_SETTINGS.as_bytes()));
}

// --- negative ---

#[test]
fn should_refuse_another_decision_kind_or_format() {
    let laya_settings = PUBLISHED_SETTINGS.replace(r#""kind":"kev""#, r#""kind":"laya""#);
    let other_format = PUBLISHED_SETTINGS.replace("abstractcore-decision/1", "other/1");

    assert!(!KevFolderLayout::Published.marks_kev(laya_settings.as_bytes()));
    assert!(!KevFolderLayout::Published.marks_kev(other_format.as_bytes()));
    assert!(!KevFolderLayout::Converted.marks_kev(b"not json"));
}

#[test]
fn should_refuse_a_published_folder_whose_head_or_backbone_it_cannot_load() {
    let other_head = PUBLISHED_SETTINGS.replace("head.safetensors", "head.pt");
    let adapter_only = PUBLISHED_SETTINGS.replace(r#""backbone":"merged""#, r#""backbone":"lora""#);

    assert!(
        KevFolderLayout::Published
            .read_settings(other_head.as_bytes())
            .unwrap_err()
            .contains("head_file")
    );
    assert!(
        KevFolderLayout::Published
            .read_settings(adapter_only.as_bytes())
            .unwrap_err()
            .contains("backbone")
    );
}

#[test]
fn should_refuse_a_temperature_or_head_width_that_is_not_positive() {
    let zero_temperature = CONVERTED_SETTINGS.replace("2.35", "0.0");
    let zero_head_width = PUBLISHED_SETTINGS.replace(r#""head_dim":256"#, r#""head_dim":0"#);

    assert!(
        KevFolderLayout::Converted
            .read_settings(zero_temperature.as_bytes())
            .unwrap_err()
            .contains("temperature")
    );
    assert!(
        KevFolderLayout::Published
            .read_settings(zero_head_width.as_bytes())
            .unwrap_err()
            .contains("head_dim")
    );
}
