//! Unit tests for the menu-bar switch edits behind `PUT /v1/config/menu-bar`.

use super::menu_bar_setting::{MenuBarSettingRequest, document_with_menu_bar_settings};

#[test]
fn should_add_the_menu_bar_section_when_the_document_has_none() {
    let edited_document = document_with_menu_bar_settings(
        serde_json::json!({"schema_version": 1}),
        &[("show_progress_and_speed", true)],
    );
    assert_eq!(
        edited_document,
        serde_json::json!({"schema_version": 1, "menu_bar": {"show_progress_and_speed": true}})
    );
}

#[test]
fn should_replace_only_the_named_switch_and_keep_every_other_setting() {
    let edited_document = document_with_menu_bar_settings(
        serde_json::json!({
            "runtime": {"model_directories": []},
            "menu_bar": {"show_progress_and_speed": true, "show_technical_details": false}
        }),
        &[("show_technical_details", true)],
    );
    assert_eq!(
        edited_document,
        serde_json::json!({
            "runtime": {"model_directories": []},
            "menu_bar": {"show_progress_and_speed": true, "show_technical_details": true}
        })
    );
}

#[test]
fn should_leave_a_document_that_is_not_an_object_for_validation_to_refuse() {
    let edited_document = document_with_menu_bar_settings(
        serde_json::json!([1, 2]),
        &[("show_progress_and_speed", true)],
    );
    assert_eq!(edited_document, serde_json::json!([1, 2]));
}

#[test]
fn should_name_only_the_switches_a_request_carries() {
    let request: MenuBarSettingRequest =
        serde_json::from_str(r#"{"show_technical_details":false}"#).expect("request parses");
    assert_eq!(
        request.requested_switches(),
        vec![("show_technical_details", false)]
    );
    let empty_request: MenuBarSettingRequest =
        serde_json::from_str("{}").expect("an empty request parses");
    assert!(empty_request.requested_switches().is_empty());
}
