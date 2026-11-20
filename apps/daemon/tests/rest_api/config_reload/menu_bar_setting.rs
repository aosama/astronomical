//! `PUT /v1/config/menu-bar`: the menu-bar agent's two display switches.

use super::config_document::{
    application_with_config, config_file_text, json_of, send_json, valid_document,
};
use super::*;

fn saved_menu_bar_section(home_directory: &std::path::Path) -> serde_json::Value {
    let saved_document: serde_json::Value =
        serde_json::from_str(&config_file_text(home_directory)).expect("the saved file is JSON");
    saved_document["menu_bar"].clone()
}

#[tokio::test]
async fn should_save_one_switch_and_apply_it_without_replacing_the_worker() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, reloadable_config) = application_with_config(home_directory.path());

    let response = send_json(
        &application,
        "PUT",
        "/v1/config/menu-bar",
        &serde_json::json!({ "show_technical_details": true }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = json_of(response).await;
    assert_eq!(response_json["status"], "saved");
    assert_eq!(response_json["reload"]["worker_restart_completed"], false);
    assert_eq!(
        saved_menu_bar_section(home_directory.path()),
        serde_json::json!({ "show_technical_details": true })
    );
    let live_config = reloadable_config.read().expect("the live config reads");
    assert!(live_config.menu_bar_shows_technical_details);
    assert!(!live_config.menu_bar_shows_progress_and_speed);
}

#[tokio::test]
async fn should_keep_the_other_switch_and_every_other_setting_when_one_changes() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    let mut authored_document = valid_document();
    authored_document["menu_bar"] = serde_json::json!({ "show_progress_and_speed": true });
    authored_document["diagnostics"] = serde_json::json!({ "log_level": "info" });
    write_raw_config_file(home_directory.path(), &authored_document.to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());

    let response = send_json(
        &application,
        "PUT",
        "/v1/config/menu-bar",
        &serde_json::json!({ "show_technical_details": false }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        saved_menu_bar_section(home_directory.path()),
        serde_json::json!({ "show_progress_and_speed": true, "show_technical_details": false })
    );
    assert!(config_file_text(home_directory.path()).contains("\"log_level\": \"info\""));
}

#[tokio::test]
async fn should_refuse_a_request_that_names_no_switch_or_gives_one_a_wrong_type() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let file_before_requests = config_file_text(home_directory.path());

    let empty_response = send_json(
        &application,
        "PUT",
        "/v1/config/menu-bar",
        &serde_json::json!({}),
    )
    .await;
    assert_eq!(empty_response.status(), StatusCode::BAD_REQUEST);

    for refused_request in [
        serde_json::json!({ "show_technical_details": "yes" }),
        serde_json::json!({ "show_speed": true }),
    ] {
        let refused_response =
            send_json(&application, "PUT", "/v1/config/menu-bar", &refused_request).await;
        assert_eq!(
            refused_response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "{refused_request}"
        );
    }
    assert_eq!(
        config_file_text(home_directory.path()),
        file_before_requests
    );
}

#[tokio::test]
async fn should_send_a_person_to_settings_when_the_file_cannot_be_read() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), "{ not json");
    let (application, _reloadable_config) = application_with_config(home_directory.path());

    let response = send_json(
        &application,
        "PUT",
        "/v1/config/menu-bar",
        &serde_json::json!({ "show_progress_and_speed": true }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(config_file_text(home_directory.path()), "{ not json");
}
