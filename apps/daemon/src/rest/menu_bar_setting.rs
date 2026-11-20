//! Internal localhost `PUT /v1/config/menu-bar` endpoint.
//!
//! The menu-bar agent's popover changes its two display switches through this
//! endpoint: the text beside the menu bar icon, and the popover's technical
//! rows. It edits only the named switches in `config.json` and then takes the same
//! validation, save and reload path as Observatory's Settings page, so the two
//! places always show the same stored value. It is a route of its own because
//! `PUT /v1/config` admits only this instance's console page: the agent is not a
//! browser and sends no origin.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use crate::application::ApplicationState;
use crate::rest::config_document::{
    reload_after_write, validate_candidate_resolves, write_error_response,
};

/// Either switch, or both; a field left out keeps what is stored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MenuBarSettingRequest {
    show_progress_and_speed: Option<bool>,
    show_technical_details: Option<bool>,
}

impl MenuBarSettingRequest {
    pub(super) fn requested_switches(&self) -> Vec<(&'static str, bool)> {
        let mut requested_switches = Vec::new();
        if let Some(show_progress_and_speed) = self.show_progress_and_speed {
            requested_switches.push(("show_progress_and_speed", show_progress_and_speed));
        }
        if let Some(show_technical_details) = self.show_technical_details {
            requested_switches.push(("show_technical_details", show_technical_details));
        }
        requested_switches
    }
}

/// The current document with the requested menu-bar switches replaced. A
/// document that is not a JSON object is returned unchanged, so validation
/// names the problem.
pub(super) fn document_with_menu_bar_settings(
    mut config_document: serde_json::Value,
    requested_switches: &[(&str, bool)],
) -> serde_json::Value {
    if let Some(document_members) = config_document.as_object_mut() {
        if !document_members.contains_key("menu_bar") {
            document_members.insert("menu_bar".to_owned(), serde_json::json!({}));
        }
        if let Some(menu_bar_members) = document_members
            .get_mut("menu_bar")
            .and_then(serde_json::Value::as_object_mut)
        {
            for (switch_name, setting) in requested_switches {
                menu_bar_members
                    .insert((*switch_name).to_owned(), serde_json::Value::Bool(*setting));
            }
        }
    }
    config_document
}

pub(crate) async fn update_menu_bar_setting(
    State(application_state): State<ApplicationState>,
    Json(menu_bar_setting_request): Json<MenuBarSettingRequest>,
) -> Response {
    let requested_switches = menu_bar_setting_request.requested_switches();
    if requested_switches.is_empty() {
        return (StatusCode::BAD_REQUEST, "name at least one menu bar switch").into_response();
    }
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.clone() else {
        return (StatusCode::NOT_FOUND, "config persistence is unavailable").into_response();
    };
    let document_digest = {
        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        let snapshot = match config::read_config_document(runtime_config_resolver.state_directory())
        {
            Ok(snapshot) => snapshot,
            Err(configuration_error) => return write_error_response(configuration_error),
        };
        let Some(current_document) = snapshot.document else {
            // A missing or unreadable file is repaired on the Settings page, which can show
            // what is wrong with it; writing one field over it here would hide that.
            return (
                StatusCode::CONFLICT,
                "config.json could not be read; open Settings in Observatory to repair it",
            )
                .into_response();
        };
        let candidate_document =
            document_with_menu_bar_settings(current_document, &requested_switches);
        if let Err(refusal_response) =
            validate_candidate_resolves(&runtime_config_resolver, &candidate_document)
        {
            return *refusal_response;
        }
        match config::save_config_document(
            runtime_config_resolver.state_directory(),
            snapshot.document_digest.as_deref(),
            &candidate_document,
        ) {
            Ok(document_digest) => document_digest,
            Err(configuration_error) => return write_error_response(configuration_error),
        }
    };
    reload_after_write(application_state, document_digest).await
}
