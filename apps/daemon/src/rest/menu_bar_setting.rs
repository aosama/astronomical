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
use crate::rest::config_section_write::write_config_section;

/// Either switch, or both; a field left out keeps what is stored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MenuBarSettingRequest {
    show_progress_and_speed: Option<bool>,
    show_technical_details: Option<bool>,
}

impl MenuBarSettingRequest {
    pub(super) fn requested_switches(&self) -> Vec<(&'static str, serde_json::Value)> {
        let mut requested_switches = Vec::new();
        if let Some(show_progress_and_speed) = self.show_progress_and_speed {
            requested_switches.push(("show_progress_and_speed", show_progress_and_speed.into()));
        }
        if let Some(show_technical_details) = self.show_technical_details {
            requested_switches.push(("show_technical_details", show_technical_details.into()));
        }
        requested_switches
    }
}

pub(crate) async fn update_menu_bar_setting(
    State(application_state): State<ApplicationState>,
    Json(menu_bar_setting_request): Json<MenuBarSettingRequest>,
) -> Response {
    let requested_switches = menu_bar_setting_request.requested_switches();
    if requested_switches.is_empty() {
        return (StatusCode::BAD_REQUEST, "name at least one menu bar switch").into_response();
    }
    write_config_section(application_state, "menu_bar", &requested_switches).await
}
