//! Internal localhost `PUT /v1/config/sharing` endpoint.
//!
//! The chat app saves the person's sharing choices through it: "Send reports
//! like this without asking" on a problem report, and the picker in its
//! "What leaves this Mac" list. It is a route of its own because
//! `PUT /v1/config` admits only Observatory's page.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use config::ProblemReportChoice;
use serde::Deserialize;

use crate::application::ApplicationState;
use crate::rest::config_section_write::write_config_section;

/// Either choice, or both; a field left out keeps what is stored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SharingSettingRequest {
    problem_reports: Option<ProblemReportChoice>,
    daily_count: Option<bool>,
}

impl SharingSettingRequest {
    pub(super) fn requested_values(&self) -> Vec<(&'static str, serde_json::Value)> {
        let mut requested_values = Vec::new();
        if let Some(problem_report_choice) = self.problem_reports {
            requested_values.push((
                "problem_reports",
                serde_json::to_value(problem_report_choice).unwrap_or_default(),
            ));
        }
        if let Some(daily_count) = self.daily_count {
            requested_values.push(("daily_count", daily_count.into()));
        }
        requested_values
    }
}

pub(crate) async fn update_sharing_setting(
    State(application_state): State<ApplicationState>,
    Json(sharing_setting_request): Json<SharingSettingRequest>,
) -> Response {
    let requested_values = sharing_setting_request.requested_values();
    if requested_values.is_empty() {
        return (StatusCode::BAD_REQUEST, "name at least one sharing choice").into_response();
    }
    write_config_section(application_state, "sharing", &requested_values).await
}
