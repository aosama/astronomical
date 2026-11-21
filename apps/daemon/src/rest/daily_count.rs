//! `GET /v1/daily-count`: what the chat app needs to ask "May this Mac be
//! counted?" and to show the choice in Settings: the exact line that would
//! be sent, the person's answer, whether to ask now, and the last day a
//! count went. The answer itself is saved through `PUT /v1/config/sharing`.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::application::ApplicationState;
use crate::daily_count::{DailyCounter, unix_day_now};

#[derive(Serialize)]
struct DailyCountView {
    /// The line this Mac sends, with its real values.
    line: String,
    /// `true`, `false`, or `null` while the person has not answered.
    answer: Option<bool>,
    should_ask: bool,
    /// Days since 1970-01-01 (UTC) of the last count sent, if any.
    last_sent_unix_day: Option<u64>,
}

pub(crate) async fn show_daily_count(
    State(application_state): State<ApplicationState>,
) -> Response {
    // No report desk means the Mac App Store build, which counts nothing.
    let state_directory = application_state
        .runtime_config_resolver
        .as_ref()
        .filter(|_| application_state.problem_report_desk.is_some())
        .map(|runtime_config_resolver| runtime_config_resolver.state_directory().to_path_buf());
    let Some(daily_counter) = state_directory
        .as_deref()
        .and_then(DailyCounter::for_this_mac)
    else {
        return (StatusCode::NOT_FOUND, "the daily count is unavailable").into_response();
    };
    let today_unix_day = unix_day_now();
    let daily_count_state = daily_counter.state(today_unix_day);
    let answer = application_state.daily_count_answer();
    Json(DailyCountView {
        line: daily_counter.line().readable_line(),
        answer,
        should_ask: daily_count_state.should_ask(answer, today_unix_day),
        last_sent_unix_day: daily_count_state.last_sent_unix_day,
    })
    .into_response()
}
