//! The local API the apps use for problem reports:
//!
//! - `GET /v1/problem-reports`: the person's choice, whether sending is set
//!   up in this build, the reports waiting for an answer and every report
//!   already sent, each as the lines the person reads.
//! - `POST /v1/problem-reports`: the person chose "Report a problem"; answers
//!   with the report they are about to describe.
//! - `POST /v1/problem-reports/{report_number}/send`: the person pressed
//!   Send report; the body may carry `{"note": "..."}`, what they typed.
//! - `POST /v1/problem-reports/{report_number}/decline`: they pressed Don't
//!   send.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use config::ProblemReportChoice;
use serde::Serialize;

use crate::application::ApplicationState;
use crate::problem_reports::{ReportSendOutcome, SentProblemReport};

#[derive(Serialize)]
struct WaitingReportView {
    report_number: u64,
    readable_lines: Vec<String>,
}

#[derive(Serialize)]
struct ProblemReportsView {
    choice: ProblemReportChoice,
    is_sending_set_up: bool,
    waiting: Vec<WaitingReportView>,
    sent: Vec<SentProblemReport>,
}

#[derive(Serialize)]
struct ReportAnswer {
    status: &'static str,
    message: String,
}

pub(crate) async fn list_problem_reports(
    State(application_state): State<ApplicationState>,
) -> Response {
    let Some(problem_report_desk) = &application_state.problem_report_desk else {
        return (StatusCode::NOT_FOUND, "problem reports are unavailable").into_response();
    };
    let mut waiting = Vec::new();
    for waiting_report in problem_report_desk.waiting() {
        waiting.push(WaitingReportView {
            report_number: waiting_report.report_number,
            readable_lines: waiting_report.report.readable_lines(),
        });
    }
    Json(ProblemReportsView {
        choice: application_state.problem_report_choice(),
        is_sending_set_up: problem_report_desk.is_sending_set_up(),
        waiting,
        sent: problem_report_desk.sent_reports(),
    })
    .into_response()
}

/// What the person typed, read from an optional JSON body. A missing or
/// unreadable body is no note: the note never blocks a report.
fn typed_note(request_body: &[u8]) -> String {
    let Ok(body) = serde_json::from_slice::<serde_json::Value>(request_body) else {
        return String::new();
    };
    body["note"].as_str().unwrap_or_default().to_owned()
}

pub(crate) async fn start_person_report(
    State(application_state): State<ApplicationState>,
) -> Response {
    let Some(problem_report_desk) = &application_state.problem_report_desk else {
        return (StatusCode::NOT_FOUND, "problem reports are unavailable").into_response();
    };
    let ready_model_id = application_state
        .generation_executor
        .worker_health_snapshot()
        .ready_model_id;
    let Some(person_report) = crate::problem_reports::person_report(ready_model_id.as_deref())
    else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "problem reports are unavailable",
        )
            .into_response();
    };
    let waiting_report = problem_report_desk.offer_from_person(person_report);
    Json(WaitingReportView {
        report_number: waiting_report.report_number,
        readable_lines: waiting_report.report.readable_lines(),
    })
    .into_response()
}

pub(crate) async fn send_problem_report(
    State(application_state): State<ApplicationState>,
    Path(report_number): Path<u64>,
    request_body: axum::body::Bytes,
) -> Response {
    let Some(problem_report_desk) = &application_state.problem_report_desk else {
        return (StatusCode::NOT_FOUND, "problem reports are unavailable").into_response();
    };
    let send_outcome = problem_report_desk
        .send_waiting_with_note(report_number, &typed_note(&request_body))
        .await;
    let (status_code, status, message) = match send_outcome {
        ReportSendOutcome::Sent => (StatusCode::OK, "sent", "Report sent. Thank you.".to_owned()),
        ReportSendOutcome::NotWaiting => (
            StatusCode::NOT_FOUND,
            "not_waiting",
            "That report was already answered.".to_owned(),
        ),
        ReportSendOutcome::NotSetUp => (
            StatusCode::SERVICE_UNAVAILABLE,
            "not_set_up",
            "Sending reports is not set up in this version yet. Nothing was sent.".to_owned(),
        ),
        ReportSendOutcome::Failed(send_error) => {
            tracing::warn!(error = %send_error, "a problem report did not go");
            (
                StatusCode::BAD_GATEWAY,
                "failed",
                "The report couldn't be sent. Check the internet connection and try again."
                    .to_owned(),
            )
        }
    };
    (status_code, Json(ReportAnswer { status, message })).into_response()
}

pub(crate) async fn decline_problem_report(
    State(application_state): State<ApplicationState>,
    Path(report_number): Path<u64>,
) -> Response {
    let Some(problem_report_desk) = &application_state.problem_report_desk else {
        return (StatusCode::NOT_FOUND, "problem reports are unavailable").into_response();
    };
    if problem_report_desk.decline(report_number) {
        let answer = ReportAnswer {
            status: "declined",
            message: "Nothing was sent.".to_owned(),
        };
        return (StatusCode::OK, Json(answer)).into_response();
    }
    let answer = ReportAnswer {
        status: "not_waiting",
        message: "That report was already answered.".to_owned(),
    };
    (StatusCode::NOT_FOUND, Json(answer)).into_response()
}
