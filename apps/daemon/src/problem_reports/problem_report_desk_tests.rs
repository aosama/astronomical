//! The desk against a stand-in receiver on loopback that records every
//! envelope it is given, so the whole path from offer to log is exercised
//! without anything leaving the Mac.

use std::sync::{Arc, Mutex};

use axum::{Router, extract::State, http::StatusCode, routing::post};
use config::ProblemReportChoice;

use super::{ProblemReportDesk, ReportSendOutcome};
use crate::problem_reports::{MacDescription, ProblemKind, ProblemReport, ReportReceiver};

#[derive(Clone)]
struct StandInReceiver {
    received_envelopes: Arc<Mutex<Vec<String>>>,
    answer_status: StatusCode,
}

async fn receive_envelope(
    State(stand_in): State<StandInReceiver>,
    envelope_body: String,
) -> StatusCode {
    stand_in
        .received_envelopes
        .lock()
        .expect("envelope list lock")
        .push(envelope_body);
    stand_in.answer_status
}

/// Starts a receiver that answers every envelope with `answer_status`, and
/// returns its DSN and the envelopes it receives.
async fn serve_stand_in_receiver(answer_status: StatusCode) -> (String, Arc<Mutex<Vec<String>>>) {
    let received_envelopes = Arc::new(Mutex::new(Vec::new()));
    let stand_in = StandInReceiver {
        received_envelopes: Arc::clone(&received_envelopes),
        answer_status,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback port should bind");
    let receiver_address = listener.local_addr().expect("bound address");
    let application = Router::new()
        .route("/api/7/envelope/", post(receive_envelope))
        .with_state(stand_in);
    tokio::spawn(async move {
        let _ = axum::serve(listener, application).await;
    });
    (
        format!("http://stand-in-key@{receiver_address}/7"),
        received_envelopes,
    )
}

fn report_of_kind(problem: ProblemKind) -> ProblemReport {
    ProblemReport {
        app_version: "0.2.126".to_owned(),
        mac: MacDescription {
            chip_name: "Apple M5 Pro".to_owned(),
            memory_gb: 48,
            macos_version: "27.2".to_owned(),
        },
        model: None,
        problem,
        memory: None,
    }
}

fn desk_for(state_directory: &std::path::Path, sentry_dsn: Option<&str>) -> ProblemReportDesk {
    ProblemReportDesk::new(
        state_directory,
        sentry_dsn.and_then(ReportReceiver::from_sentry_dsn),
    )
}

#[tokio::test]
async fn should_send_a_waiting_report_once_and_log_exactly_what_the_person_read() {
    let (sentry_dsn, received_envelopes) = serve_stand_in_receiver(StatusCode::OK).await;
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let desk = desk_for(state_directory.path(), Some(&sentry_dsn));
    desk.offer(
        report_of_kind(ProblemKind::ModelLoadFailed),
        ProblemReportChoice::Ask,
    )
    .await;
    let report_number = desk.waiting()[0].report_number;
    assert!(
        received_envelopes.lock().expect("lock").is_empty(),
        "asking must not send"
    );
    assert_eq!(
        desk.send_waiting(report_number).await,
        ReportSendOutcome::Sent
    );
    assert_eq!(
        desk.send_waiting(report_number).await,
        ReportSendOutcome::NotWaiting
    );
    assert_eq!(received_envelopes.lock().expect("lock").len(), 1);
    let sent_reports = desk.sent_reports();
    assert_eq!(sent_reports.len(), 1);
    assert_eq!(
        sent_reports[0].readable_lines,
        report_of_kind(ProblemKind::ModelLoadFailed).readable_lines()
    );
    assert!(desk.waiting().is_empty());
}

#[tokio::test]
async fn should_send_without_asking_only_when_the_person_chose_always() {
    let (sentry_dsn, received_envelopes) = serve_stand_in_receiver(StatusCode::OK).await;
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let desk = desk_for(state_directory.path(), Some(&sentry_dsn));
    desk.offer(
        report_of_kind(ProblemKind::DownloadFailed),
        ProblemReportChoice::Always,
    )
    .await;
    assert_eq!(received_envelopes.lock().expect("lock").len(), 1);
    assert!(desk.waiting().is_empty());
    desk.offer(
        report_of_kind(ProblemKind::EngineDidNotStart),
        ProblemReportChoice::Never,
    )
    .await;
    assert_eq!(received_envelopes.lock().expect("lock").len(), 1);
    assert!(desk.waiting().is_empty(), "never must not even offer");
}

#[tokio::test]
async fn should_keep_the_report_waiting_when_sending_is_not_set_up() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let desk = desk_for(state_directory.path(), None);
    assert!(!desk.is_sending_set_up());
    desk.offer(
        report_of_kind(ProblemKind::ModelLoadFailed),
        ProblemReportChoice::Ask,
    )
    .await;
    let report_number = desk.waiting()[0].report_number;
    assert_eq!(
        desk.send_waiting(report_number).await,
        ReportSendOutcome::NotSetUp
    );
    assert_eq!(desk.waiting().len(), 1);
    assert!(desk.sent_reports().is_empty());
}

#[tokio::test]
async fn should_keep_the_report_waiting_and_log_nothing_when_the_receiver_refuses_it() {
    let (sentry_dsn, _received_envelopes) =
        serve_stand_in_receiver(StatusCode::TOO_MANY_REQUESTS).await;
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let desk = desk_for(state_directory.path(), Some(&sentry_dsn));
    desk.offer(
        report_of_kind(ProblemKind::ModelLoadFailed),
        ProblemReportChoice::Ask,
    )
    .await;
    let report_number = desk.waiting()[0].report_number;
    assert!(matches!(
        desk.send_waiting(report_number).await,
        ReportSendOutcome::Failed(_)
    ));
    assert_eq!(desk.waiting().len(), 1);
    assert!(desk.sent_reports().is_empty());
}

#[tokio::test]
async fn should_drop_a_declined_report_without_sending_it() {
    let (sentry_dsn, received_envelopes) = serve_stand_in_receiver(StatusCode::OK).await;
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let desk = desk_for(state_directory.path(), Some(&sentry_dsn));
    desk.offer(
        report_of_kind(ProblemKind::ReplyOutOfMemory),
        ProblemReportChoice::Ask,
    )
    .await;
    let report_number = desk.waiting()[0].report_number;
    assert!(desk.decline(report_number));
    assert!(!desk.decline(report_number));
    assert!(received_envelopes.lock().expect("lock").is_empty());
    assert!(desk.sent_reports().is_empty());
}
