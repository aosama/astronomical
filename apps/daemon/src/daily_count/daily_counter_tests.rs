//! The counter against a stand-in receiver on loopback, so the rule "nothing
//! is sent unless the person agreed" is checked on the wire.

use std::sync::{Arc, Mutex};

use axum::{Router, extract::State, http::StatusCode, routing::post};

use super::{DailyCountOutcome, DailyCounter};
use crate::daily_count::DailyCountLine;
use crate::problem_reports::{MacDescription, ReportReceiver};

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
        .expect("lock")
        .push(envelope_body);
    stand_in.answer_status
}

async fn counter_with_stand_in(
    state_directory: &std::path::Path,
    answer_status: StatusCode,
) -> (DailyCounter, Arc<Mutex<Vec<String>>>) {
    let received_envelopes = Arc::new(Mutex::new(Vec::new()));
    let stand_in = StandInReceiver {
        received_envelopes: Arc::clone(&received_envelopes),
        answer_status,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback port");
    let receiver_address = listener.local_addr().expect("bound address");
    let application = Router::new()
        .route("/api/7/envelope/", post(receive_envelope))
        .with_state(stand_in);
    tokio::spawn(async move {
        let _ = axum::serve(listener, application).await;
    });
    let report_receiver =
        ReportReceiver::from_sentry_dsn(&format!("http://stand-in@{receiver_address}/7"));
    let line = DailyCountLine {
        app_version: "0.2.127".to_owned(),
        mac: MacDescription {
            chip_name: "Apple M5 Pro".to_owned(),
            memory_gb: 48,
            macos_version: "27.2".to_owned(),
        },
    };
    (
        DailyCounter::new(state_directory, report_receiver, line),
        received_envelopes,
    )
}

#[tokio::test]
async fn should_send_nothing_while_unanswered_or_after_no_thanks() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let (daily_counter, received_envelopes) =
        counter_with_stand_in(state_directory.path(), StatusCode::OK).await;
    assert_eq!(
        daily_counter.send_if_due(None, 20_001).await,
        DailyCountOutcome::NotAgreed
    );
    assert_eq!(
        daily_counter.send_if_due(Some(false), 20_001).await,
        DailyCountOutcome::NotAgreed
    );
    assert!(received_envelopes.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn should_send_once_a_day_after_the_person_agreed() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let (daily_counter, received_envelopes) =
        counter_with_stand_in(state_directory.path(), StatusCode::OK).await;
    assert_eq!(
        daily_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::Sent
    );
    assert_eq!(
        daily_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::AlreadySentToday
    );
    assert_eq!(received_envelopes.lock().expect("lock").len(), 1);
    assert_eq!(
        daily_counter.send_if_due(Some(true), 20_002).await,
        DailyCountOutcome::Sent
    );
    assert_eq!(received_envelopes.lock().expect("lock").len(), 2);
    assert_eq!(daily_counter.state(20_002).last_sent_unix_day, Some(20_002));
}

#[tokio::test]
async fn should_not_send_again_after_a_restart_on_the_same_day() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let (first_counter, _) = counter_with_stand_in(state_directory.path(), StatusCode::OK).await;
    assert_eq!(
        first_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::Sent
    );
    let (restarted_counter, received_after_restart) =
        counter_with_stand_in(state_directory.path(), StatusCode::OK).await;
    assert_eq!(
        restarted_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::AlreadySentToday
    );
    assert!(received_after_restart.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn should_try_again_later_when_the_receiver_refuses_the_count() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let (daily_counter, _) =
        counter_with_stand_in(state_directory.path(), StatusCode::SERVICE_UNAVAILABLE).await;
    assert!(matches!(
        daily_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::Failed(_)
    ));
    assert_eq!(
        daily_counter.state(20_001).last_sent_unix_day,
        None,
        "a failed send is not a sent day"
    );
}

#[tokio::test]
async fn should_send_nothing_from_a_development_instance_even_after_the_person_agreed() {
    let home_directory = tempfile::tempdir().expect("temporary folder");
    let development_paths = config::AstronomicalInstancePaths::for_home_directory(
        home_directory.path(),
        config::AstronomicalRuntimeInstance::Development,
    );
    let daily_counter =
        DailyCounter::for_this_mac(&development_paths).expect("this Mac can be described");

    assert_eq!(
        daily_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::NotSetUp
    );
    assert_eq!(
        daily_counter.state(20_001).last_sent_unix_day,
        None,
        "nothing was sent, so no day is marked sent"
    );
}

#[tokio::test]
async fn should_stop_sending_once_the_switch_is_turned_off() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let (daily_counter, received_envelopes) =
        counter_with_stand_in(state_directory.path(), StatusCode::OK).await;
    assert_eq!(
        daily_counter.send_if_due(Some(true), 20_001).await,
        DailyCountOutcome::Sent
    );
    assert_eq!(
        daily_counter.send_if_due(Some(false), 20_002).await,
        DailyCountOutcome::NotAgreed
    );
    assert_eq!(received_envelopes.lock().expect("lock").len(), 1);
}
