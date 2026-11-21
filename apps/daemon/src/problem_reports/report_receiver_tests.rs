use super::{ReportReceiver, sentry_envelope_body};
use crate::problem_reports::{CatalogModelName, MacDescription, ProblemKind, ProblemReport};

fn load_failure_report() -> ProblemReport {
    ProblemReport {
        app_version: "0.2.126".to_owned(),
        mac: MacDescription {
            chip_name: "Apple M5 Pro".to_owned(),
            memory_gb: 48,
            macos_version: "27.2".to_owned(),
        },
        model: Some(CatalogModelName::Listed(
            "mlx-community/Qwen3.5-2B-4bit".to_owned(),
        )),
        problem: ProblemKind::ModelLoadFailed,
        facts: Vec::new(),
        recent_events: Vec::new(),
        note: None,
    }
}

#[test]
fn should_read_the_envelope_address_from_a_sentry_dsn() {
    let report_receiver =
        ReportReceiver::from_sentry_dsn("https://abc123@o1.ingest.example.com/4567")
            .expect("a well-formed DSN should be read");
    assert_eq!(
        report_receiver.envelope_url(),
        "https://o1.ingest.example.com/api/4567/envelope/"
    );
}

#[test]
fn should_refuse_a_dsn_without_a_key_host_or_numeric_project() {
    for sentry_dsn in [
        "",
        "not a dsn",
        "https://o1.ingest.example.com/4567",
        "https://@o1.ingest.example.com/4567",
        "https://abc123@/4567",
        "https://abc123@o1.ingest.example.com/project",
        "ftp://abc123@o1.ingest.example.com/4567",
    ] {
        assert_eq!(
            ReportReceiver::from_sentry_dsn(sentry_dsn),
            None,
            "{sentry_dsn:?}"
        );
    }
}

#[test]
fn should_send_to_the_compiled_in_project_over_https() {
    let report_receiver =
        ReportReceiver::compiled_in().expect("the compiled-in DSN should be readable");
    assert!(
        report_receiver.envelope_url().starts_with("https://"),
        "reports must never leave the Mac unencrypted"
    );
}

#[test]
fn should_send_only_the_report_fields_in_the_envelope() {
    let envelope_body = sentry_envelope_body(&load_failure_report(), "0123abcd", 1_700_000_000);
    let envelope_lines: Vec<&str> = envelope_body.lines().collect();
    assert_eq!(envelope_lines.len(), 3);
    assert_eq!(envelope_lines[1], r#"{"type":"event"}"#);
    let event: serde_json::Value =
        serde_json::from_str(envelope_lines[2]).expect("the event line should be JSON");
    assert_eq!(event["release"], "astronomical@0.2.126");
    assert_eq!(event["message"]["formatted"], "model load failed");
    assert_eq!(
        event["fingerprint"],
        serde_json::json!(["model_load_failed"])
    );
    assert_eq!(
        event["extra"]["report"]["model"],
        "mlx-community/Qwen3.5-2B-4bit"
    );
    let mut event_keys: Vec<&String> = event
        .as_object()
        .expect("event is an object")
        .keys()
        .collect();
    event_keys.sort();
    assert_eq!(
        event_keys,
        [
            "event_id",
            "extra",
            "fingerprint",
            "level",
            "message",
            "platform",
            "release",
            "sdk",
            "tags",
            "timestamp",
            "user"
        ],
        "no device or breadcrumb fields may be sent"
    );
    assert_eq!(
        event["user"],
        serde_json::json!({"ip_address": "0.0.0.0"}),
        "the only user field is a placeholder address that maps to no place"
    );
    assert_eq!(
        event["sdk"]["settings"]["infer_ip"], "never",
        "Sentry must not work out an address or a city from the connection"
    );
    assert_eq!(
        event["sdk"]["settings"]["infer_ip"], "never",
        "Sentry must not work out an address or a city from the connection"
    );
}
