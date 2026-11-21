//! Where a report goes, and the exact bytes sent. Reports are posted to a
//! Sentry project as one hand-built envelope instead of through Sentry's SDK:
//! the SDK adds device, user and breadcrumb details on its own, while a
//! hand-built event carries the report's fixed fields and nothing else.

use std::time::Duration;

use serde_json::json;

use super::ProblemReport;

/// The receiver file compiled into the app; see its `$comment`.
const RECEIVER_FILE_TEXT: &str = include_str!("../../problem_report_receiver.json");

/// Long enough for a slow connection, short enough that a person pressing
/// Send report is told quickly when it did not go.
const SEND_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReportReceiver {
    envelope_url: String,
    public_key: String,
}

impl ReportReceiver {
    /// The receiver compiled into this build, or `None` while it has none.
    pub(crate) fn compiled_in() -> Option<Self> {
        let receiver_file: serde_json::Value = serde_json::from_str(RECEIVER_FILE_TEXT).ok()?;
        Self::from_sentry_dsn(receiver_file["sentry_dsn"].as_str()?)
    }

    /// Reads a Sentry DSN, `SCHEME://PUBLIC_KEY@HOST/PROJECT_ID`.
    pub(crate) fn from_sentry_dsn(sentry_dsn: &str) -> Option<Self> {
        let (scheme, address) = sentry_dsn.split_once("://")?;
        let (public_key, host_and_project) = address.split_once('@')?;
        let (host, project_id) = host_and_project.rsplit_once('/')?;
        let has_valid_parts = matches!(scheme, "https" | "http")
            && !public_key.is_empty()
            && !host.is_empty()
            && !project_id.is_empty()
            && project_id
                .bytes()
                .all(|project_id_byte| project_id_byte.is_ascii_digit());
        has_valid_parts.then(|| Self {
            envelope_url: format!("{scheme}://{host}/api/{project_id}/envelope/"),
            public_key: public_key.to_owned(),
        })
    }

    #[cfg(test)]
    pub(crate) fn envelope_url(&self) -> &str {
        &self.envelope_url
    }

    /// Posts `envelope_body` and succeeds only on a 2xx answer.
    pub(crate) async fn send(
        &self,
        http_client: &reqwest::Client,
        envelope_body: String,
    ) -> Result<(), String> {
        let receiver_answer = http_client
            .post(&self.envelope_url)
            .timeout(SEND_TIMEOUT)
            .header("Content-Type", "application/x-sentry-envelope")
            .header(
                "X-Sentry-Auth",
                format!(
                    "Sentry sentry_version=7, sentry_key={}, sentry_client=astronomical/1",
                    self.public_key
                ),
            )
            .body(envelope_body)
            .send()
            .await
            .map_err(|send_error| format!("the report could not be sent: {send_error}"))?;
        if receiver_answer.status().is_success() {
            return Ok(());
        }
        Err(format!(
            "the receiver answered {}",
            receiver_answer.status()
        ))
    }
}

/// The envelope for one report: a header line, an item header line and the
/// event. Reports group by problem kind, and the report's fields travel as
/// the event's tags and its `report` extra, so the developers see exactly
/// what the person saw.
pub(crate) fn sentry_envelope_body(
    report: &ProblemReport,
    event_id: &str,
    sent_at_unix_seconds: u64,
) -> String {
    let problem_name = serde_json::to_value(report.problem).unwrap_or_default();
    let event = json!({
        "event_id": event_id,
        "timestamp": sent_at_unix_seconds,
        "platform": "other",
        "level": "error",
        "release": format!("astronomical@{}", report.app_version),
        "message": {"formatted": report.problem.plain_description()},
        "fingerprint": [problem_name],
        "tags": {
            "problem": problem_name,
            "chip": report.mac.chip_name,
            "memory_gb": report.mac.memory_gb.to_string(),
            "macos": report.mac.macos_version,
        },
        "extra": {"report": report},
    });
    let envelope_header =
        json!({"event_id": event_id, "sent_at_unix_seconds": sent_at_unix_seconds});
    let item_header = json!({"type": "event"});
    format!("{envelope_header}\n{item_header}\n{event}\n")
}

#[cfg(test)]
#[path = "report_receiver_tests.rs"]
mod report_receiver_tests;
