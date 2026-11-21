//! The one place a report's fate is decided: offered to the person, sent, or
//! declined, according to their choice. Sending is only reachable through
//! here, so the person's choice cannot be skipped by any caller.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use config::ProblemReportChoice;

use super::{
    ProblemReport, ReportReceiver, SentProblemReport, SentReportLog, WaitingProblemReport,
    WaitingReports, sentry_envelope_body,
};

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReportSendOutcome {
    Sent,
    /// No report waits under that number; it was already answered.
    NotWaiting,
    /// This build has no receiver yet; the report keeps waiting.
    NotSetUp,
    /// The receiver could not be reached; the report keeps waiting.
    Failed(String),
}

pub(crate) struct ProblemReportDesk {
    waiting_reports: WaitingReports,
    sent_report_log: SentReportLog,
    report_receiver: Option<ReportReceiver>,
    http_client: reqwest::Client,
}

impl ProblemReportDesk {
    pub(crate) fn new(state_directory: &Path, report_receiver: Option<ReportReceiver>) -> Self {
        // reqwest is built without a TLS provider of its own; see
        // `install_default_tls_crypto_provider`.
        crate::library::crypto_provider::install_default_tls_crypto_provider();
        Self {
            waiting_reports: WaitingReports::default(),
            sent_report_log: SentReportLog::in_state_directory(state_directory),
            report_receiver,
            http_client: reqwest::Client::new(),
        }
    }

    pub(crate) fn is_sending_set_up(&self) -> bool {
        self.report_receiver.is_some()
    }

    pub(crate) fn waiting(&self) -> Vec<WaitingProblemReport> {
        self.waiting_reports.waiting()
    }

    pub(crate) fn sent_reports(&self) -> Vec<SentProblemReport> {
        self.sent_report_log.sent_reports()
    }

    /// Handles a new report by the person's choice: waits for them when
    /// asking, sends at once when they chose always, and drops it when they
    /// chose never.
    pub(crate) async fn offer(&self, report: ProblemReport, choice: ProblemReportChoice) {
        match choice {
            ProblemReportChoice::Never => {}
            ProblemReportChoice::Ask => {
                self.waiting_reports.offer(report);
            }
            ProblemReportChoice::Always => {
                let report_number = self.waiting_reports.offer(report);
                let send_outcome = self.send_waiting(report_number).await;
                if send_outcome != ReportSendOutcome::Sent {
                    tracing::warn!(
                        ?send_outcome,
                        "a report the person chose to always send did not go"
                    );
                }
            }
        }
    }

    /// Sends the report waiting under `report_number` after the person
    /// pressed Send report, and records it. A report that did not go keeps
    /// waiting, so the person can try again.
    pub(crate) async fn send_waiting(&self, report_number: u64) -> ReportSendOutcome {
        let Some(report) = self.waiting_reports.take(report_number) else {
            return ReportSendOutcome::NotWaiting;
        };
        let Some(report_receiver) = &self.report_receiver else {
            self.waiting_reports.offer(report);
            return ReportSendOutcome::NotSetUp;
        };
        let sent_at_unix_seconds = unix_seconds_now();
        let envelope_body = sentry_envelope_body(&report, &new_event_id(), sent_at_unix_seconds);
        if let Err(send_error) = report_receiver.send(&self.http_client, envelope_body).await {
            self.waiting_reports.offer(report);
            return ReportSendOutcome::Failed(send_error);
        }
        let sent_report = SentProblemReport {
            sent_at_unix_seconds,
            readable_lines: report.readable_lines(),
        };
        if let Err(log_error) = self.sent_report_log.record(&sent_report) {
            tracing::warn!(error = %log_error, "a sent report could not be added to the sent-reports log");
        }
        ReportSendOutcome::Sent
    }

    /// Drops the report the person chose not to send. Returns whether one
    /// was waiting under that number.
    pub(crate) fn decline(&self, report_number: u64) -> bool {
        self.waiting_reports.take(report_number).is_some()
    }
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_secs())
        .unwrap_or_default()
}

/// A 32-hex-digit event id. It only has to be unique per report; the clock in
/// nanoseconds is, and it says nothing about the Mac or the person.
fn new_event_id() -> String {
    let nanoseconds_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_nanos())
        .unwrap_or_default();
    format!("{nanoseconds_since_epoch:032x}")
}

#[cfg(test)]
#[path = "problem_report_desk_tests.rs"]
mod problem_report_desk_tests;
