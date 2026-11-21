//! Sends the daily count when it is due, and is the one place that decides
//! whether it may be sent at all.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{DailyCountLine, DailyCountState, unix_day_now};
use crate::ResolvedRuntimeConfig;
use crate::problem_reports::{MacDescription, ReportReceiver};

/// How long after start the first look happens, so a launch is never slowed.
const FIRST_LOOK_DELAY: Duration = Duration::from_secs(60);

/// How often the counter looks whether a count is due. The look reads one
/// small file; a short interval means a count goes out soon after the person
/// agrees and soon after midnight UTC, without a timer aimed at either.
const LOOK_INTERVAL: Duration = Duration::from_secs(300);

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DailyCountOutcome {
    /// The person has not agreed (unanswered, or said no).
    NotAgreed,
    AlreadySentToday,
    /// This build has no receiver.
    NotSetUp,
    Sent,
    /// The receiver could not be reached; the next look tries again.
    Failed(String),
}

pub(crate) struct DailyCounter {
    state_directory: PathBuf,
    report_receiver: Option<ReportReceiver>,
    http_client: reqwest::Client,
    line: DailyCountLine,
}

impl DailyCounter {
    pub(crate) fn new(
        state_directory: &Path,
        report_receiver: Option<ReportReceiver>,
        line: DailyCountLine,
    ) -> Self {
        // reqwest is built without a TLS provider of its own; see
        // `install_default_tls_crypto_provider`.
        crate::library::crypto_provider::install_default_tls_crypto_provider();
        Self {
            state_directory: state_directory.to_path_buf(),
            report_receiver,
            http_client: reqwest::Client::new(),
            line,
        }
    }

    /// The counter for this Mac and this build, or `None` when the Mac
    /// cannot be described, in which case no count is offered or sent.
    pub(crate) fn for_this_mac(state_directory: &Path) -> Option<Self> {
        let line = DailyCountLine {
            app_version: env!("CARGO_PKG_VERSION").to_owned(),
            mac: MacDescription::read_this_mac()?,
        };
        Some(Self::new(
            state_directory,
            ReportReceiver::compiled_in(),
            line,
        ))
    }

    pub(crate) fn line(&self) -> &DailyCountLine {
        &self.line
    }

    pub(crate) fn state(&self, today_unix_day: u64) -> DailyCountState {
        DailyCountState::load_or_start(&self.state_directory, today_unix_day)
    }

    /// Sends today's count if the person agreed and none went today.
    pub(crate) async fn send_if_due(
        &self,
        daily_count_answer: Option<bool>,
        today_unix_day: u64,
    ) -> DailyCountOutcome {
        let mut daily_count_state = self.state(today_unix_day);
        if daily_count_answer != Some(true) {
            return DailyCountOutcome::NotAgreed;
        }
        if daily_count_state.is_sent_today(today_unix_day) {
            return DailyCountOutcome::AlreadySentToday;
        }
        let Some(report_receiver) = &self.report_receiver else {
            return DailyCountOutcome::NotSetUp;
        };
        let sent_at_unix_seconds = unix_seconds_now();
        let envelope_body = self
            .line
            .envelope_body(&new_event_id(), sent_at_unix_seconds);
        if let Err(send_error) = report_receiver.send(&self.http_client, envelope_body).await {
            return DailyCountOutcome::Failed(send_error);
        }
        daily_count_state.last_sent_unix_day = Some(today_unix_day);
        if let Err(save_error) = daily_count_state.save(&self.state_directory) {
            tracing::warn!(error = %save_error, "the daily count's sent day could not be saved");
        }
        DailyCountOutcome::Sent
    }
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_secs())
        .unwrap_or_default()
}

/// A 32-hex-digit event id from the clock: unique per event, and nothing
/// about the Mac, so two days' counts cannot be linked through it.
fn new_event_id() -> String {
    let nanoseconds_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_nanos())
        .unwrap_or_default();
    format!("{nanoseconds_since_epoch:032x}")
}

fn daily_count_answer(reloadable_config: &RwLock<ResolvedRuntimeConfig>) -> Option<bool> {
    reloadable_config
        .read()
        .ok()
        .and_then(|resolved_runtime_config| {
            resolved_runtime_config.sharing_settings.daily_count_answer
        })
}

/// Looks every few minutes whether a count is due, for as long as the engine
/// runs. It does nothing in the Mac App Store build, whose privacy
/// declarations say no data is collected (issue #2413).
pub async fn keep_daily_count(
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    instance_paths: config::AstronomicalInstancePaths,
) {
    if instance_paths.keeps_state_in_app_container() {
        return;
    }
    let Some(daily_counter) = DailyCounter::for_this_mac(instance_paths.state_directory()) else {
        tracing::warn!("the daily count is off: this Mac could not be described");
        return;
    };
    tokio::time::sleep(FIRST_LOOK_DELAY).await;
    loop {
        let outcome = daily_counter
            .send_if_due(daily_count_answer(&reloadable_config), unix_day_now())
            .await;
        match outcome {
            DailyCountOutcome::Sent => tracing::info!("the daily count was sent"),
            DailyCountOutcome::Failed(send_error) => {
                tracing::warn!(error = %send_error, "the daily count was not sent")
            }
            DailyCountOutcome::NotAgreed
            | DailyCountOutcome::AlreadySentToday
            | DailyCountOutcome::NotSetUp => {}
        }
        tokio::time::sleep(LOOK_INTERVAL).await;
    }
}

#[cfg(test)]
#[path = "daily_counter_tests.rs"]
mod daily_counter_tests;
